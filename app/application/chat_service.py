import difflib
import logging
import re
from typing import Dict, Any, Optional, List

from gigachat.models import Messages, MessagesRole

from app.domain.models import RawAction, HistoryItem, ChatRequest, ChatResponse
from app.ports.llm import LLMPort
from app.ports.knowledge_base import KnowledgeBasePort

logger = logging.getLogger("SiteOfSites_AI")

def mask_pii(text: str) -> str:
    text = re.sub(r'\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b', '[PHONE_MASKED]', text)
    text = re.sub(r'\b(?:\d[ -]*?){13,16}\b', '[CARD_MASKED]', text)
    text = re.sub(r'\b\d{4}\s+\d{6}\b', '[PASSPORT_MASKED]', text)
    text = re.sub(r'(?i)(api[_-]?key|secret|token|password|пароль)\s*[:=]\s*\S+', r'\1: [SENSITIVE_MASKED]', text)
    return text

def extract_json_from_llm(reply_text: str) -> Optional[RawAction]:
    text = reply_text.strip()
    try:
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            json_str = match.group(0)
            return RawAction.model_validate_json(json_str) 
        return None
    except Exception as e:
        logger.warning(f"Failed to parse JSON from LLM response: {e}")
        return None

class ChatService:
    def __init__(self, llm_adapter: LLMPort, kb_adapter: KnowledgeBasePort):
        self.llm_adapter = llm_adapter
        self.kb_adapter = kb_adapter

    def _get_system_instruction(self, knowledge_context_str: str, is_authenticated: bool) -> str:
        # Определяем статус для нейросети
        auth_status = "АВТОРИЗОВАН" if is_authenticated else "АНОНИМ (НЕ ВХОДИЛ В АККАУНТ)"
        
        return f"""
Ты — эксклюзивный интеллектуальный помощник службы поддержки платформы "SiteOfSites". Твоя задача — помочь пользователю, отвечая на вопросы и распознавая его намерения для выполнения действий.

Твои основные задачи:
1. Предоставлять точные и полезные ответы на вопросы пользователей, используя только предоставленную базу знаний.
2. Распознавать явные намерения пользователя для выполнения транзакционных действий.

Твоя личность и ограничения (СТРОГО СОБЛЮДАТЬ):
1. НИКОГДА не упоминай, что ты GigaChat, создан Сбером, OpenAI или Яндексом. Отвечай: "Я разработан технической командой SiteOfSites".
2. КАТЕГОРИЧЕСКИ ЗАПРЕЩЕНО отвечать на вопросы, не связанные с SiteOfSites.
3. Используй ТОЛЬКО базу знаний. Если информации нет, скажи об этом.
4. ПРАВИЛО УТОЧНЕНИЯ ДАННЫХ: Если пользователь хочет сменить ЛОГИН (почту) или ИМЯ (никнейм), НО еще НЕ указал на какое именно, НЕ генерируй JSON с интентами change_email/change_username. Просто ответь обычным текстом и спроси: "На какую почту/имя вы хотите изменить?".
5. РАЗГРАНИЧЕНИЕ ПОЛЕЙ: ЛОГИН — это всегда Email (содержит символ @). ИМЯ — это никнейм или имя пользователя (НЕ содержит @). Если пользователь дает почту, это всегда change_email.
6. КАТЕГОРИЧЕСКИ ЗАПРЕЩЕНО запрашивать пароли, номера карт и т.д.

--- ВАЖНО: СТАТУС ПОЛЬЗОВАТЕЛЯ ---
Статус пользователя в данный момент: {auth_status}
Если статус "АНОНИМ", ты КАТЕГОРИЧЕСКИ НЕ ИМЕЕШЬ ПРАВА возвращать JSON-интенты. Если АНОНИМ просит сбросить пароль, отменить подписку или изменить профиль, ответь обычным текстом: "Для выполнения этого действия вам необходимо сначала войти в свой аккаунт".

КРИТИЧЕСКИ ВАЖНО (ТРАНЗАКЦИОННЫЙ РЕЖИМ):
ЕСЛИ пользователь АВТОРИЗОВАН и хочет выполнить действие (или спрашивает, как его выполнить), верни ТОЛЬКО JSON объект (без лишнего текста). Для действий отмены подписки и смены пароля дополнительные данные от пользователя НЕ требуются, сразу возвращай JSON.
Поле intent должно строго содержать ОДНО из значений:
- "change_username" (смена имени пользователя/ника, ОБЯЗАТЕЛЬНО должен быть параметр: new_username)
- "change_email" (смена почты/логина, ОБЯЗАТЕЛЬНО должен быть параметр: new_email)
- "cancel_premium" (отмена подписки)
- "reset_password" (если просят изменить, сбросить или обновить пароль, а также если спрашивают, КАК это сделать)
- "general_query" (для любых других вопросов или если нужны уточнения)

Формат JSON (если выполняем действие):
{{
  "intent": "выбранный_интент",
  "parameters": {{ "new_email": "example@email.com" }},
  "reply_text": "Ок, сейчас сделаем."
}}

Если это просто вопрос или уточнение данных — отвечай обычным текстом без JSON!

БАЗА ЗНАНИЙ:
{knowledge_context_str}
"""

    def _get_fast_reply(self, message: str, kb_dict: Dict[str, Any]) -> Optional[str]:
        msg_lower = message.strip().lower()
        clean_msg = re.sub(r'[^\w\s]', '', msg_lower)
        
        greetings = ["привет", "здравствуйте", "добрый день", "доброе утро", "добрый вечер", "хай", "hello", "hi", "приветствую"]
        if any(greet == clean_msg or clean_msg.startswith(greet + " ") for greet in greetings):
            return "Здравствуйте! Я AI-помощник SiteOfSites. Чем могу помочь с хостингом или настройкой профиля?"

        faqs = kb_dict.get("faq", {})
        if not faqs:
            return None
            
        for q, a in faqs.items():
            if q.lower() == msg_lower:
                return a
                
        matches = difflib.get_close_matches(message, faqs.keys(), n=1, cutoff=0.75)
        if matches:
            return faqs[matches[0]]
                
        return None

    async def process_chat_request(self, request: ChatRequest) -> ChatResponse:
        _, knowledge_context_dict = self.kb_adapter.get_knowledge_base()

        fast_reply = self._get_fast_reply(request.message, knowledge_context_dict)
        if fast_reply:
            return ChatResponse(reply=fast_reply, status="success", source="cache")

        relevant_chunks = self.kb_adapter.search_relevant_chunks(request.message, top_k=4)
        knowledge_context_str = "\n\n".join(relevant_chunks) if relevant_chunks else "Нет релевантной информации в базе знаний."

        # Передаем статус авторизации в промпт
        system_instruction = self._get_system_instruction(knowledge_context_str, request.is_authenticated)
        messages_payload = [Messages(role=MessagesRole.SYSTEM, content=system_instruction.strip())]

        for msg in request.history[-6:]:
            masked_content = mask_pii(msg.content)
            messages_payload.append(Messages(role=msg.role, content=masked_content))
            
        masked_user_message = mask_pii(request.message)
        messages_payload.append(Messages(role=MessagesRole.USER, content=masked_user_message))

        reply_text = await self.llm_adapter.complete(
            messages=messages_payload,
            temperature=0.1,
            max_tokens=1000
        )
        
        structured_data = extract_json_from_llm(reply_text)
        
        if structured_data:
            try:
                validated_params = structured_data.validate_params()
                return ChatResponse(
                    reply=structured_data.reply_text or reply_text,
                    intent=structured_data.intent,
                    parameters=validated_params,
                    status="success",
                )
            except ValueError as e:
                logger.warning(f"Parameter validation failed for LLM intent: {e}")
                safe_reply = "Пожалуйста, уточните данные для выполнения этого действия."
                if structured_data.reply_text:
                    safe_reply = structured_data.reply_text
                return ChatResponse(reply=safe_reply, status="success", source="llm")
            
        return ChatResponse(reply=reply_text, status="success", source="llm")