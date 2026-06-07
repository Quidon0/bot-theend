import json
import os
import difflib
import logging
import re
from typing import Optional, Dict, Any, Tuple
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from gigachat import GigaChat

# --- Configuration & Logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("SiteOfSites_AI")

# Prefer environment variable for security, with a fallback for local development if necessary.
GIGACHAT_CREDENTIALS = os.environ.get(
    "GIGACHAT_CREDENTIALS",
    "MDE5YzQ3MzAtM2EwYS03MWJiLWE0NzctNGE1NmU3ZDg5MTE0Ojc3OWRkNmJmLTkyZmItNDZkNC1hNTQ4LTAzYzU2YzEyZTY5Mw=="
)
VERIFY_SSL = os.environ.get("GIGACHAT_VERIFY_SSL", "False").lower() in ("true", "1", "yes")

# --- Models ---
class TransactionalAction(BaseModel):
    """Schema for validating structured actions from LLM"""
    action: str
    parameters: Dict[str, Any] = {}
    message: str = ""

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=1500)
    user_id: Optional[str] = None
    history: list = []

class ChatResponse(BaseModel):
    reply: str
    status: str
    source: str
    action: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None

# --- Knowledge Base Manager ---
class KnowledgeBaseManager:
    """Manages the loading and caching of the knowledge base file."""
    def __init__(self, filepath: str = "knowledge_base.json"):
        self.filepath = filepath
        self._cache_str: str = "{}"
        self._cache_dict: Dict[str, Any] = {}
        self._last_mtime: float = 0.0

    def get_knowledge_base(self) -> Tuple[str, Dict[str, Any]]:
        """Loads and caches the knowledge base, reloading only if the file changes."""
        if not os.path.exists(self.filepath):
            logger.warning(f"Knowledge base file not found at {self.filepath}")
            return self._cache_str, self._cache_dict
            
        try:
            current_mtime = os.path.getmtime(self.filepath)
            if current_mtime > self._last_mtime:
                with open(self.filepath, "r", encoding="utf-8") as f:
                    self._cache_str = f.read()
                    self._cache_dict = json.loads(self._cache_str)
                    self._last_mtime = current_mtime
                logger.info(f"Knowledge base updated from {self.filepath}")
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in knowledge base: {e}")
        except Exception as e:
            logger.error(f"Error reading knowledge base: {e}")
            
        return self._cache_str, self._cache_dict

kb_manager = KnowledgeBaseManager()

# --- State ---
ml_models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize GigaChat client on startup
    logger.info("Initializing GigaChat client...")
    try:
        ml_models["giga_client"] = GigaChat(credentials=GIGACHAT_CREDENTIALS, verify_ssl_certs=VERIFY_SSL)
        logger.info("GigaChat client initialized successfully.")
    except Exception as e:
        logger.error(f"Failed to initialize GigaChat client: {e}")
    
    # Preload KB
    kb_manager.get_knowledge_base()
    
    yield
    
    # Cleanup
    if "giga_client" in ml_models:
        logger.info("Shutting down AI service.")
        ml_models.clear()

app = FastAPI(
    title="SiteOfSites AI Support",
    description="Microservice for handling AI interactions via GigaChat",
    version="1.0.0",
    lifespan=lifespan
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Services ---
def get_fast_reply(message: str, kb_dict: Dict[str, Any]) -> Optional[str]:
    """Provides instant replies for simple greetings and known FAQs without LLM invocation."""
    msg_lower = message.strip().lower()
    
    greetings = {"привет", "здравствуйте", "добрый день", "доброе утро", "добрый вечер", "хай", "hello", "hi"}
    if msg_lower in greetings:
        return "Здравствуйте! Я AI-помощник SiteOfSites. Чем могу помочь с хостингом или настройкой профиля?"

    faqs = kb_dict.get("faq", {})
    if not faqs:
        return None
        
    # Exact match
    for q, a in faqs.items():
        if q.lower() == msg_lower:
            return a
            
    # Fuzzy match
    matches = difflib.get_close_matches(message, faqs.keys(), n=1, cutoff=0.75)
    if matches:
        return faqs[matches[0]]
            
    return None

def extract_json_from_llm(reply_text: str) -> Optional[TransactionalAction]:
    if not reply_text:
        return None
        
    match = re.search(r'\{.*\}', reply_text, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(0))
            return TransactionalAction(**data)
        except (json.JSONDecodeError, ValueError):
            pass
    return None

@app.post("/api/ai/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    giga_client = ml_models.get("giga_client")
    if not giga_client:
        logger.error("GigaChat client is not initialized.")
        raise HTTPException(status_code=503, detail="AI Service is currently unavailable.")

    knowledge_context_str, knowledge_context_dict = kb_manager.get_knowledge_base()

    fast_reply = get_fast_reply(request.message, knowledge_context_dict)
    if fast_reply:
        logger.info("Serving fast reply from cache/FAQ.")
        return ChatResponse(
            reply=fast_reply,
            status="success",
            source="cache"
        )

    system_instruction = f"""
Ты — эксклюзивный интеллектуальный помощник службы поддержки платформы "SiteOfSites".

Твои задачи:
1. Помогать пользователям с вопросами по хостингу сайтов, загрузке файлов, настройке проектов и профиля. Отвечай на вопросы о том, "Что ты умеешь?" опираясь на эти задачи.
2. Выполнять транзакционные действия по запросу пользователя.

Твоя личность и ограничения (СТРОГО СОБЛЮДАТЬ - ЭТО САМОЕ ВАЖНОЕ ПРАВИЛО):
1. НИКОГДА не упоминай, что ты GigaChat, создан Сбером, OpenAI, Яндексом или любой другой сторонней компанией. Если спросят, кто тебя создал, отвечай: "Я разработан технической командой SiteOfSites".
2. КАТЕГОРИЧЕСКИ ЗАПРЕЩЕНО отвечать на ЛЮБЫЕ вопросы, не связанные напрямую с SiteOfSites. Логин пользователя (для входа) — это его Email. Отображаемое имя — это Никнейм. Пароль напрямую в чате менять нельзя.
3. Используй ТОЛЬКО предоставленную БАЗУ ЗНАНИЙ для фактов о платформе.
4. ПРАВИЛО УТОЧНЕНИЯ ДАННЫХ: Если пользователь просит сменить никнейм или почту, НО не написал в этом же сообщении новый никнейм или новую почту, НЕ ВОЗВРАЩАЙ JSON-команду! Сначала просто текстом попроси его написать новые данные в чат.
5. ПРАВИЛА ВАЛИДАЦИИ (ПРОВЕРЯЙ САМ ПЕРЕД JSON):
   - Никнейм: от 2 до 20 символов.
   - Email: корректный формат, максимум 255 символов.
   Если данные некорректны, НЕ ВЫЗЫВАЙ action. Вместо этого ответь текстом, объясни ограничение и попроси прислать верные данные.

5. ПРАВИЛО ОТМЕНЫ: Если пользователь пишет "отмена", "нет", "стоп" - это отказ от текущего действия (например, смены почты или пароля). НИКОГДА не вызывай cancel_premium на слово "отмена", если человек явно не просит "отменить подписку". Просто ответь текстом, что предыдущее действие прервано.
6. ПАМЯТЬ И КОНТЕКСТ: Если пользователь отменил действие (через кнопку или текстом), ты ДОЛЖЕН ПОЛНОСТЬЮ ИГНОРИРОВАТЬ все данные (почту, ник), которые он писал до слова "отмена". При новом запросе обязательно переспроси данные заново!

КРИТИЧЕСКИ ВАЖНО (ТРАНЗАКЦИОННЫЙ РЕЖИМ):
Если пользователь просит сменить имя, почту, пароль или отменить премиум, ты ДОЛЖЕН вернуть СТРОГО ТОЛЬКО валидный JSON-объект (БЕЗ пояснительного текста, БЕЗ Markdown блоков). Текст подтверждения пиши ВНУТРИ поля "message".

Формат JSON:
{{"action": "имя_команды", "parameters": {{"ключ": "значение"}}, "message": "Что ответить пользователю"}}

Доступные действия (Возвращать JSON ТОЛЬКО если есть все параметры):
1. Изменить никнейм/имя (команда: change_username). Параметры: {{"new_username": "новый ник"}}
2. Изменить почту/логин (команда: change_email). Параметры: {{"new_email": "новый email"}}
3. Отменить премиум (команда: cancel_premium). Параметры: {{}}
4. Сбросить пароль (команда: reset_password). Параметры: {{}} 
*(Примечание: для сброса пароля новые данные не нужны, сразу возвращай команду)*

Пример: 
Пользователь: "Смени мне почту на mail@test.com"
Твой ответ: {{"action": "change_email", "parameters": {{"new_email": "mail@test.com"}}, "message": "Вы запросили изменение основного Email."}}

Если это обычный вопрос, отвечай просто текстом.

БАЗА ЗНАНИЙ:
{knowledge_context_str}
"""

    try:
        logger.info("Sending request to GigaChat LLM.")
        
        messages_payload = [{"role": "system", "content": system_instruction.strip()}]
        
        for msg in request.history[-6:]:  
            role = "user" if msg.get("sender") == "user" else "assistant"
            # Очищаем историю от старых кнопок-ссылок, чтобы не путать ИИ
            clean_content = re.sub(r'\s*\[.*?\]\(.*?\)', '', msg.get("message", ""))
            messages_payload.append({"role": role, "content": clean_content})
            
        messages_payload.append({"role": "user", "content": request.message})

        response = giga_client.chat({
            "messages": messages_payload
        })
        
        reply_text = response.choices[0].message.content
        
        # Check if response is JSON (transactional mode)
        action_data = extract_json_from_llm(reply_text)
        
        if action_data:
            action_name = action_data.action
            params = action_data.parameters
            bot_message = action_data.message

            # ЗАЩИТА 1: Если LLM вызвала смену почты, но забыла спросить саму почту
            if action_name == "change_email" and not params.get("new_email"):
                return ChatResponse(
                    reply="Какую новую почту вы хотите привязать к аккаунту? Напишите её в чат.",
                    status="success",
                    source="backend_correction"
                )
                
            # ЗАЩИТА 2: Если LLM вызвала смену ника, но забыла спросить сам ник
            if action_name == "change_username" and not params.get("new_username"):
                return ChatResponse(
                    reply="Какой новый никнейм вы хотите установить? Напишите его в чат.",
                    status="success",
                    source="backend_correction"
                )

            # ЗАЩИТА 3: Если LLM всё сделала правильно, но забыла написать пояснительный текст
            if not bot_message:
                bot_message = "Пожалуйста, подтвердите действие:"

            logger.info(f"Transactional action detected and validated: {action_name}")
            return ChatResponse(
                reply=bot_message,
                status="success",
                source="llm_action",
                action=action_name,
                parameters=params
            )
            
        # Если JSON не найден, отдаем как обычный текст
        return ChatResponse(
            reply=reply_text,
            status="success",
            source="llm"
        )

    except Exception as e:
        logger.error(f"LLM Generation Error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error during response generation.")



@app.get("/health")
async def health_check():
    status = "healthy" if ml_models.get("giga_client") else "degraded"
    return {"status": status, "service": "SiteOfSites AI Support"}

@app.get("/")
async def root():
    return {"message": "AI Support Service is running. Use /api/ai/chat endpoint."}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("AI_SERVICE_PORT", 8001))
    uvicorn.run(app, host="0.0.0.0", port=port)
