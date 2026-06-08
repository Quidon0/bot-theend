import asyncio
import logging
from typing import List

from gigachat import GigaChat
from gigachat.models import Chat, Messages, MessagesRole # <--- Добавили импорт Chat
import pybreaker
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type

from app.ports.llm import LLMPort
from app.config.settings import settings

logger = logging.getLogger("SiteOfSites_AI")

# Circuit Breaker configuration
llm_breaker = pybreaker.CircuitBreaker(
    fail_max=5, 
    reset_timeout=60,
    exclude=[asyncio.TimeoutError] # TimeoutError should not open the circuit breaker immediately, but contribute to failures
)

class GigaChatAdapter(LLMPort):
    def __init__(self, credentials, verify_ssl):
        self.client = GigaChat(credentials=credentials, verify_ssl_certs=verify_ssl)

    @llm_breaker
    @retry(
        wait=wait_exponential(multiplier=1, min=1, max=10),
        stop=stop_after_attempt(3),
        retry=retry_if_exception_type((asyncio.TimeoutError, Exception)),
    )
    async def complete(self, messages: List[Messages], temperature: float, max_tokens: int) -> str:
        async with asyncio.timeout(settings.GIGACHAT_TIMEOUT):
            # Упаковываем параметры в объект Chat перед отправкой
            payload = Chat(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            response = await self.client.achat(payload)
            return response.choices[0].message.content

    async def is_healthy(self) -> bool:
        try:
            # Здесь тоже исправляем на объект Chat
            payload = Chat(
                messages=[Messages(role=MessagesRole.USER, content="ping")],
                max_tokens=1
            )
            await self.client.achat(payload)
            return True
        except Exception:
            return False

    async def close(self):
        await self.client.aclose()