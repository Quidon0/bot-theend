from abc import ABC, abstractmethod
from typing import List
from gigachat.models import Messages

class LLMPort(ABC):
    @abstractmethod
    async def complete(self, messages: List[Messages], temperature: float, max_tokens: int) -> str:
        pass

    @abstractmethod
    async def is_healthy(self) -> bool:
        pass

    @abstractmethod
    async def close(self):
        pass