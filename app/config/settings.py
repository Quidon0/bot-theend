from typing import List, Union, Optional
from pydantic import Field, ConfigDict, field_validator
from pydantic_settings import BaseSettings
import json
from urllib.parse import quote_plus

class Settings(BaseSettings):
    GIGACHAT_CREDENTIALS: str = Field(..., min_length=10) # Обязательно, без дефолта
    GIGACHAT_VERIFY_SSL: bool = True # В продакшене должно быть True
    INTERNAL_API_KEY: str = Field(..., min_length=16)
    ALLOWED_ORIGINS: Union[List[str], str] = ["https://siteofsites.ru"]
    GIGACHAT_TIMEOUT: int = 25
    KB_FILE_PATH: str = "knowledge_base.json"
    
    # DB Configuration for RAG
    DB_HOST: str = "localhost"
    DB_PORT: str = "5432"
    DB_USER: str = "postgres"
    DB_PASSWORD: str = "Sctorlorn25565"
    DB_NAME: str = "siteofsites"

    @property
    def database_url(self) -> str:
        return f"postgresql://{self.DB_USER}:{quote_plus(self.DB_PASSWORD)}@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"

    @field_validator('ALLOWED_ORIGINS', mode='before')
    @classmethod
    def parse_allowed_origins(cls, v):
        if isinstance(v, str):
            try:
                # Try to parse as JSON first
                parsed = json.loads(v)
                if isinstance(parsed, list):
                    return parsed
            except json.JSONDecodeError:
                pass
            # If not JSON, assume comma-separated
            return [origin.strip() for origin in v.split(',') if origin.strip()]
        return v

    model_config = ConfigDict(env_file=".env", env_file_encoding="utf-8")

settings = Settings()