import logging
from typing import Dict, Any, Tuple, List
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
import json
import os

from app.ports.knowledge_base import KnowledgeBasePort
from app.infrastructure.db_models import Base, KnowledgeChunk
from sentence_transformers import SentenceTransformer

logger = logging.getLogger("SiteOfSites_AI")

class PgVectorKnowledgeBaseAdapter(KnowledgeBasePort):
    def __init__(self, db_url: str, fallback_filepath: str = "knowledge_base.json"):
        self.db_url = db_url
        self.fallback_filepath = fallback_filepath
        try:
            self.engine = create_engine(self.db_url)
            self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
            
            # Включаем расширение pgvector
            self._ensure_pgvector_extension()
            
            # Создаем таблицы (если их нет)
            Base.metadata.create_all(bind=self.engine)
            
            logger.info("PgVector connection initialized successfully.")
            
            # Инициализация модели эмбеддингов
            logger.info("Loading sentence-transformer model for embeddings...")
            self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("Embedding model loaded.")
            
            self._is_db_ready = True
            
        except Exception as e:
            logger.error(f"Failed to connect to database for pgvector: {e}")
            self.engine = None
            self.SessionLocal = None
            self.embedder = None
            self._is_db_ready = False

    def _ensure_pgvector_extension(self):
        try:
            with self.engine.begin() as conn:
                conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        except Exception as e:
            logger.warning(f"Could not create vector extension: {e}. Check database permissions.")

    def search_relevant_chunks(self, query: str, top_k: int = 4) -> List[str]:
        """
        Ищет релевантные куски текста в векторной БД с использованием косинусного расстояния.
        """
        if not self._is_db_ready or not self.embedder:
            logger.warning("DB or Embedder not ready, returning empty chunks.")
            return []

        try:
            # 1. Векторизуем запрос
            query_embedding = self.embedder.encode(query).tolist()
            
            # 2. Ищем в базе
            with self.SessionLocal() as session:
                # Используем cosine_distance (<=>)
                results = session.query(KnowledgeChunk).order_by(
                    KnowledgeChunk.embedding.cosine_distance(query_embedding)
                ).limit(top_k).all()
                
                chunks = []
                for row in results:
                    if row.question:
                        chunks.append(f"В: {row.question}\nО: {row.content}")
                    else:
                        chunks.append(row.content)
                        
                return chunks
        except Exception as e:
            logger.error(f"Error during vector search: {e}")
            return []

    def get_knowledge_base(self) -> Tuple[str, Dict[str, Any]]:
        """
        Возвращает словарь (FAQ для быстрого ответа) и строку (резервный вариант, если БД пуста).
        В идеале строка контекста больше не нужна, так как работает search_relevant_chunks.
        """
        try:
            with open(self.fallback_filepath, "r", encoding="utf-8") as f:
                content = f.read()
                data = json.loads(content)
                return content, data
        except Exception as e:
            logger.error(f"Fallback KB reading failed: {e}")
            return "{}", {}
