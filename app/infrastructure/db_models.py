from sqlalchemy import Column, Integer, String, Text
from sqlalchemy.orm import declarative_base
from pgvector.sqlalchemy import Vector

Base = declarative_base()

class KnowledgeChunk(Base):
    __tablename__ = "knowledge_chunks"

    id = Column(Integer, primary_key=True, index=True)
    category = Column(String(100), index=True) # например: "faq", "general", "billing"
    question = Column(Text, nullable=True)     # Для FAQ - вопрос
    content = Column(Text, nullable=False)     # Сам текст (ответ или абзац статьи)
    embedding = Column(Vector(384))            # Вектор (размерность 384 для all-MiniLM-L6-v2)
