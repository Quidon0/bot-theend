import os
import json
import logging
import sys
from urllib.parse import quote_plus

# Добавляем родительскую директорию в PYTHONPATH для импортов
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sentence_transformers import SentenceTransformer
from app.infrastructure.db_models import Base, KnowledgeChunk

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s]: %(message)s")
logger = logging.getLogger("Migrate_KB")

def get_db_url():
    db_host = os.environ.get("DB_HOST", "localhost")
    db_port = os.environ.get("DB_PORT", "5432")
    db_user = os.environ.get("DB_USER", "postgres")
    db_password = os.environ.get("DB_PASSWORD", "Sctorlorn25565")
    db_name = os.environ.get("DB_NAME", "siteofsites")
    return f"postgresql://{db_user}:{quote_plus(db_password)}@{db_host}:{db_port}/{db_name}"

def main():
    kb_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "knowledge_base.json")
    if not os.path.exists(kb_path):
        logger.error(f"File not found: {kb_path}")
        return

    logger.info("Loading knowledge base from JSON...")
    with open(kb_path, "r", encoding="utf-8") as f:
        kb_data = json.loads(f.read())

    db_url = get_db_url()
    logger.info(f"Connecting to database: {db_url}")
    
    try:
        engine = create_engine(db_url)
        SessionLocal = sessionmaker(bind=engine)
        Base.metadata.create_all(bind=engine)
    except Exception as e:
        logger.error(f"Database connection failed: {e}")
        return

    logger.info("Loading sentence-transformers model...")
    embedder = SentenceTransformer('all-MiniLM-L6-v2')

    chunks_to_insert = []

    # Обработка FAQ
    if "faq" in kb_data:
        logger.info("Processing FAQ section...")
        for question, answer in kb_data["faq"].items():
            content_text = f"В: {question}\nО: {answer}"
            embedding = embedder.encode(content_text).tolist()
            chunks_to_insert.append(KnowledgeChunk(
                category="faq",
                question=question,
                content=answer,
                embedding=embedding
            ))

    # Обработка других секций (пример для general_info, если она есть)
    for section_name, section_content in kb_data.items():
        if section_name == "faq":
            continue
            
        logger.info(f"Processing section: {section_name}...")
        
        # Если секция - это список строк
        if isinstance(section_content, list):
            for item in section_content:
                if isinstance(item, str):
                    embedding = embedder.encode(item).tolist()
                    chunks_to_insert.append(KnowledgeChunk(
                        category=section_name,
                        content=item,
                        embedding=embedding
                    ))
        # Если секция - это строка
        elif isinstance(section_content, str):
            # Простой чанкинг по абзацам
            paragraphs = [p.strip() for p in section_content.split('\n\n') if p.strip()]
            for p in paragraphs:
                embedding = embedder.encode(p).tolist()
                chunks_to_insert.append(KnowledgeChunk(
                    category=section_name,
                    content=p,
                    embedding=embedding
                ))
        # Если словарь
        elif isinstance(section_content, dict):
            for key, val in section_content.items():
                content_text = f"{key}: {val}"
                embedding = embedder.encode(content_text).tolist()
                chunks_to_insert.append(KnowledgeChunk(
                    category=section_name,
                    question=key,
                    content=str(val),
                    embedding=embedding
                ))

    if chunks_to_insert:
        logger.info(f"Inserting {len(chunks_to_insert)} chunks into the database...")
        with SessionLocal() as session:
            # Опционально: очистить старые данные
            session.query(KnowledgeChunk).delete()
            session.add_all(chunks_to_insert)
            session.commit()
        logger.info("Migration completed successfully.")
    else:
        logger.warning("No chunks found to insert.")

if __name__ == "__main__":
    main()
