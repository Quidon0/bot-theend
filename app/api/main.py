import logging
import uuid
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware

from app.config.settings import settings
from app.api.routers import chat
from app.infrastructure.llm_adapters import GigaChatAdapter
from app.infrastructure.pgvector_kb_adapter import PgVectorKnowledgeBaseAdapter
from app.application.chat_service import ChatService

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("SiteOfSites_AI")

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Initializing adapters...")
    try:
        llm_adapter = GigaChatAdapter(
            credentials=settings.GIGACHAT_CREDENTIALS,
            verify_ssl=settings.GIGACHAT_VERIFY_SSL
        )
        kb_adapter = PgVectorKnowledgeBaseAdapter(
            db_url=settings.database_url,
            fallback_filepath=settings.KB_FILE_PATH
        )
        
        app.state.llm_adapter = llm_adapter
        app.state.kb_adapter = kb_adapter
        app.state.chat_service = ChatService(llm_adapter=llm_adapter, kb_adapter=kb_adapter)
        
        logger.info("Adapters initialized successfully.")
    except Exception as e:
        logger.critical(f"Critical failure during startup: {e}")
        # В продакшене лучше упасть, чем работать некорректно
        raise
    
    # Preload KB
    kb_adapter.get_knowledge_base()
    yield

    if hasattr(app.state, "llm_adapter") and app.state.llm_adapter:
        await app.state.llm_adapter.close()

app = FastAPI(
    title="SiteOfSites AI Support",
    lifespan=lifespan,
    docs_url=None,
    redoc_url=None,
    openapi_url=None
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_methods=["POST"],
    allow_headers=["Authorization", "Content-Type", "X-Internal-Key"],
)

@app.middleware("http")
async def add_request_id(request: Request, call_next):
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response

app.include_router(chat.router, prefix="/api/ai")

@app.get("/health/live")
async def liveness_check():
    return {"status": "live"}

@app.get("/health/ready")
async def readiness_check(request: Request):
    llm_adapter = getattr(request.app.state, "llm_adapter", None)
    kb_adapter = getattr(request.app.state, "kb_adapter", None)
    
    if not llm_adapter:
        raise HTTPException(status_code=503, detail="LLM not initialized")
    
    # Проверяем базу знаний
    if kb_adapter:
        _, kb_dict = kb_adapter.get_knowledge_base()
        if not kb_dict:
            raise HTTPException(status_code=503, detail="KB not loaded")
    else:
        raise HTTPException(status_code=503, detail="KB adapter not initialized")
        
    if not await llm_adapter.is_healthy():
         return {"status": "degraded", "details": "LLM provider unreachable"}

    return {"status": "ready"}

@app.get("/")
async def root():
    return {"message": "AI Support Service is running. Use /api/ai/chat endpoint."}
