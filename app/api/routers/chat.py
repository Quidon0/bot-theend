import uuid
from typing import Optional
from fastapi import APIRouter, HTTPException, Depends, Header, Request

from app.domain.models import ChatRequest, ChatResponse
from app.application.chat_service import ChatService
from app.config.settings import settings

# This router expects chat_service to be injected or available globally.
# For simplicity, we can pass it as a dependency or inject it through request state.
router = APIRouter()

async def verify_internal_auth(x_internal_key: str = Header(...)):
    """Проверка, что запрос пришел от нашего Backend"""
    if x_internal_key != settings.INTERNAL_API_KEY:
        raise HTTPException(status_code=403, detail="Forbidden: Invalid Internal Key")

def get_chat_service(request: Request) -> ChatService:
    service = getattr(request.app.state, "chat_service", None)
    if not service:
        raise HTTPException(status_code=503, detail="Chat service not initialized")
    return service

@router.post("/chat", response_model=ChatResponse, dependencies=[Depends(verify_internal_auth)])
async def chat_endpoint(request: ChatRequest, service: ChatService = Depends(get_chat_service)):
    try:
        response = await service.process_chat_request(request)
        return response
    except Exception as e:
        # Check if it's already an HTTPException from timeout, etc.
        if isinstance(e, HTTPException):
            raise e
        import logging
        logger = logging.getLogger("SiteOfSites_AI")
        logger.error(f"LLM Generation Error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error during response generation.")
