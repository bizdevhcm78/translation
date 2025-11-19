from fastapi import APIRouter
from datetime import datetime, timezone
from ml.model_manager import model_manager
from core.config import settings

router = APIRouter(tags=["Health"])

@router.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "status": "running",
        "models_loaded": len(model_manager.models),
        "supported_translations": list(settings.SUPPORTED_TRANSLATIONS.keys())
    }

@router.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy" if model_manager.is_loaded() else "unhealthy",
        "models_loaded": len(model_manager.models),
        "device": model_manager.device,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }