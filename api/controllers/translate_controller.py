from fastapi import APIRouter, HTTPException, status
from datetime import datetime, timezone
from models.translation import TranslateRequest, TranslateResponse
from services.translation_service import TranslationService
from ml.model_manager import model_manager
from core.exceptions import ModelNotLoadedException
from core.logging import logger

router = APIRouter(prefix="/translate", tags=["Translation"])

@router.post("", response_model=TranslateResponse)
def translate(req: TranslateRequest):
    """
    Translate texts to target language
    
    Supported translation pairs:
    - English ↔ Vietnamese
    - Japanese → English
    """
    if not model_manager.is_loaded():
        logger.error("Translation attempted before models loaded")
        raise ModelNotLoadedException()
    
    total_start = datetime.now(timezone.utc)
    
    # Translate batch
    results, success_count, error_count = TranslationService.translate_batch(
        req.texts, 
        req.target_lang
    )
    
    total_end = datetime.now(timezone.utc)
    total_duration = (total_end - total_start).total_seconds()
    
    logger.info(
        f"Batch translation completed: {success_count} success, "
        f"{error_count} errors, {total_duration:.3f}s total"
    )
    
    return TranslateResponse(
        translations=results,
    )