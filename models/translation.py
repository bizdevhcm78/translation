from pydantic import BaseModel, Field, field_validator
from typing import List, Optional

class TranslationResult(BaseModel):
    """Individual translation result"""
    detected_source_language: str
    text: str
    error: Optional[str] = None

class TranslateRequest(BaseModel):
    """Translation request model"""
    texts: List[str] = Field(
        ..., 
        min_items=1, 
        max_items=100, 
        description="List of texts to translate"
    )
    target_lang: str = Field(
        ..., 
        min_length=2, 
        max_length=2, 
        description="Target language code"
    )
    
    @field_validator('texts')
    def validate_texts(cls, v):
        for text in v:
            if not text or not text.strip():
                raise ValueError("Empty texts are not allowed")
            if len(text) > 5000:
                raise ValueError("Text length cannot exceed 5000 characters")
        return v
    
    @field_validator('target_lang')
    def validate_target_lang(cls, v):
        supported_langs = {'en', 'vi', 'ja'}
        if v.lower() not in supported_langs:
            raise ValueError(f"Target language must be one of: {supported_langs}")
        return v.lower()

class TranslateResponse(BaseModel):
    """Translation response model"""
    translations: List[TranslationResult]