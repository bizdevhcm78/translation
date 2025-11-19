import torch
from transformers import MarianMTModel, MarianTokenizer
from typing import Dict
from core.config import settings
from core.logging import logger

class ModelManager:
    """Manages ML model lifecycle"""
    
    def __init__(self):
        self.models: Dict[str, MarianMTModel] = {}
        self.tokenizers: Dict[str, MarianTokenizer] = {}
        self.device = None
    
    def load_models(self):
        """Load all translation models"""
        try:
            logger.info("Starting model loading process...")
            
            # Determine device
            if settings.DEVICE == "auto":
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                self.device = settings.DEVICE
            
            logger.info(f"Using device: {self.device}")
            
            # Load each model
            for key, model_name in settings.SUPPORTED_TRANSLATIONS.items():
                logger.info(f"Loading {key} model from {model_name}...")
                
                self.tokenizers[key] = MarianTokenizer.from_pretrained(model_name)
                self.models[key] = MarianMTModel.from_pretrained(model_name)
                self.models[key].to(self.device)
                
                logger.info(f"✓ {key} model loaded successfully")
            
            logger.info(f"All {len(self.models)} models loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load models: {str(e)}")
            raise
    
    def cleanup(self):
        """Cleanup models and free memory"""
        logger.info("Cleaning up models...")
        self.models.clear()
        self.tokenizers.clear()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def get_model(self, model_key: str):
        """Get model and tokenizer by key"""
        return self.models.get(model_key), self.tokenizers.get(model_key)
    
    def is_loaded(self) -> bool:
        """Check if models are loaded"""
        return bool(self.models and self.tokenizers)

# Global model manager instance
model_manager = ModelManager()
