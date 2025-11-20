import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from core.config import settings
from core.logging import logger

class ModelManagerMeta:
    """Manages ML model lifecycle"""
    
    def __init__(self):
        self.models: AutoModelForSeq2SeqLM = None
        self.tokenizers: AutoTokenizer = None  # Changed from NllbTokenizer
        self.models_1_3: AutoModelForSeq2SeqLM = None
        self.tokenizers_1_3: AutoTokenizer = None  # Changed from NllbTokenizer
        self.models_mit_sua: AutoModelForSeq2SeqLM = None
        self.tokenizers_mit_sua: AutoTokenizer = None  # Changed from NllbTokenizer
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
            
            # Use AutoTokenizer instead of NllbTokenizer
            self.tokenizers = AutoTokenizer.from_pretrained(
                settings.SUPPORTED_TRANSLATIONS_META,
                use_fast=True
            )
            self.models = AutoModelForSeq2SeqLM.from_pretrained(
                settings.SUPPORTED_TRANSLATIONS_META
            )
            self.models.to(self.device)
            # Use AutoTokenizer instead of NllbTokenizer
            self.tokenizers_1_3 = AutoTokenizer.from_pretrained(
                settings.SUPPORTED_TRANSLATIONS_META_1_3B,
                use_fast=True
            )
            self.models_1_3 = AutoModelForSeq2SeqLM.from_pretrained(
                settings.SUPPORTED_TRANSLATIONS_META_1_3B
            )
            self.models_1_3.to(self.device)

            # Use AutoTokenizer instead of NllbTokenizer model mit_sua
            self.tokenizers_mit_sua = AutoTokenizer.from_pretrained(
                settings.SUPPORTED_TRANSLATIONS_MIT_SUA,
                use_fast=True
            )
            self.models_mit_sua = AutoModelForSeq2SeqLM.from_pretrained(
                settings.SUPPORTED_TRANSLATIONS_MIT_SUA
            )
            self.models_mit_sua.to(self.device)
            
            logger.info(f"Model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load models: {str(e)}")
            raise
    
    def cleanup(self):
        """Cleanup models and free memory"""
        logger.info("Cleaning up models...")
        self.models = None
        self.tokenizers = None
        self.models_1_3 = None
        self.tokenizers_1_3 = None
        self.models_mit_sua = None
        self.tokenizers_mit_sua = None
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def get_model(self):
        """Get model and tokenizer"""
        return self.models, self.tokenizers

    def get_model_1_3(self):
        """Get model and tokenizer"""
        return self.models_1_3, self.tokenizers_1_3

    def get_model_mit_sua(self):
        """Get model and tokenizer"""
        return self.models_mit_sua, self.tokenizers_mit_sua

    def is_loaded(self) -> bool:
        """Check if models are loaded"""
        return bool(self.models and self.tokenizers and self.models_1_3 and self.tokenizers_1_3 and self.models_mit_sua and self.tokenizers_mit_sua)

# Global model manager instance
model_manager_meta = ModelManagerMeta()