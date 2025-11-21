import torch
from transformers import (
    MarianMTModel,
    MarianTokenizer,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    M2M100Tokenizer,
)
from typing import Dict
from core.config import settings
from core.logging import logger


class ModelManager:
    """Manages ML model lifecycle"""

    def __init__(self):
        self.models: Dict[str, MarianMTModel] = {}
        self.tokenizers: Dict[str, MarianTokenizer] = {}
        self.device = None

        # Models
        self.model_meta = None
        self.tokenizer_meta = None

        self.model_meta_1_3B = None
        self.tokenizer_meta_1_3B = None

        self.model_mit_sua = None
        self.tokenizer_mit_sua = None

        self.model_m2m100_1_2B = None
        self.tokenizer_m2m100_1_2B: M2M100Tokenizer = None

    # ---------------------------
    # Helper functions
    # ---------------------------
    def _load_single_model(self, model_path):
        """Load a single model + tokenizer safely"""
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        model.to(self.device)
        return model, tokenizer

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

            # ---- Load MIT SUA ----
            if settings.SUPPORTED_MIT_SUA:
                self.model_mit_sua, self.tokenizer_mit_sua = self._load_single_model(
                    settings.SUPPORTED_TRANSLATIONS_MIT_SUA
                )
                logger.info("Loaded MIT-Sua model successfully")

            # ---- Load M2M100 1.2B ----
            if settings.SUPPORTED_M2M100_1_2B:
                self.model_m2m100_1_2B, self.tokenizer_m2m100_1_2B = (
                    self._load_single_model(settings.SUPPORTED_TRANSLATIONS_M2M100_1_2B)
                )
                logger.info("Loaded M2M100-1.2B successfully")

            # ---- Load META models ----
            if settings.SUPPORTED_META:
                self.model_meta, self.tokenizer_meta = self._load_single_model(
                    settings.SUPPORTED_TRANSLATIONS_META
                )
                logger.info("Loaded Meta model successfully")

            if settings.SUPPORTED_META_1_3B:
                self.model_meta_1_3B, self.tokenizer_meta_1_3B = (
                    self._load_single_model(settings.SUPPORTED_TRANSLATIONS_META_1_3B)
                )
                logger.info("Loaded Meta 1.3B successfully")
            logger.info(f"Models loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load models: {str(e)}")
            raise

    def cleanup(self):
        """Cleanup models and free memory"""
        logger.info("Cleaning up models...")
        self.models.clear()
        self.tokenizers.clear()
        self.model_meta = None
        self.tokenizer_meta = None
        self.model_meta_1_3B = None
        self.tokenizer_meta_1_3B = None
        self.model_mit_sua = None
        self.tokenizer_mit_sua = None
        self.model_m2m100_1_2B = None
        self.tokenizer_m2m100_1_2B = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ---------------------------
    # Getters
    # ---------------------------
    def get_model(self, model_key: str):
        """Get model and tokenizer by key"""
        return self.models.get(model_key), self.tokenizers.get(model_key)

    def get_model_meta(self):
        return self.model_meta, self.tokenizer_meta

    def get_model_meta_1_3B(self):
        return self.model_meta_1_3B, self.tokenizer_meta_1_3B

    def get_model_mit_sua(self):
        return self.model_mit_sua, self.tokenizer_mit_sua

    def get_model_m2m100_1_2B(self):
        return self.model_m2m100_1_2B, self.tokenizer_m2m100_1_2B

    # ---------------------------
    # is_loaded
    # ---------------------------
    def is_loaded(self) -> bool:
        """Check if at least one model is loaded"""
        return any(
            [
                self.models,
                self.model_meta,
                self.model_meta_1_3B,
                self.model_mit_sua,
                self.model_m2m100_1_2B,
            ]
        )


# Global model manager instance
model_manager = ModelManager()
