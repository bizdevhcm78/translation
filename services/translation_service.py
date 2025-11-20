import torch
from langdetect import detect, LangDetectException
from datetime import datetime, timezone
from typing import List
from models.translation import TranslationResult
from ml.model_manager import model_manager
from core.logging import logger


class TranslationService:
    """Translation business logic"""

    @staticmethod
    def translate_batch(
        texts: List[str], target_lang: str
    ) -> tuple[List[TranslationResult], int, int]:
        """
        Translate a batch of texts

        Returns: (results, success_count, error_count)
        """
        # Filter out empty/whitespace-only texts
        cleaned_texts = [text.strip() for text in texts if text and text.strip()]

        # Log for debugging
        logger.info(
            f"Received {len(texts)} texts, processing {len(cleaned_texts)} non-empty texts"
        )

        results = []
        success_count = 0
        error_count = 0

        for idx, text in enumerate(cleaned_texts):  # Use cleaned_texts
            try:
                result = TranslationService._translate_single(text, target_lang, idx)
                results.append(result)

                if result.error:
                    error_count += 1
                else:
                    success_count += 1

            except Exception as e:
                logger.error(f"Text {idx + 1} unexpected error: {str(e)}")
                results.append(
                    TranslationResult(detected_source_language="unknown", text="")
                )
                error_count += 1

        return results, success_count, error_count

    @staticmethod
    def _translate_single(text: str, target_lang: str, idx: int) -> TranslationResult:
        """Translate a single text"""
        try:
            # Detect source language
            detected_lang = detect(text)
            start = datetime.now(timezone.utc)

            # Get model key
            model_key = f"{detected_lang}-{target_lang}"

            # Check if translation pair is supported
            model, tokenizer = model_manager.get_model(model_key)

            if not model or not tokenizer:
                error_msg = f"Translation from '{detected_lang}' to '{target_lang}' not supported"
                logger.warning(f"Text {idx + 1}: {error_msg}")
                return TranslationResult(
                    detected_source_language=detected_lang, text="", error=error_msg
                )

            # Perform translation
            inputs = tokenizer([text], return_tensors="pt", padding=True)
            inputs = {k: v.to(model_manager.device) for k, v in inputs.items()}

            with torch.no_grad():
                translated_tokens = model.generate(**inputs)

            translated_text = tokenizer.decode(
                translated_tokens[0], skip_special_tokens=True
            )
            if target_lang == "ja":
                translated_text = postprocess_japanese(translated_text)
            end = datetime.now(timezone.utc)
            duration = (end - start).total_seconds()

            logger.info(
                f"Text {idx + 1} translated in {duration:.3f}s ({detected_lang} → {target_lang})"
            )

            return TranslationResult(
                detected_source_language=detected_lang, text=translated_text
            )

        except LangDetectException:
            logger.error(f"Text {idx + 1} language detection failed")
            return TranslationResult(
                detected_source_language="unknown", text=""
            )


def preprocess_text(text: str) -> str:
    """Clean text before translation"""
    # Remove extra whitespace
    text = " ".join(text.split())
    # Ensure proper punctuation
    text = text.strip()
    return text

def postprocess_japanese(text: str) -> str:
    """Clean Japanese output"""
    # Remove spaces between Japanese characters
    import re
    # Remove spaces between non-ASCII characters (Japanese)
    text = re.sub(r'(?<=[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF])\s+(?=[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF])', '', text)
    return text.strip()