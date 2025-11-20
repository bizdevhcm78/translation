import torch
from langdetect import detect, LangDetectException
from datetime import datetime, timezone
from typing import List
import re
from models.translation import TranslationResult
from ml.model_manager_meta import model_manager_meta
from core.logging import logger


class TranslationServiceMeta:
    """Translation business logic"""

    @staticmethod
    def translate_batch(
        texts: List[str], target_lang: str, is_1_3B=False
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

        for idx, text in enumerate(cleaned_texts):
            try:
                if is_1_3B is False:
                    result = TranslationServiceMeta._translate_single(
                        text, target_lang, idx
                    )
                else:
                    result = TranslationServiceMeta._translate_single_1_3B(
                        text, target_lang, idx
                    )
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
    def translate_batch_mit_sua(
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

        for idx, text in enumerate(cleaned_texts):
            try:
                result = TranslationServiceMeta._translate_single_mit_sua(
                    text, target_lang, idx
                )

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
            # Detect source language (returns ISO code like "en", "ja", etc.)
            detected_lang_iso = detect(text)

            # Convert to NLLB format
            detected_lang_nllb = get_source_lang_meta(detected_lang_iso)

            start = datetime.now(timezone.utc)

            # Check if translation pair is supported
            model, tokenizer = model_manager_meta.get_model()

            if not model or not tokenizer:
                error_msg = f"Translation from '{detected_lang_iso}' to '{target_lang}' not supported"
                logger.warning(f"Text {idx + 1}: {error_msg}")
                return TranslationResult(
                    detected_source_language=detected_lang_iso, text="", error=error_msg
                )

            # Preprocess text
            text = preprocess_text(text)

            # Get target language code in NLLB format
            tgt_lang = get_source_lang_meta(target_lang)

            if not detected_lang_nllb or not tgt_lang:
                error_msg = (
                    f"Unsupported language pair: {detected_lang_iso} → {target_lang}"
                )
                logger.warning(f"Text {idx + 1}: {error_msg}")
                return TranslationResult(
                    detected_source_language=detected_lang_iso, text="", error=error_msg
                )

            logger.info(f"Translating: {detected_lang_nllb} → {tgt_lang}")

            # Set source language for tokenizer
            tokenizer.src_lang = detected_lang_nllb

            # Tokenize input
            inputs = tokenizer(
                text, return_tensors="pt", padding=True, truncation=True, max_length=512
            )
            inputs = {k: v.to(model_manager_meta.device) for k, v in inputs.items()}

            # CRITICAL: Get forced_bos_token_id for target language
            # For NllbTokenizerFast, use convert_tokens_to_ids
            try:
                forced_bos_token_id = tokenizer.convert_tokens_to_ids(tgt_lang)
            except:
                # Fallback: manually get the token ID
                forced_bos_token_id = (
                    tokenizer.lang_code_to_id.get(
                        tgt_lang, tokenizer.convert_tokens_to_ids(tgt_lang)
                    )
                    if hasattr(tokenizer, "lang_code_to_id")
                    else tokenizer.convert_tokens_to_ids(tgt_lang)
                )

            logger.info(
                f"Using forced_bos_token_id: {forced_bos_token_id} for {tgt_lang}"
            )

            # Perform translation with proper parameters
            with torch.no_grad():
                translated_tokens = model.generate(
                    **inputs,
                    forced_bos_token_id=forced_bos_token_id,
                    max_length=512,
                    num_beams=5,
                    length_penalty=1.0,
                    no_repeat_ngram_size=3,
                    early_stopping=True,
                )

            translated_text = tokenizer.decode(
                translated_tokens[0], skip_special_tokens=True
            )

            # Post-process for specific languages
            if target_lang == "ja":
                translated_text = postprocess_japanese(translated_text)

            end = datetime.now(timezone.utc)
            duration = (end - start).total_seconds()

            logger.info(
                f"Text {idx + 1} translated in {duration:.3f}s ({detected_lang_iso} → {target_lang}): {translated_text[:50]}..."
            )

            return TranslationResult(
                detected_source_language=detected_lang_iso, text=translated_text
            )

        except LangDetectException:
            logger.error(f"Text {idx + 1} language detection failed")
            return TranslationResult(
                detected_source_language="unknown",
                text="",
                error="Language detection failed",
            )
        except Exception as e:
            logger.error(f"Text {idx + 1} translation error: {str(e)}", exc_info=True)
            return TranslationResult(
                detected_source_language="unknown", text="", error=str(e)
            )

    @staticmethod
    def _translate_single_1_3B(
        text: str, target_lang: str, idx: int
    ) -> TranslationResult:
        """Translate a single text"""
        try:
            # Detect source language (returns ISO code like "en", "ja", etc.)
            detected_lang_iso = detect(text)

            # Convert to NLLB format
            detected_lang_nllb = get_source_lang_meta(detected_lang_iso)

            start = datetime.now(timezone.utc)

            # Check if translation pair is supported
            model, tokenizer = model_manager_meta.get_model_1_3()

            if not model or not tokenizer:
                error_msg = f"Translation from '{detected_lang_iso}' to '{target_lang}' not supported"
                logger.warning(f"Text {idx + 1}: {error_msg}")
                return TranslationResult(
                    detected_source_language=detected_lang_iso, text="", error=error_msg
                )

            # Preprocess text
            text = preprocess_text(text)

            # Get target language code in NLLB format
            tgt_lang = get_source_lang_meta(target_lang)

            if not detected_lang_nllb or not tgt_lang:
                error_msg = (
                    f"Unsupported language pair: {detected_lang_iso} → {target_lang}"
                )
                logger.warning(f"Text {idx + 1}: {error_msg}")
                return TranslationResult(
                    detected_source_language=detected_lang_iso, text="", error=error_msg
                )

            logger.info(f"Translating: {detected_lang_nllb} → {tgt_lang}")

            # Set source language for tokenizer
            tokenizer.src_lang = detected_lang_nllb

            # Tokenize input
            inputs = tokenizer(
                text, return_tensors="pt", padding=True, truncation=True, max_length=512
            )
            inputs = {k: v.to(model_manager_meta.device) for k, v in inputs.items()}

            # CRITICAL: Get forced_bos_token_id for target language
            # For NllbTokenizerFast, use convert_tokens_to_ids
            try:
                forced_bos_token_id = tokenizer.convert_tokens_to_ids(tgt_lang)
            except:
                # Fallback: manually get the token ID
                forced_bos_token_id = (
                    tokenizer.lang_code_to_id.get(
                        tgt_lang, tokenizer.convert_tokens_to_ids(tgt_lang)
                    )
                    if hasattr(tokenizer, "lang_code_to_id")
                    else tokenizer.convert_tokens_to_ids(tgt_lang)
                )

            logger.info(
                f"Using forced_bos_token_id: {forced_bos_token_id} for {tgt_lang}"
            )

            # Perform translation with proper parameters
            with torch.no_grad():
                translated_tokens = model.generate(
                    **inputs,
                    forced_bos_token_id=forced_bos_token_id,
                    max_length=512,
                    num_beams=5,
                    length_penalty=1.0,
                    no_repeat_ngram_size=3,
                    early_stopping=True,
                )

            translated_text = tokenizer.decode(
                translated_tokens[0], skip_special_tokens=True
            )

            # Post-process for specific languages
            if target_lang == "ja":
                translated_text = postprocess_japanese(translated_text)

            end = datetime.now(timezone.utc)
            duration = (end - start).total_seconds()

            logger.info(
                f"Text {idx + 1} translated in {duration:.3f}s ({detected_lang_iso} → {target_lang}): {translated_text[:50]}..."
            )

            return TranslationResult(
                detected_source_language=detected_lang_iso, text=translated_text
            )

        except LangDetectException:
            logger.error(f"Text {idx + 1} language detection failed")
            return TranslationResult(
                detected_source_language="unknown",
                text="",
                error="Language detection failed",
            )
        except Exception as e:
            logger.error(f"Text {idx + 1} translation error: {str(e)}", exc_info=True)
            return TranslationResult(
                detected_source_language="unknown", text="", error=str(e)
            )

    @staticmethod
    def _translate_single_mit_sua(
        text: str, target_lang: str, idx: int
    ) -> TranslationResult:
        """Translate a single text"""
        try:
            # Detect source language (returns ISO code like "en", "ja", etc.)
            detected_lang_iso = detect(text)

            start = datetime.now(timezone.utc)

            # Check if translation pair is supported
            model, tokenizer = model_manager_meta.get_model_mit_sua()

            if not model or not tokenizer:
                error_msg = f"Translation from '{detected_lang_iso}' to '{target_lang}' not supported"
                logger.warning(f"Text {idx + 1}: {error_msg}")
                return TranslationResult(
                    detected_source_language=detected_lang_iso, text="", error=error_msg
                )

            # Preprocess text
            text = preprocess_text(text)

            # Tokenize input
            inputs = tokenizer(
                text, return_tensors="pt", padding=True, truncation=True, max_length=512
            )
            inputs = {k: v.to(model_manager_meta.device) for k, v in inputs.items()}

            # Perform translation with proper parameters
            with torch.no_grad():
                translated_tokens = model.generate(
                    **inputs,
                    max_length=512,
                    num_beams=5,
                    length_penalty=1.0,
                    no_repeat_ngram_size=3,
                    early_stopping=True,
                )

            translated_text = tokenizer.decode(
                translated_tokens[0], skip_special_tokens=True
            )

            # Post-process for specific languages
            if target_lang == "ja":
                translated_text = postprocess_japanese(translated_text)

            end = datetime.now(timezone.utc)
            duration = (end - start).total_seconds()

            logger.info(
                f"Text {idx + 1} translated in {duration:.3f}s ({detected_lang_iso} → {target_lang}): {translated_text[:50]}..."
            )

            return TranslationResult(
                detected_source_language=detected_lang_iso, text=translated_text
            )

        except LangDetectException:
            logger.error(f"Text {idx + 1} language detection failed")
            return TranslationResult(
                detected_source_language="unknown",
                text="",
                error="Language detection failed",
            )
        except Exception as e:
            logger.error(f"Text {idx + 1} translation error: {str(e)}", exc_info=True)
            return TranslationResult(
                detected_source_language="unknown", text="", error=str(e)
            )


def preprocess_text(text: str) -> str:
    """Clean text before translation"""
    # Remove extra whitespace
    text = " ".join(text.split())
    # Ensure proper punctuation spacing
    text = re.sub(r"\s+([.,!?])", r"\1", text)
    text = text.strip()
    return text


def postprocess_japanese(text: str) -> str:
    """Clean Japanese output"""
    # Remove spaces between Japanese characters
    text = re.sub(
        r"(?<=[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF])\s+(?=[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF])",
        "",
        text,
    )
    return text.strip()


def get_source_lang_meta(lang_code: str) -> str:
    """Convert ISO 639-1 codes to NLLB language codes"""
    lang_mapping = {
        "en": "eng_Latn",
        "ja": "jpn_Jpan",
        "vi": "vie_Latn",
        "zh": "zho_Hans",  # Simplified Chinese
        "ko": "kor_Hang",  # Korean
        "th": "tha_Thai",  # Thai
        "fr": "fra_Latn",  # French
        "de": "deu_Latn",  # German
        "es": "spa_Latn",  # Spanish
    }
    return lang_mapping.get(lang_code, "")
