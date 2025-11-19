from fastapi import HTTPException, status

class TranslationException(HTTPException):
    """Base translation exception"""
    def __init__(self, detail: str):
        super().__init__(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=detail
        )

class ModelNotLoadedException(HTTPException):
    """Model not loaded exception"""
    def __init__(self):
        super().__init__(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Translation models not loaded yet"
        )

class UnsupportedLanguagePairException(HTTPException):
    """Unsupported language pair exception"""
    def __init__(self, source: str, target: str):
        super().__init__(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Translation from '{source}' to '{target}' not supported"
        )