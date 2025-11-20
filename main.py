from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

import uvicorn
from core.config import settings
from core.logging import logger
from ml.model_manager import model_manager
from ml.model_manager_meta import model_manager_meta
from api.controllers import translate_controller, health_controller


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    logger.info(f"Starting {settings.APP_NAME} v{settings.APP_VERSION}")
    model_manager.load_models()
    model_manager_meta.load_models()

    yield

    # Shutdown
    logger.info("Shutting down application...")
    model_manager.cleanup()
    model_manager_meta.cleanup()

# Create FastAPI app
app = FastAPI(
    title=settings.APP_NAME,
    description="Multi-language translation service using Helsinki-NLP models",
    version=settings.APP_VERSION,
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(health_controller.router)
app.include_router(translate_controller.router)


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level=settings.LOG_LEVEL.lower(),
    )
