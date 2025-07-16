"""
BWR Plots Backend API

FastAPI application for the BWR Plots frontend refactor.
Provides REST API endpoints for data processing and plot generation.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging

from core.config import settings
from api.routes import health
from api.middleware import setup_all_middleware

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    logger.info("Starting BWR Plots Backend API")
    yield
    logger.info("Shutting down BWR Plots Backend API")


# Create FastAPI application
app = FastAPI(
    title="BWR Plots API",
    description="Backend API for BWR Plots data visualization application",
    version="1.0.0",
    lifespan=lifespan
)

# Configure middleware
setup_all_middleware(app, settings.CORS_ORIGINS)

# Include routers
app.include_router(health.router, prefix="/api/v1", tags=["health"])

# Import and include data routes
from api.routes import data
app.include_router(data.router, prefix="/api/v1/data", tags=["data"])

# Import and include plot routes
from api.routes import plots
app.include_router(plots.router, prefix="/api/v1", tags=["plots"])

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "BWR Plots Backend API",
        "version": "1.0.0",
        "status": "running"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8005,
        reload=True,
        log_level="info"
    ) 