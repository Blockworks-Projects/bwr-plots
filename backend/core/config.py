"""
Application configuration settings

Manages environment variables and application settings using Pydantic.
"""

from pydantic_settings import BaseSettings
from typing import List
import os


class Settings(BaseSettings):
    """Application settings"""
    
    # API Configuration
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "BWR Plots API"
    VERSION: str = "1.0.0"
    
    # CORS Configuration
    CORS_ORIGINS: List[str] = [
        "http://localhost:3000",  # Next.js development server
        "http://127.0.0.1:3000",
        "http://localhost:8080",  # Alternative frontend port
        "http://127.0.0.1:8080",
    ]
    
    # File Upload Configuration
    MAX_FILE_SIZE: int = 100 * 1024 * 1024  # 100MB
    ALLOWED_FILE_TYPES: List[str] = [".csv", ".xlsx", ".xls"]
    UPLOAD_DIR: str = "storage/uploads"
    SESSION_DIR: str = "storage/sessions"
    
    # Session Configuration
    SESSION_TIMEOUT: int = 3600  # 1 hour in seconds
    SESSION_CLEANUP_INTERVAL: int = 300  # 5 minutes in seconds
    
    # Redis Configuration (for future use)
    REDIS_URL: str = "redis://localhost:6379"
    USE_REDIS: bool = False
    
    # Logging Configuration
    LOG_LEVEL: str = "INFO"
    
    # Development Configuration
    DEBUG: bool = False
    RELOAD: bool = False
    
    class Config:
        env_file = ".env"
        case_sensitive = True


# Create settings instance
settings = Settings()

# Ensure storage directories exist
os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
os.makedirs(settings.SESSION_DIR, exist_ok=True) 