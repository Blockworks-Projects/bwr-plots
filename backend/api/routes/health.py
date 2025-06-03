"""
Health check endpoints

Provides health check and status endpoints for monitoring.
"""

from fastapi import APIRouter
from datetime import datetime
import sys
import os

router = APIRouter()


@router.get("/health")
async def health_check():
    """Basic health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "service": "BWR Plots API"
    }


@router.get("/health/detailed")
async def detailed_health_check():
    """Detailed health check with system information"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "service": "BWR Plots API",
        "version": "1.0.0",
        "python_version": sys.version,
        "system_info": {
            "platform": sys.platform,
            "python_executable": sys.executable,
            "current_working_directory": os.getcwd()
        },
        "storage": {
            "uploads_dir_exists": os.path.exists("storage/uploads"),
            "sessions_dir_exists": os.path.exists("storage/sessions")
        }
    } 