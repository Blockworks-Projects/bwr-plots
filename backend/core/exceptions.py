"""
Custom exception handlers for the BWR Plots API

Defines custom exceptions and their handlers for consistent error responses.
"""

from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse
from typing import Any, Dict
import logging

logger = logging.getLogger(__name__)


class BWRPlotsException(Exception):
    """Base exception for BWR Plots application"""
    
    def __init__(self, message: str, status_code: int = 500, details: Dict[str, Any] = None):
        self.message = message
        self.status_code = status_code
        self.details = details or {}
        super().__init__(self.message)


class FileUploadException(BWRPlotsException):
    """Exception raised during file upload operations"""
    
    def __init__(self, message: str, details: Dict[str, Any] = None):
        super().__init__(message, status_code=400, details=details)


class DataProcessingException(BWRPlotsException):
    """Exception raised during data processing operations"""
    
    def __init__(self, message: str, details: Dict[str, Any] = None):
        super().__init__(message, status_code=422, details=details)


class PlotGenerationException(BWRPlotsException):
    """Exception raised during plot generation"""
    
    def __init__(self, message: str, details: Dict[str, Any] = None):
        super().__init__(message, status_code=422, details=details)


class SessionNotFoundException(BWRPlotsException):
    """Exception raised when session is not found"""
    
    def __init__(self, session_id: str):
        message = f"Session {session_id} not found or expired"
        super().__init__(message, status_code=404, details={"session_id": session_id})


async def bwr_plots_exception_handler(request: Request, exc: BWRPlotsException) -> JSONResponse:
    """Handler for BWR Plots custom exceptions"""
    logger.error(f"BWR Plots Exception: {exc.message}", extra={"details": exc.details})
    
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": True,
            "message": exc.message,
            "details": exc.details,
            "type": exc.__class__.__name__
        }
    )


async def general_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handler for general exceptions"""
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    
    return JSONResponse(
        status_code=500,
        content={
            "error": True,
            "message": "Internal server error",
            "details": {},
            "type": "InternalServerError"
        }
    )


async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    """Handler for HTTP exceptions"""
    logger.warning(f"HTTP Exception: {exc.detail}")
    
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": True,
            "message": exc.detail,
            "details": {},
            "type": "HTTPException"
        }
    ) 