"""
API Middleware

Middleware configurations for the FastAPI application.
"""

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
import time
import logging

logger = logging.getLogger(__name__)


def setup_cors_middleware(app: FastAPI, cors_origins: list):
    """
    Set up CORS middleware for the FastAPI application
    
    Args:
        app: FastAPI application instance
        cors_origins: List of allowed origins
    """
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=["*"],
        expose_headers=["*"]
    )


def setup_security_middleware(app: FastAPI):
    """
    Set up security middleware
    
    Args:
        app: FastAPI application instance
    """
    # Add trusted host middleware for production
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=["*"]  # Configure this properly for production
    )


async def add_process_time_header(request: Request, call_next):
    """
    Add processing time header to responses
    """
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response


async def log_requests(request: Request, call_next):
    """
    Log all incoming requests
    """
    start_time = time.time()
    
    # Log request
    logger.info(f"Request: {request.method} {request.url}")
    
    # Process request
    response = await call_next(request)
    
    # Log response
    process_time = time.time() - start_time
    logger.info(
        f"Response: {response.status_code} - {process_time:.4f}s"
    )
    
    return response


def setup_all_middleware(app: FastAPI, cors_origins: list):
    """
    Set up all middleware for the application
    
    Args:
        app: FastAPI application instance
        cors_origins: List of allowed CORS origins
    """
    setup_cors_middleware(app, cors_origins)
    setup_security_middleware(app)
    
    # Add custom middleware
    app.middleware("http")(add_process_time_header)
    app.middleware("http")(log_requests) 