"""
Authentication and dependency injection utilities.

This module provides dependency functions for FastAPI endpoints,
including user authentication and authorization.
"""

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)

# Security scheme for JWT tokens (optional)
security = HTTPBearer(auto_error=False)


async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
) -> Dict[str, Any]:
    """
    Get current authenticated user.
    
    For now, this is a placeholder that returns a mock user.
    In a production environment, this would validate JWT tokens
    and return actual user information.
    
    Args:
        credentials: Optional HTTP authorization credentials
        
    Returns:
        Dict containing user information
        
    Raises:
        HTTPException: If authentication fails (in production)
    """
    # For development/testing, return a mock user
    # In production, you would validate the JWT token here
    mock_user = {
        "user_id": "test_user_123",
        "username": "test_user",
        "email": "test@example.com",
        "is_active": True,
        "permissions": ["read", "write"]
    }
    
    # If credentials are provided, you could validate them here
    if credentials:
        logger.info(f"User authenticated with token: {credentials.credentials[:10]}...")
    else:
        logger.info("No authentication credentials provided, using mock user")
    
    return mock_user


async def get_admin_user(
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Get current user and verify admin permissions.
    
    Args:
        current_user: Current authenticated user
        
    Returns:
        Dict containing admin user information
        
    Raises:
        HTTPException: If user is not an admin
    """
    # Check if user has admin permissions
    if "admin" not in current_user.get("permissions", []):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin permissions required"
        )
    
    return current_user


def get_optional_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
) -> Optional[Dict[str, Any]]:
    """
    Get current user if authenticated, otherwise return None.
    
    This is useful for endpoints that work for both authenticated
    and anonymous users.
    
    Args:
        credentials: Optional HTTP authorization credentials
        
    Returns:
        Dict containing user information or None
    """
    if credentials:
        # In production, validate the token here
        return {
            "user_id": "test_user_123",
            "username": "test_user",
            "email": "test@example.com",
            "is_active": True,
            "permissions": ["read", "write"]
        }
    
    return None 