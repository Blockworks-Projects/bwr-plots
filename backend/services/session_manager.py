"""
Session Management Service

Handles user sessions, data storage, and cleanup for the BWR Plots API.
"""

import uuid
import json
import os
import time
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
import pandas as pd
import logging

from core.config import settings
from core.exceptions import SessionNotFoundException, DataProcessingException

logger = logging.getLogger(__name__)


class SessionManager:
    """Manages user sessions and temporary data storage"""
    
    def __init__(self):
        self.sessions: Dict[str, Dict[str, Any]] = {}
        self.session_dir = settings.SESSION_DIR
        self.upload_dir = settings.UPLOAD_DIR
        self.session_timeout = settings.SESSION_TIMEOUT
        
        # Ensure directories exist
        os.makedirs(self.session_dir, exist_ok=True)
        os.makedirs(self.upload_dir, exist_ok=True)
    
    def create_session(self) -> str:
        """Create a new session and return session ID"""
        session_id = str(uuid.uuid4())
        session_data = {
            "id": session_id,
            "created_at": datetime.utcnow().isoformat(),
            "last_accessed": datetime.utcnow().isoformat(),
            "data": None,
            "original_filename": None,
            "file_path": None,
            "columns": [],
            "data_types": {},
            "row_count": 0,
            "manipulations": []
        }
        
        self.sessions[session_id] = session_data
        self._save_session(session_id)
        
        logger.info(f"Created new session: {session_id}")
        return session_id
    
    def get_session(self, session_id: str) -> Dict[str, Any]:
        """Get session data by ID"""
        if session_id not in self.sessions:
            # Try to load from disk
            self._load_session(session_id)
        
        if session_id not in self.sessions:
            raise SessionNotFoundException(session_id)
        
        # Update last accessed time
        self.sessions[session_id]["last_accessed"] = datetime.utcnow().isoformat()
        self._save_session(session_id)
        
        return self.sessions[session_id]
    
    def update_session(self, session_id: str, data: Dict[str, Any]) -> None:
        """Update session data"""
        session = self.get_session(session_id)
        session.update(data)
        session["last_accessed"] = datetime.utcnow().isoformat()
        self._save_session(session_id)
        
        logger.debug(f"Updated session: {session_id}")
    
    def store_dataframe(self, session_id: str, df: pd.DataFrame, filename: str = None) -> None:
        """Store DataFrame in session"""
        session = self.get_session(session_id)
        
        # Save DataFrame to parquet for efficient storage
        df_path = os.path.join(self.session_dir, f"{session_id}_data.parquet")
        df.to_parquet(df_path)
        
        # Update session metadata
        session.update({
            "data_path": df_path,
            "original_filename": filename,
            "columns": df.columns.tolist(),
            "data_types": df.dtypes.astype(str).to_dict(),
            "row_count": len(df),
            "last_accessed": datetime.utcnow().isoformat()
        })
        
        self._save_session(session_id)
        logger.info(f"Stored DataFrame for session {session_id}: {len(df)} rows, {len(df.columns)} columns")
    
    def get_dataframe(self, session_id: str) -> pd.DataFrame:
        """Retrieve DataFrame from session"""
        session = self.get_session(session_id)
        
        if "data_path" not in session or not os.path.exists(session["data_path"]):
            raise DataProcessingException("No data found in session")
        
        df = pd.read_parquet(session["data_path"])
        logger.debug(f"Retrieved DataFrame for session {session_id}: {len(df)} rows")
        return df
    
    def add_manipulation(self, session_id: str, manipulation: Dict[str, Any]) -> None:
        """Add a data manipulation to session history"""
        session = self.get_session(session_id)
        session["manipulations"].append({
            "timestamp": datetime.utcnow().isoformat(),
            "operation": manipulation
        })
        self._save_session(session_id)
    
    def delete_session(self, session_id: str) -> None:
        """Delete session and cleanup files"""
        if session_id in self.sessions:
            session = self.sessions[session_id]
            
            # Clean up files
            if "data_path" in session and os.path.exists(session["data_path"]):
                os.remove(session["data_path"])
            
            if "file_path" in session and os.path.exists(session["file_path"]):
                os.remove(session["file_path"])
            
            # Remove session file
            session_file = os.path.join(self.session_dir, f"{session_id}.json")
            if os.path.exists(session_file):
                os.remove(session_file)
            
            # Remove from memory
            del self.sessions[session_id]
            
            logger.info(f"Deleted session: {session_id}")
    
    def cleanup_expired_sessions(self) -> int:
        """Clean up expired sessions"""
        current_time = datetime.utcnow()
        expired_sessions = []
        
        for session_id, session_data in self.sessions.items():
            last_accessed = datetime.fromisoformat(session_data["last_accessed"])
            if (current_time - last_accessed).total_seconds() > self.session_timeout:
                expired_sessions.append(session_id)
        
        # Also check session files on disk
        if os.path.exists(self.session_dir):
            for filename in os.listdir(self.session_dir):
                if filename.endswith(".json"):
                    session_id = filename[:-5]  # Remove .json extension
                    session_file = os.path.join(self.session_dir, filename)
                    
                    try:
                        with open(session_file, 'r') as f:
                            session_data = json.load(f)
                        
                        last_accessed = datetime.fromisoformat(session_data["last_accessed"])
                        if (current_time - last_accessed).total_seconds() > self.session_timeout:
                            if session_id not in expired_sessions:
                                expired_sessions.append(session_id)
                    except (json.JSONDecodeError, KeyError, ValueError):
                        # Invalid session file, mark for deletion
                        expired_sessions.append(session_id)
        
        # Delete expired sessions
        for session_id in expired_sessions:
            try:
                self.delete_session(session_id)
            except Exception as e:
                logger.error(f"Error deleting expired session {session_id}: {e}")
        
        if expired_sessions:
            logger.info(f"Cleaned up {len(expired_sessions)} expired sessions")
        
        return len(expired_sessions)
    
    def _save_session(self, session_id: str) -> None:
        """Save session data to disk"""
        session_file = os.path.join(self.session_dir, f"{session_id}.json")
        session_data = self.sessions[session_id].copy()
        
        # Remove non-serializable data
        if "data" in session_data:
            del session_data["data"]
        
        with open(session_file, 'w') as f:
            json.dump(session_data, f, indent=2)
    
    def _load_session(self, session_id: str) -> None:
        """Load session data from disk"""
        session_file = os.path.join(self.session_dir, f"{session_id}.json")
        
        if os.path.exists(session_file):
            try:
                with open(session_file, 'r') as f:
                    session_data = json.load(f)
                self.sessions[session_id] = session_data
                logger.debug(f"Loaded session from disk: {session_id}")
            except (json.JSONDecodeError, KeyError) as e:
                logger.error(f"Error loading session {session_id}: {e}")
                # Clean up corrupted session file
                os.remove(session_file)


# Global session manager instance
session_manager = SessionManager() 