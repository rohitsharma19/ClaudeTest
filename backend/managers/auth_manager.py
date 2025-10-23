"""
Authentication Manager for RAG Application.

Handles user authentication and session management.
"""

from typing import Dict, Any, Optional
import secrets
from datetime import datetime, timedelta


class AuthManager:
    """Manages user authentication and sessions."""

    def __init__(self, auth_config: Dict[str, Any]):
        """
        Initialize authentication manager.

        Args:
            auth_config: Authentication configuration dictionary
        """
        self.auth_config = auth_config
        self.username = auth_config.get('username', 'admin')
        self.password = auth_config.get('password', 'admin123')

        # In-memory session storage (session_token -> user_info)
        self.sessions: Dict[str, Dict[str, Any]] = {}

        # Session expiry time (24 hours)
        self.session_expiry_hours = 24

    def authenticate(self, username: str, password: str) -> Optional[str]:
        """
        Authenticate user and create session.

        Args:
            username: Username
            password: Password

        Returns:
            Session token if authentication successful, None otherwise
        """
        if username == self.username and password == self.password:
            # Generate session token
            session_token = secrets.token_urlsafe(32)

            # Store session
            self.sessions[session_token] = {
                'username': username,
                'created_at': datetime.now(),
                'expires_at': datetime.now() + timedelta(hours=self.session_expiry_hours)
            }

            return session_token

        return None

    def validate_session(self, session_token: str) -> bool:
        """
        Validate if session token is valid and not expired.

        Args:
            session_token: Session token to validate

        Returns:
            True if session is valid, False otherwise
        """
        if session_token not in self.sessions:
            return False

        session = self.sessions[session_token]

        # Check if session expired
        if datetime.now() > session['expires_at']:
            # Remove expired session
            del self.sessions[session_token]
            return False

        return True

    def get_user(self, session_token: str) -> Optional[str]:
        """
        Get username from session token.

        Args:
            session_token: Session token

        Returns:
            Username if session is valid, None otherwise
        """
        if not self.validate_session(session_token):
            return None

        return self.sessions[session_token]['username']

    def logout(self, session_token: str) -> bool:
        """
        Logout user by removing session.

        Args:
            session_token: Session token

        Returns:
            True if logout successful, False otherwise
        """
        if session_token in self.sessions:
            del self.sessions[session_token]
            return True

        return False

    def cleanup_expired_sessions(self) -> None:
        """Remove all expired sessions."""
        expired_tokens = []

        for token, session in self.sessions.items():
            if datetime.now() > session['expires_at']:
                expired_tokens.append(token)

        for token in expired_tokens:
            del self.sessions[token]
