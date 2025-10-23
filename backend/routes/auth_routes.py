"""
Authentication Routes for RAG Application.
"""

from flask import Blueprint, request, jsonify, current_app

bp = Blueprint('auth', __name__)


@bp.route('/login', methods=['POST'])
def login():
    """
    User login endpoint.

    Expected JSON:
        {
            "username": "string",
            "password": "string"
        }

    Returns:
        JSON with session token or error
    """
    try:
        data = request.get_json()

        if not data or 'username' not in data or 'password' not in data:
            return jsonify({'error': 'Username and password required'}), 400

        username = data['username']
        password = data['password']

        auth_manager = current_app.config['AUTH_MANAGER']
        logger = current_app.config['LOGGER']

        # Authenticate
        session_token = auth_manager.authenticate(username, password)

        if session_token:
            logger.info(f"User logged in: {username}")
            return jsonify({
                'success': True,
                'session_token': session_token,
                'username': username
            }), 200
        else:
            logger.warning(f"Failed login attempt: {username}")
            return jsonify({'error': 'Invalid username or password'}), 401

    except Exception as e:
        current_app.config['LOGGER'].error(f"Login error: {str(e)}")
        return jsonify({'error': 'Login failed'}), 500


@bp.route('/logout', methods=['POST'])
def logout():
    """
    User logout endpoint.

    Expected JSON:
        {
            "session_token": "string"
        }

    Returns:
        JSON with success status
    """
    try:
        data = request.get_json()

        if not data or 'session_token' not in data:
            return jsonify({'error': 'Session token required'}), 400

        session_token = data['session_token']

        auth_manager = current_app.config['AUTH_MANAGER']
        logger = current_app.config['LOGGER']

        # Logout
        success = auth_manager.logout(session_token)

        if success:
            logger.info("User logged out")
            return jsonify({'success': True}), 200
        else:
            return jsonify({'error': 'Invalid session'}), 401

    except Exception as e:
        current_app.config['LOGGER'].error(f"Logout error: {str(e)}")
        return jsonify({'error': 'Logout failed'}), 500


@bp.route('/validate', methods=['POST'])
def validate_session():
    """
    Validate session token.

    Expected JSON:
        {
            "session_token": "string"
        }

    Returns:
        JSON with validation status and username
    """
    try:
        data = request.get_json()

        if not data or 'session_token' not in data:
            return jsonify({'error': 'Session token required'}), 400

        session_token = data['session_token']

        auth_manager = current_app.config['AUTH_MANAGER']

        # Validate session
        username = auth_manager.get_user(session_token)

        if username:
            return jsonify({
                'valid': True,
                'username': username
            }), 200
        else:
            return jsonify({'valid': False}), 401

    except Exception as e:
        current_app.config['LOGGER'].error(f"Session validation error: {str(e)}")
        return jsonify({'error': 'Validation failed'}), 500
