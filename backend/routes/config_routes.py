"""
Configuration Routes for RAG Application.
"""

from flask import Blueprint, request, jsonify, current_app

bp = Blueprint('config', __name__)


@bp.route('/', methods=['GET'])
def get_config():
    """
    Get current configuration.

    Query params:
        session_token: Session token for authentication

    Returns:
        JSON with configuration (sensitive info redacted)
    """
    try:
        session_token = request.args.get('session_token')

        if not session_token:
            return jsonify({'error': 'Session token required'}), 400

        # Check authentication
        auth_manager = current_app.config['AUTH_MANAGER']
        username = auth_manager.get_user(session_token)

        if not username:
            return jsonify({'error': 'Invalid session'}), 401

        config_manager = current_app.config['CONFIG_MANAGER']

        # Get configuration (redact sensitive info)
        config_data = config_manager.get_all()

        # Redact API keys
        if 'llm' in config_data and 'api_key' in config_data['llm']:
            config_data['llm']['api_key'] = '***' if config_data['llm']['api_key'] else ''

        if 'embeddings' in config_data and 'api_key' in config_data['embeddings']:
            config_data['embeddings']['api_key'] = '***' if config_data['embeddings']['api_key'] else ''

        if 'authentication' in config_data:
            config_data['authentication'] = {'username': config_data['authentication'].get('username', 'admin')}

        return jsonify({'config': config_data}), 200

    except Exception as e:
        current_app.config['LOGGER'].error(f"Get config error: {str(e)}")
        return jsonify({'error': 'Failed to retrieve configuration'}), 500


@bp.route('/', methods=['PUT'])
def update_config():
    """
    Update configuration parameters.

    Expected JSON:
        {
            "session_token": "string",
            "updates": {
                "chunking": {"chunk_size": 1000, "chunk_overlap": 200},
                "retrieval": {"top_k": 5}
            }
        }

    Returns:
        JSON with updated configuration
    """
    try:
        data = request.get_json()

        if not data or 'session_token' not in data or 'updates' not in data:
            return jsonify({'error': 'Session token and updates required'}), 400

        session_token = data['session_token']
        updates = data['updates']

        # Check authentication
        auth_manager = current_app.config['AUTH_MANAGER']
        username = auth_manager.get_user(session_token)

        if not username:
            return jsonify({'error': 'Invalid session'}), 401

        logger = current_app.config['LOGGER']
        config_manager = current_app.config['CONFIG_MANAGER']

        # Update configuration
        config_manager.update_config(updates)

        logger.info(f"Configuration updated by {username}: {updates}")

        # Update retrieval manager if top_k changed
        if 'retrieval' in updates and 'top_k' in updates['retrieval']:
            retrieval_manager = current_app.config['RETRIEVAL_MANAGER']
            retrieval_manager.top_k = updates['retrieval']['top_k']

        return jsonify({
            'success': True,
            'message': 'Configuration updated successfully'
        }), 200

    except Exception as e:
        current_app.config['LOGGER'].error(f"Update config error: {str(e)}")
        return jsonify({'error': f'Failed to update configuration: {str(e)}'}), 500
