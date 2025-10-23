"""
Query Routes for RAG Application.
"""

from flask import Blueprint, request, jsonify, current_app
import json
import os
from datetime import datetime
from pathlib import Path

bp = Blueprint('query', __name__)

CHAT_HISTORY_DIR = './data/chat_history'
Path(CHAT_HISTORY_DIR).mkdir(parents=True, exist_ok=True)


def get_chat_history_path(username):
    """Get path to chat history file for user."""
    return os.path.join(CHAT_HISTORY_DIR, f"{username}_history.json")


def load_chat_history(username):
    """Load chat history for user."""
    import os
    history_path = get_chat_history_path(username)
    if os.path.exists(history_path):
        try:
            with open(history_path, 'r') as f:
                return json.load(f)
        except:
            return []
    return []


def save_chat_history(username, history):
    """Save chat history for user."""
    import os
    history_path = get_chat_history_path(username)
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)


@bp.route('/query', methods=['POST'])
def query():
    """
    Submit a question and get an answer.

    Expected JSON:
        {
            "session_token": "string",
            "question": "string",
            "top_k": int (optional)
        }

    Returns:
        JSON with answer, sources, and metadata
    """
    try:
        data = request.get_json()

        if not data or 'session_token' not in data or 'question' not in data:
            return jsonify({'error': 'Session token and question required'}), 400

        session_token = data['session_token']
        question = data['question']
        top_k = data.get('top_k')

        # Check authentication
        auth_manager = current_app.config['AUTH_MANAGER']
        username = auth_manager.get_user(session_token)

        if not username:
            return jsonify({'error': 'Invalid session'}), 401

        logger = current_app.config['LOGGER']
        config = current_app.config['CONFIG_MANAGER']
        llm = current_app.config['LLM']
        retrieval_manager = current_app.config['RETRIEVAL_MANAGER']

        logger.info(f"Query from {username}: {question}")

        # Retrieve relevant documents
        retrieved_docs = retrieval_manager.retrieve(question, top_k)

        logger.info(f"Retrieved {len(retrieved_docs)} documents")

        # Format context for LLM
        context = retrieval_manager.format_context(retrieved_docs)

        # Get system prompt
        system_prompt = config.get_system_prompt()

        # Generate answer
        answer = llm.generate_response(
            prompt=question,
            context=context,
            system_prompt=system_prompt
        )

        logger.info(f"Answer generated for {username}")

        # Get source information
        sources = retrieval_manager.get_sources(retrieved_docs)

        # Prepare response
        response_data = {
            'answer': answer,
            'sources': sources,
            'question': question,
            'timestamp': datetime.now().isoformat()
        }

        # Save to chat history
        try:
            history = load_chat_history(username)
            history.append(response_data)
            save_chat_history(username, history)
        except Exception as e:
            logger.warning(f"Failed to save chat history: {str(e)}")

        return jsonify(response_data), 200

    except Exception as e:
        current_app.config['LOGGER'].error(f"Query error: {str(e)}")
        return jsonify({'error': f'Query failed: {str(e)}'}), 500


@bp.route('/chat/history', methods=['GET'])
def get_history():
    """
    Get chat history for user.

    Query params:
        session_token: Session token for authentication

    Returns:
        JSON with chat history
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

        # Load chat history
        history = load_chat_history(username)

        return jsonify({'history': history}), 200

    except Exception as e:
        current_app.config['LOGGER'].error(f"Get history error: {str(e)}")
        return jsonify({'error': 'Failed to retrieve chat history'}), 500


@bp.route('/chat/history', methods=['DELETE'])
def clear_history():
    """
    Clear chat history for user.

    Query params:
        session_token: Session token for authentication

    Returns:
        JSON with success status
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

        logger = current_app.config['LOGGER']

        # Clear history
        save_chat_history(username, [])

        logger.info(f"Chat history cleared for {username}")

        return jsonify({'success': True, 'message': 'Chat history cleared'}), 200

    except Exception as e:
        current_app.config['LOGGER'].error(f"Clear history error: {str(e)}")
        return jsonify({'error': 'Failed to clear chat history'}), 500
