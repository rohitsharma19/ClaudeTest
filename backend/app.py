"""
Flask Application for RAG Backend.

Main entry point for the Flask REST API.
"""

from flask import Flask, jsonify
from flask_cors import CORS
import os
from pathlib import Path

from config.config_manager import get_config
from utils.logger import get_logger
from managers.llm_manager import LLMFactory
from managers.embedding_manager import EmbeddingFactory
from managers.vector_db_manager import VectorDBFactory
from managers.auth_manager import AuthManager
from managers.retrieval_manager import RetrievalManager

# Import routes
from routes import auth_routes, document_routes, query_routes, config_routes


def create_app(config_path: str = "backend/config/config.json"):
    """
    Create and configure Flask application.

    Args:
        config_path: Path to configuration file

    Returns:
        Configured Flask app
    """
    app = Flask(__name__)
    app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size

    # Enable CORS for Streamlit frontend
    CORS(app)

    # Load configuration
    config = get_config(config_path)

    # Setup logger
    logger = get_logger(__name__, config.get_logging_config())
    logger.info("Starting RAG Application Backend")

    # Ensure required directories exist
    Path("./data/uploads").mkdir(parents=True, exist_ok=True)
    Path("./data/vectordb").mkdir(parents=True, exist_ok=True)
    Path("./logs").mkdir(parents=True, exist_ok=True)

    # Initialize components
    try:
        logger.info("Initializing components...")

        # Initialize managers
        llm = LLMFactory.create_llm(config.get_llm_config())
        logger.info(f"LLM initialized: {config.get('llm.provider')}")

        embedding = EmbeddingFactory.create_embedding(config.get_embeddings_config())
        logger.info(f"Embeddings initialized: {config.get('embeddings.provider')}")

        vector_db = VectorDBFactory.create_vector_db(config.get_vector_db_config())
        logger.info(f"Vector DB initialized: {config.get('vector_db.type')}")

        auth_manager = AuthManager(config.get_auth_config())
        logger.info("Authentication manager initialized")

        retrieval_manager = RetrievalManager(
            embedding_model=embedding,
            vector_db=vector_db,
            top_k=config.get('retrieval.top_k', 5)
        )
        logger.info("Retrieval manager initialized")

        # Store components in app config for access in routes
        app.config['CONFIG_MANAGER'] = config
        app.config['LLM'] = llm
        app.config['EMBEDDING'] = embedding
        app.config['VECTOR_DB'] = vector_db
        app.config['AUTH_MANAGER'] = auth_manager
        app.config['RETRIEVAL_MANAGER'] = retrieval_manager
        app.config['LOGGER'] = logger

        logger.info("All components initialized successfully")

    except Exception as e:
        logger.error(f"Failed to initialize components: {str(e)}")
        raise

    # Register blueprints
    app.register_blueprint(auth_routes.bp, url_prefix='/api/auth')
    app.register_blueprint(document_routes.bp, url_prefix='/api/documents')
    app.register_blueprint(query_routes.bp, url_prefix='/api')
    app.register_blueprint(config_routes.bp, url_prefix='/api/config')

    logger.info("Routes registered")

    # Health check endpoint
    @app.route('/health', methods=['GET'])
    def health_check():
        """Health check endpoint."""
        return jsonify({'status': 'healthy'}), 200

    # Error handlers
    @app.errorhandler(404)
    def not_found(error):
        """Handle 404 errors."""
        return jsonify({'error': 'Endpoint not found'}), 404

    @app.errorhandler(500)
    def internal_error(error):
        """Handle 500 errors."""
        logger.error(f"Internal server error: {str(error)}")
        return jsonify({'error': 'Internal server error'}), 500

    return app


if __name__ == '__main__':
    app = create_app()
    app.run(host='0.0.0.0', port=5000, debug=True)
