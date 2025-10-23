"""
Document Routes for RAG Application.
"""

from flask import Blueprint, request, jsonify, current_app
from werkzeug.utils import secure_filename
import os
from pathlib import Path
import json

from processors.pdf_processor import PDFProcessor
from processors.excel_processor import ExcelProcessor
from utils.helpers import generate_doc_id, sanitize_filename

bp = Blueprint('documents', __name__)

UPLOAD_FOLDER = './data/uploads'
ALLOWED_EXTENSIONS = {'pdf', 'xlsx', 'xls'}


def allowed_file(filename):
    """Check if file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def get_processor(file_extension, chunk_size, chunk_overlap):
    """Get appropriate document processor based on file extension."""
    if file_extension == 'pdf':
        return PDFProcessor(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    elif file_extension in ['xlsx', 'xls']:
        return ExcelProcessor(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    else:
        raise ValueError(f"Unsupported file type: {file_extension}")


@bp.route('/upload', methods=['POST'])
def upload_documents():
    """
    Upload and process documents.

    Expected: multipart/form-data with files and session_token

    Returns:
        JSON with upload status and processed document info
    """
    try:
        # Check authentication
        session_token = request.form.get('session_token')
        if not session_token:
            return jsonify({'error': 'Authentication required'}), 401

        auth_manager = current_app.config['AUTH_MANAGER']
        username = auth_manager.get_user(session_token)

        if not username:
            return jsonify({'error': 'Invalid session'}), 401

        # Check if files were uploaded
        if 'files' not in request.files:
            return jsonify({'error': 'No files provided'}), 400

        files = request.files.getlist('files')

        if not files or all(f.filename == '' for f in files):
            return jsonify({'error': 'No files selected'}), 400

        logger = current_app.config['LOGGER']
        config = current_app.config['CONFIG_MANAGER']
        embedding = current_app.config['EMBEDDING']
        vector_db = current_app.config['VECTOR_DB']

        # Get chunking configuration
        chunk_size = config.get('chunking.chunk_size', 1000)
        chunk_overlap = config.get('chunking.chunk_overlap', 200)

        uploaded_files = []
        errors = []

        for file in files:
            if file and file.filename and allowed_file(file.filename):
                try:
                    # Secure and sanitize filename
                    original_filename = secure_filename(file.filename)
                    filename = sanitize_filename(original_filename)

                    # Generate unique doc ID
                    doc_id = generate_doc_id()

                    # Save file
                    file_path = os.path.join(UPLOAD_FOLDER, f"{doc_id}_{filename}")
                    file.save(file_path)

                    logger.info(f"File uploaded: {filename} by {username}")

                    # Get file extension
                    file_extension = filename.rsplit('.', 1)[1].lower()

                    # Process document
                    processor = get_processor(file_extension, chunk_size, chunk_overlap)
                    chunks, metadatas = processor.process_document(
                        file_path=file_path,
                        filename=filename,
                        user=username
                    )

                    # Add doc_id to all metadata
                    for metadata in metadatas:
                        metadata['doc_id'] = doc_id

                    logger.info(f"Document processed: {filename} - {len(chunks)} chunks")

                    # Generate embeddings
                    embeddings = embedding.embed_batch(chunks)

                    logger.info(f"Embeddings generated for: {filename}")

                    # Store in vector database
                    vector_db.add_documents(
                        chunks=chunks,
                        embeddings=embeddings,
                        metadatas=metadatas
                    )

                    logger.info(f"Document indexed: {filename}")

                    uploaded_files.append({
                        'doc_id': doc_id,
                        'filename': filename,
                        'chunks': len(chunks),
                        'file_type': file_extension
                    })

                except Exception as e:
                    logger.error(f"Error processing {file.filename}: {str(e)}")
                    errors.append({
                        'filename': file.filename,
                        'error': f"Failed to process file: {str(e)}"
                    })
            else:
                errors.append({
                    'filename': file.filename if file else 'unknown',
                    'error': 'Invalid file type. Only PDF and Excel files are allowed.'
                })

        response = {
            'success': len(uploaded_files) > 0,
            'uploaded': uploaded_files,
            'errors': errors
        }

        return jsonify(response), 200 if len(uploaded_files) > 0 else 400

    except Exception as e:
        current_app.config['LOGGER'].error(f"Upload error: {str(e)}")
        return jsonify({'error': f'Upload failed: {str(e)}'}), 500


@bp.route('/', methods=['GET'])
def list_documents():
    """
    List all uploaded documents.

    Query params:
        session_token: Session token for authentication

    Returns:
        JSON with list of documents
    """
    try:
        # Check authentication
        session_token = request.args.get('session_token')
        if not session_token:
            return jsonify({'error': 'Authentication required'}), 401

        auth_manager = current_app.config['AUTH_MANAGER']
        username = auth_manager.get_user(session_token)

        if not username:
            return jsonify({'error': 'Invalid session'}), 401

        vector_db = current_app.config['VECTOR_DB']

        # Get all documents
        documents = vector_db.get_all_documents()

        return jsonify({'documents': documents}), 200

    except Exception as e:
        current_app.config['LOGGER'].error(f"List documents error: {str(e)}")
        return jsonify({'error': 'Failed to retrieve documents'}), 500


@bp.route('/<doc_id>', methods=['DELETE'])
def delete_document(doc_id):
    """
    Delete a document.

    Args:
        doc_id: Document ID to delete

    Query params:
        session_token: Session token for authentication

    Returns:
        JSON with deletion status
    """
    try:
        # Check authentication
        session_token = request.args.get('session_token')
        if not session_token:
            return jsonify({'error': 'Authentication required'}), 401

        auth_manager = current_app.config['AUTH_MANAGER']
        username = auth_manager.get_user(session_token)

        if not username:
            return jsonify({'error': 'Invalid session'}), 401

        logger = current_app.config['LOGGER']
        vector_db = current_app.config['VECTOR_DB']

        # Delete from vector database
        vector_db.delete_document(doc_id)

        # Delete file from uploads folder
        upload_files = Path(UPLOAD_FOLDER).glob(f"{doc_id}_*")
        for file_path in upload_files:
            file_path.unlink()

        logger.info(f"Document deleted: {doc_id} by {username}")

        return jsonify({'success': True, 'message': 'Document deleted'}), 200

    except Exception as e:
        current_app.config['LOGGER'].error(f"Delete document error: {str(e)}")
        return jsonify({'error': 'Failed to delete document'}), 500
