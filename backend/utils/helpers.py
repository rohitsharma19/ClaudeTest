"""
Helper utilities for RAG Application.
"""

import os
import hashlib
from datetime import datetime
from typing import Dict, Any, List
import uuid


def generate_doc_id() -> str:
    """
    Generate a unique document ID.

    Returns:
        Unique document identifier
    """
    return str(uuid.uuid4())


def get_file_hash(file_path: str) -> str:
    """
    Calculate MD5 hash of a file.

    Args:
        file_path: Path to file

    Returns:
        MD5 hash of file contents
    """
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def get_timestamp() -> str:
    """
    Get current timestamp in ISO format.

    Returns:
        ISO formatted timestamp string
    """
    return datetime.now().isoformat()


def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename to remove problematic characters.

    Args:
        filename: Original filename

    Returns:
        Sanitized filename
    """
    # Remove or replace problematic characters
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        filename = filename.replace(char, '_')
    return filename


def format_metadata(
    filename: str,
    file_type: str,
    upload_date: str,
    user: str,
    page_number: int = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Format metadata dictionary for document chunks.

    Args:
        filename: Name of the file
        file_type: Type of file (pdf, excel)
        upload_date: Date of upload
        user: User who uploaded
        page_number: Page number (for PDFs)
        **kwargs: Additional metadata fields

    Returns:
        Formatted metadata dictionary
    """
    metadata = {
        'filename': filename,
        'file_type': file_type,
        'upload_date': upload_date,
        'user': user
    }

    if page_number is not None:
        metadata['page_number'] = page_number

    # Add any additional metadata
    metadata.update(kwargs)

    return metadata


def chunk_text(
    text: str,
    chunk_size: int,
    chunk_overlap: int
) -> List[str]:
    """
    Split text into chunks with overlap.

    Args:
        text: Text to chunk
        chunk_size: Size of each chunk in characters
        chunk_overlap: Number of overlapping characters

    Returns:
        List of text chunks
    """
    if not text:
        return []

    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)

        # Move to next chunk with overlap
        start = end - chunk_overlap

        # Ensure we make progress
        if chunk_overlap >= chunk_size:
            start = end

    return chunks


def ensure_directory_exists(directory_path: str) -> None:
    """
    Ensure a directory exists, create if it doesn't.

    Args:
        directory_path: Path to directory
    """
    os.makedirs(directory_path, exist_ok=True)
