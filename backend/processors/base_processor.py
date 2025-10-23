"""
Base Document Processor for RAG Application.

Provides abstract base class for document processing.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Tuple
from datetime import datetime


class BaseDocumentProcessor(ABC):
    """Abstract base class for document processors."""

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initialize document processor.

        Args:
            chunk_size: Size of text chunks in characters
            chunk_overlap: Overlap between chunks in characters
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    @abstractmethod
    def extract_text(self, file_path: str) -> str:
        """
        Extract text from document.

        Args:
            file_path: Path to document file

        Returns:
            Extracted text

        Raises:
            Exception: If text extraction fails
        """
        pass

    def chunk_text(self, text: str) -> List[str]:
        """
        Split text into chunks with overlap.

        Args:
            text: Text to chunk

        Returns:
            List of text chunks
        """
        if not text:
            return []

        chunks = []
        start = 0
        text_length = len(text)

        while start < text_length:
            end = start + self.chunk_size
            chunk = text[start:end]

            # Only add non-empty chunks
            if chunk.strip():
                chunks.append(chunk)

            # Move to next chunk with overlap
            start = end - self.chunk_overlap

            # Ensure we make progress
            if self.chunk_overlap >= self.chunk_size:
                start = end

        return chunks

    def extract_metadata(
        self,
        file_path: str,
        filename: str,
        user: str = "system",
        **kwargs
    ) -> Dict[str, Any]:
        """
        Extract metadata from document.

        Args:
            file_path: Path to document file
            filename: Original filename
            user: User who uploaded the document
            **kwargs: Additional metadata

        Returns:
            Metadata dictionary
        """
        metadata = {
            'filename': filename,
            'upload_date': datetime.now().isoformat(),
            'user': user,
            'file_path': file_path
        }

        metadata.update(kwargs)
        return metadata

    @abstractmethod
    def process_document(
        self,
        file_path: str,
        filename: str,
        user: str = "system"
    ) -> Tuple[List[str], List[Dict[str, Any]]]:
        """
        Process document into chunks with metadata.

        Args:
            file_path: Path to document file
            filename: Original filename
            user: User who uploaded

        Returns:
            Tuple of (chunks, metadatas)

        Raises:
            Exception: If processing fails
        """
        pass
