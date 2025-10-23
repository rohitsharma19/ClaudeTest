"""
PDF Document Processor for RAG Application.
"""

from typing import List, Dict, Any, Tuple
from .base_processor import BaseDocumentProcessor
import os


class PDFProcessor(BaseDocumentProcessor):
    """PDF document processor."""

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """Initialize PDF processor."""
        super().__init__(chunk_size, chunk_overlap)

    def extract_text(self, file_path: str) -> str:
        """
        Extract text from PDF file.

        Args:
            file_path: Path to PDF file

        Returns:
            Extracted text

        Raises:
            Exception: If PDF extraction fails
        """
        try:
            import PyPDF2

            text = ""
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                num_pages = len(pdf_reader.pages)

                for page_num in range(num_pages):
                    page = pdf_reader.pages[page_num]
                    page_text = page.extract_text()
                    if page_text:
                        text += f"\n--- Page {page_num + 1} ---\n{page_text}"

            return text
        except ImportError:
            raise ImportError("PyPDF2 package not installed. Run: pip install PyPDF2")
        except Exception as e:
            raise Exception(f"PDF extraction error: {str(e)}")

    def extract_text_with_pages(self, file_path: str) -> List[Tuple[str, int]]:
        """
        Extract text from PDF with page numbers.

        Args:
            file_path: Path to PDF file

        Returns:
            List of tuples (text, page_number)

        Raises:
            Exception: If PDF extraction fails
        """
        try:
            import PyPDF2

            pages_text = []
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                num_pages = len(pdf_reader.pages)

                for page_num in range(num_pages):
                    page = pdf_reader.pages[page_num]
                    page_text = page.extract_text()
                    if page_text:
                        pages_text.append((page_text, page_num + 1))

            return pages_text
        except Exception as e:
            raise Exception(f"PDF extraction error: {str(e)}")

    def process_document(
        self,
        file_path: str,
        filename: str,
        user: str = "system"
    ) -> Tuple[List[str], List[Dict[str, Any]]]:
        """
        Process PDF document into chunks with metadata.

        Args:
            file_path: Path to PDF file
            filename: Original filename
            user: User who uploaded

        Returns:
            Tuple of (chunks, metadatas)
        """
        try:
            # Extract text with page numbers
            pages_text = self.extract_text_with_pages(file_path)

            all_chunks = []
            all_metadatas = []

            # Process each page
            for page_text, page_num in pages_text:
                # Chunk the page text
                page_chunks = self.chunk_text(page_text)

                # Create metadata for each chunk
                for chunk in page_chunks:
                    all_chunks.append(chunk)

                    metadata = self.extract_metadata(
                        file_path=file_path,
                        filename=filename,
                        user=user,
                        file_type='pdf',
                        page_number=page_num
                    )
                    all_metadatas.append(metadata)

            return all_chunks, all_metadatas
        except Exception as e:
            raise Exception(f"PDF processing error: {str(e)}")
