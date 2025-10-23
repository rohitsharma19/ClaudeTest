"""
Excel Document Processor for RAG Application.
"""

from typing import List, Dict, Any, Tuple
from .base_processor import BaseDocumentProcessor
import os


class ExcelProcessor(BaseDocumentProcessor):
    """Excel document processor."""

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """Initialize Excel processor."""
        super().__init__(chunk_size, chunk_overlap)

    def extract_text(self, file_path: str, sheet_name: int = 0) -> str:
        """
        Extract text from Excel file.

        Args:
            file_path: Path to Excel file
            sheet_name: Sheet index to process (default: 0 for first sheet)

        Returns:
            Extracted text

        Raises:
            Exception: If Excel extraction fails
        """
        try:
            import pandas as pd

            # Read Excel file
            df = pd.read_excel(file_path, sheet_name=sheet_name)

            # Convert to text representation
            text = df.to_string(index=False)

            return text
        except ImportError:
            raise ImportError("pandas and openpyxl packages required. Run: pip install pandas openpyxl")
        except Exception as e:
            raise Exception(f"Excel extraction error: {str(e)}")

    def extract_rows_as_text(self, file_path: str, sheet_name: int = 0) -> List[str]:
        """
        Extract each row from Excel as separate text entries.

        Args:
            file_path: Path to Excel file
            sheet_name: Sheet index to process

        Returns:
            List of text representations of rows

        Raises:
            Exception: If Excel extraction fails
        """
        try:
            import pandas as pd

            # Read Excel file
            df = pd.read_excel(file_path, sheet_name=sheet_name)

            rows_text = []

            # Get column headers
            headers = df.columns.tolist()

            # Process each row
            for idx, row in df.iterrows():
                # Create text representation of row
                row_parts = []
                for col in headers:
                    value = row[col]
                    if pd.notna(value):  # Skip NaN values
                        row_parts.append(f"{col}: {value}")

                if row_parts:
                    row_text = " | ".join(row_parts)
                    rows_text.append(row_text)

            return rows_text
        except Exception as e:
            raise Exception(f"Excel row extraction error: {str(e)}")

    def process_document(
        self,
        file_path: str,
        filename: str,
        user: str = "system"
    ) -> Tuple[List[str], List[Dict[str, Any]]]:
        """
        Process Excel document into chunks with metadata.

        Each row becomes a separate chunk as per requirements.

        Args:
            file_path: Path to Excel file
            filename: Original filename
            user: User who uploaded

        Returns:
            Tuple of (chunks, metadatas)
        """
        try:
            # Extract rows as separate text entries
            rows_text = self.extract_rows_as_text(file_path)

            all_chunks = []
            all_metadatas = []

            # Each row is a chunk
            for row_num, row_text in enumerate(rows_text, start=1):
                all_chunks.append(row_text)

                metadata = self.extract_metadata(
                    file_path=file_path,
                    filename=filename,
                    user=user,
                    file_type='excel',
                    row_number=row_num
                )
                all_metadatas.append(metadata)

            return all_chunks, all_metadatas
        except Exception as e:
            raise Exception(f"Excel processing error: {str(e)}")
