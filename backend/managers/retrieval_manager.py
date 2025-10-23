"""
Retrieval Manager for RAG Application.

Orchestrates the retrieval process for question answering.
"""

from typing import Dict, Any, List
from .embedding_manager import BaseEmbedding
from .vector_db_manager import BaseVectorDB


class RetrievalManager:
    """Manages document retrieval for RAG."""

    def __init__(
        self,
        embedding_model: BaseEmbedding,
        vector_db: BaseVectorDB,
        top_k: int = 5
    ):
        """
        Initialize retrieval manager.

        Args:
            embedding_model: Embedding model instance
            vector_db: Vector database instance
            top_k: Number of documents to retrieve
        """
        self.embedding_model = embedding_model
        self.vector_db = vector_db
        self.top_k = top_k

    def retrieve(self, query: str, top_k: int = None) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents for a query.

        Args:
            query: User query
            top_k: Number of results to retrieve (overrides default if provided)

        Returns:
            List of retrieved documents with metadata and scores

        Raises:
            Exception: If retrieval fails
        """
        try:
            # Use provided top_k or default
            k = top_k if top_k is not None else self.top_k

            # Generate query embedding
            query_embedding = self.embedding_model.embed_text(query)

            # Search vector database
            results = self.vector_db.search(query_embedding, k)

            return results
        except Exception as e:
            raise Exception(f"Retrieval error: {str(e)}")

    def format_context(self, retrieved_docs: List[Dict[str, Any]]) -> str:
        """
        Format retrieved documents into context string for LLM.

        Args:
            retrieved_docs: List of retrieved documents

        Returns:
            Formatted context string
        """
        if not retrieved_docs:
            return "No relevant context found."

        context_parts = []
        for i, doc in enumerate(retrieved_docs, 1):
            metadata = doc.get('metadata', {})
            text = doc.get('text', '')
            score = doc.get('score', 0)

            source_info = f"Source {i}"
            if metadata.get('filename'):
                source_info += f" - {metadata['filename']}"
            if metadata.get('page_number'):
                source_info += f" (Page {metadata['page_number']})"
            elif metadata.get('row_number'):
                source_info += f" (Row {metadata['row_number']})"

            context_parts.append(f"[{source_info}]\n{text}\n")

        return "\n".join(context_parts)

    def get_sources(self, retrieved_docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Extract source information from retrieved documents.

        Args:
            retrieved_docs: List of retrieved documents

        Returns:
            List of source information dictionaries
        """
        sources = []
        for doc in retrieved_docs:
            metadata = doc.get('metadata', {})
            source = {
                'text': doc.get('text', ''),
                'score': doc.get('score', 0),
                'filename': metadata.get('filename', 'Unknown'),
                'file_type': metadata.get('file_type', 'unknown')
            }

            # Add page or row number if available
            if metadata.get('page_number'):
                source['page_number'] = metadata['page_number']
            elif metadata.get('row_number'):
                source['row_number'] = metadata['row_number']

            sources.append(source)

        return sources
