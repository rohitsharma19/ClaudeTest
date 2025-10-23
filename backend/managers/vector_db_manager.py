"""
Vector Database Manager for RAG Application.

Provides abstract base class and concrete implementations for vector databases.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Tuple
import os
from pathlib import Path


class BaseVectorDB(ABC):
    """Abstract base class for vector databases."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize vector database.

        Args:
            config: Vector DB configuration dictionary
        """
        self.config = config
        self.persist_directory = config.get('persist_directory', './data/vectordb')

        # Ensure directory exists
        Path(self.persist_directory).mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def add_documents(
        self,
        chunks: List[str],
        embeddings: List[List[float]],
        metadatas: List[Dict[str, Any]],
        ids: Optional[List[str]] = None
    ) -> None:
        """
        Add documents to vector database.

        Args:
            chunks: List of text chunks
            embeddings: List of embedding vectors
            metadatas: List of metadata dictionaries
            ids: Optional list of document IDs

        Raises:
            Exception: If adding documents fails
        """
        pass

    @abstractmethod
    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Search for similar documents.

        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return

        Returns:
            List of search results with metadata and scores

        Raises:
            Exception: If search fails
        """
        pass

    @abstractmethod
    def delete_document(self, doc_id: str) -> None:
        """
        Delete document from database.

        Args:
            doc_id: Document ID to delete

        Raises:
            Exception: If deletion fails
        """
        pass

    @abstractmethod
    def get_all_documents(self) -> List[Dict[str, Any]]:
        """
        Get all documents in the database.

        Returns:
            List of all documents with metadata
        """
        pass


class ChromaDBManager(BaseVectorDB):
    """ChromaDB implementation."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize ChromaDB."""
        super().__init__(config)
        try:
            import chromadb
            from chromadb.config import Settings

            self.client = chromadb.PersistentClient(
                path=self.persist_directory,
                settings=Settings(anonymized_telemetry=False)
            )

            # Get or create collection
            self.collection = self.client.get_or_create_collection(
                name="documents",
                metadata={"hnsw:space": "cosine"}
            )
        except ImportError:
            raise ImportError("chromadb package not installed. Run: pip install chromadb")

    def add_documents(
        self,
        chunks: List[str],
        embeddings: List[List[float]],
        metadatas: List[Dict[str, Any]],
        ids: Optional[List[str]] = None
    ) -> None:
        """Add documents to ChromaDB."""
        try:
            if ids is None:
                import uuid
                ids = [str(uuid.uuid4()) for _ in chunks]

            self.collection.add(
                documents=chunks,
                embeddings=embeddings,
                metadatas=metadatas,
                ids=ids
            )
        except Exception as e:
            raise Exception(f"ChromaDB add error: {str(e)}")

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """Search ChromaDB for similar documents."""
        try:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k
            )

            # Format results
            formatted_results = []
            if results['ids'] and len(results['ids']) > 0:
                for i in range(len(results['ids'][0])):
                    formatted_results.append({
                        'id': results['ids'][0][i],
                        'text': results['documents'][0][i],
                        'metadata': results['metadatas'][0][i],
                        'score': 1 - results['distances'][0][i]  # Convert distance to similarity
                    })

            return formatted_results
        except Exception as e:
            raise Exception(f"ChromaDB search error: {str(e)}")

    def delete_document(self, doc_id: str) -> None:
        """Delete document from ChromaDB."""
        try:
            # Get all documents
            all_docs = self.collection.get()

            # Find documents with matching filename
            ids_to_delete = []
            for i, metadata in enumerate(all_docs['metadatas']):
                if metadata.get('doc_id') == doc_id or metadata.get('filename') == doc_id:
                    ids_to_delete.append(all_docs['ids'][i])

            if ids_to_delete:
                self.collection.delete(ids=ids_to_delete)
        except Exception as e:
            raise Exception(f"ChromaDB delete error: {str(e)}")

    def get_all_documents(self) -> List[Dict[str, Any]]:
        """Get all documents from ChromaDB."""
        try:
            results = self.collection.get()

            # Extract unique documents by filename
            documents = {}
            if results['metadatas']:
                for metadata in results['metadatas']:
                    filename = metadata.get('filename', 'unknown')
                    if filename not in documents:
                        documents[filename] = metadata

            return list(documents.values())
        except Exception as e:
            raise Exception(f"ChromaDB get all error: {str(e)}")


class FAISSManager(BaseVectorDB):
    """FAISS implementation."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize FAISS."""
        super().__init__(config)
        try:
            import faiss
            import pickle

            self.faiss = faiss
            self.index = None
            self.documents = []
            self.metadatas = []
            self.ids = []

            self.index_path = os.path.join(self.persist_directory, 'faiss.index')
            self.data_path = os.path.join(self.persist_directory, 'faiss_data.pkl')

            # Load existing index if available
            self._load_index()
        except ImportError:
            raise ImportError("faiss-cpu package not installed. Run: pip install faiss-cpu")

    def _load_index(self) -> None:
        """Load existing FAISS index from disk."""
        import pickle

        if os.path.exists(self.index_path) and os.path.exists(self.data_path):
            try:
                self.index = self.faiss.read_index(self.index_path)
                with open(self.data_path, 'rb') as f:
                    data = pickle.load(f)
                    self.documents = data['documents']
                    self.metadatas = data['metadatas']
                    self.ids = data['ids']
            except Exception as e:
                print(f"Warning: Could not load FAISS index: {e}")
                self.index = None

    def _save_index(self) -> None:
        """Save FAISS index to disk."""
        import pickle

        if self.index is not None:
            self.faiss.write_index(self.index, self.index_path)
            with open(self.data_path, 'wb') as f:
                pickle.dump({
                    'documents': self.documents,
                    'metadatas': self.metadatas,
                    'ids': self.ids
                }, f)

    def add_documents(
        self,
        chunks: List[str],
        embeddings: List[List[float]],
        metadatas: List[Dict[str, Any]],
        ids: Optional[List[str]] = None
    ) -> None:
        """Add documents to FAISS."""
        try:
            import numpy as np
            import uuid

            if ids is None:
                ids = [str(uuid.uuid4()) for _ in chunks]

            embeddings_array = np.array(embeddings).astype('float32')

            # Create index if it doesn't exist
            if self.index is None:
                dimension = embeddings_array.shape[1]
                self.index = self.faiss.IndexFlatL2(dimension)

            # Add to index
            self.index.add(embeddings_array)

            # Store documents and metadata
            self.documents.extend(chunks)
            self.metadatas.extend(metadatas)
            self.ids.extend(ids)

            # Save to disk
            self._save_index()
        except Exception as e:
            raise Exception(f"FAISS add error: {str(e)}")

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """Search FAISS for similar documents."""
        try:
            import numpy as np

            if self.index is None or self.index.ntotal == 0:
                return []

            query_array = np.array([query_embedding]).astype('float32')
            distances, indices = self.index.search(query_array, min(top_k, self.index.ntotal))

            # Format results
            formatted_results = []
            for i, idx in enumerate(indices[0]):
                if idx < len(self.documents):
                    # Convert L2 distance to similarity score
                    similarity = 1 / (1 + distances[0][i])
                    formatted_results.append({
                        'id': self.ids[idx],
                        'text': self.documents[idx],
                        'metadata': self.metadatas[idx],
                        'score': float(similarity)
                    })

            return formatted_results
        except Exception as e:
            raise Exception(f"FAISS search error: {str(e)}")

    def delete_document(self, doc_id: str) -> None:
        """Delete document from FAISS (requires rebuilding index)."""
        try:
            # Find indices to keep
            indices_to_keep = []
            for i, metadata in enumerate(self.metadatas):
                if metadata.get('doc_id') != doc_id and metadata.get('filename') != doc_id:
                    indices_to_keep.append(i)

            if len(indices_to_keep) == len(self.documents):
                return  # Nothing to delete

            # Rebuild with remaining documents
            self.documents = [self.documents[i] for i in indices_to_keep]
            self.metadatas = [self.metadatas[i] for i in indices_to_keep]
            self.ids = [self.ids[i] for i in indices_to_keep]

            # Reset index
            self.index = None
            self._save_index()
        except Exception as e:
            raise Exception(f"FAISS delete error: {str(e)}")

    def get_all_documents(self) -> List[Dict[str, Any]]:
        """Get all documents from FAISS."""
        try:
            # Extract unique documents by filename
            documents = {}
            for metadata in self.metadatas:
                filename = metadata.get('filename', 'unknown')
                if filename not in documents:
                    documents[filename] = metadata

            return list(documents.values())
        except Exception as e:
            raise Exception(f"FAISS get all error: {str(e)}")


class VectorDBFactory:
    """Factory class for creating vector database instances."""

    @staticmethod
    def create_vector_db(config: Dict[str, Any]) -> BaseVectorDB:
        """
        Create vector database instance based on type.

        Args:
            config: Vector DB configuration dictionary

        Returns:
            BaseVectorDB instance

        Raises:
            ValueError: If type is not supported
        """
        db_type = config.get('type', '').lower()

        if db_type == 'chromadb':
            return ChromaDBManager(config)
        elif db_type == 'faiss':
            return FAISSManager(config)
        else:
            raise ValueError(f"Unsupported vector database type: {db_type}")
