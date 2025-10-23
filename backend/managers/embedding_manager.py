"""
Embedding Manager for RAG Application.

Provides abstract base class and concrete implementations for various embedding providers.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List
import os
import numpy as np


class BaseEmbedding(ABC):
    """Abstract base class for embedding providers."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize embedding provider.

        Args:
            config: Embedding configuration dictionary
        """
        self.config = config
        self.model = config.get('model', '')
        self.api_key = config.get('api_key', '')

    @abstractmethod
    def embed_text(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.

        Args:
            text: Text to embed

        Returns:
            Embedding vector as list of floats

        Raises:
            Exception: If embedding generation fails
        """
        pass

    @abstractmethod
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors

        Raises:
            Exception: If embedding generation fails
        """
        pass


class OpenAIEmbeddings(BaseEmbedding):
    """OpenAI Embeddings implementation."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize OpenAI embeddings."""
        super().__init__(config)
        try:
            from openai import OpenAI

            api_key = self.api_key or os.getenv('OPENAI_API_KEY')
            if not api_key:
                raise ValueError("OpenAI API key not provided")

            self.client = OpenAI(api_key=api_key)
        except ImportError:
            raise ImportError("openai package not installed. Run: pip install openai")

    def embed_text(self, text: str) -> List[float]:
        """Generate embedding using OpenAI API."""
        try:
            response = self.client.embeddings.create(
                model=self.model,
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            raise Exception(f"OpenAI embedding error: {str(e)}")

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        try:
            response = self.client.embeddings.create(
                model=self.model,
                input=texts
            )
            return [item.embedding for item in response.data]
        except Exception as e:
            raise Exception(f"OpenAI embedding error: {str(e)}")


class HuggingFaceEmbeddings(BaseEmbedding):
    """HuggingFace Embeddings implementation."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize HuggingFace embeddings."""
        super().__init__(config)
        try:
            from transformers import AutoTokenizer, AutoModel
            import torch

            self.tokenizer = AutoTokenizer.from_pretrained(self.model)
            self.model_obj = AutoModel.from_pretrained(self.model)
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.model_obj.to(self.device)
        except ImportError:
            raise ImportError("transformers and torch packages required. Run: pip install transformers torch")

    def _mean_pooling(self, model_output, attention_mask):
        """Apply mean pooling to model output."""
        import torch

        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def embed_text(self, text: str) -> List[float]:
        """Generate embedding using HuggingFace model."""
        import torch

        try:
            encoded_input = self.tokenizer([text], padding=True, truncation=True, return_tensors='pt')
            encoded_input = {k: v.to(self.device) for k, v in encoded_input.items()}

            with torch.no_grad():
                model_output = self.model_obj(**encoded_input)

            embeddings = self._mean_pooling(model_output, encoded_input['attention_mask'])
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

            return embeddings[0].cpu().numpy().tolist()
        except Exception as e:
            raise Exception(f"HuggingFace embedding error: {str(e)}")

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        import torch

        try:
            encoded_input = self.tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
            encoded_input = {k: v.to(self.device) for k, v in encoded_input.items()}

            with torch.no_grad():
                model_output = self.model_obj(**encoded_input)

            embeddings = self._mean_pooling(model_output, encoded_input['attention_mask'])
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

            return embeddings.cpu().numpy().tolist()
        except Exception as e:
            raise Exception(f"HuggingFace embedding error: {str(e)}")


class SentenceTransformerEmbeddings(BaseEmbedding):
    """Sentence Transformers Embeddings implementation."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize Sentence Transformers embeddings."""
        super().__init__(config)
        try:
            from sentence_transformers import SentenceTransformer

            self.model_obj = SentenceTransformer(self.model)
        except ImportError:
            raise ImportError("sentence-transformers package required. Run: pip install sentence-transformers")

    def embed_text(self, text: str) -> List[float]:
        """Generate embedding using Sentence Transformers."""
        try:
            embedding = self.model_obj.encode(text, convert_to_numpy=True)
            return embedding.tolist()
        except Exception as e:
            raise Exception(f"Sentence Transformer embedding error: {str(e)}")

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        try:
            embeddings = self.model_obj.encode(texts, convert_to_numpy=True)
            return embeddings.tolist()
        except Exception as e:
            raise Exception(f"Sentence Transformer embedding error: {str(e)}")


class EmbeddingFactory:
    """Factory class for creating embedding instances."""

    @staticmethod
    def create_embedding(config: Dict[str, Any]) -> BaseEmbedding:
        """
        Create embedding instance based on provider.

        Args:
            config: Embedding configuration dictionary

        Returns:
            BaseEmbedding instance

        Raises:
            ValueError: If provider is not supported
        """
        provider = config.get('provider', '').lower()

        if provider == 'openai':
            return OpenAIEmbeddings(config)
        elif provider == 'huggingface':
            return HuggingFaceEmbeddings(config)
        elif provider == 'sentence_transformers':
            return SentenceTransformerEmbeddings(config)
        else:
            raise ValueError(f"Unsupported embedding provider: {provider}")
