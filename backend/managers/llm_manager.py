"""
LLM Manager for RAG Application.

Provides abstract base class and concrete implementations for various LLM providers.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
import os


class BaseLLM(ABC):
    """Abstract base class for LLM providers."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize LLM provider.

        Args:
            config: LLM configuration dictionary
        """
        self.config = config
        self.model = config.get('model', '')
        self.api_key = config.get('api_key', '')

    @abstractmethod
    def generate_response(self, prompt: str, context: str, system_prompt: str = "") -> str:
        """
        Generate response from LLM.

        Args:
            prompt: User query/prompt
            context: Retrieved context from documents
            system_prompt: System prompt for the LLM

        Returns:
            Generated response string

        Raises:
            Exception: If LLM API call fails
        """
        pass


class OpenAILLM(BaseLLM):
    """OpenAI LLM implementation."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize OpenAI LLM."""
        super().__init__(config)
        try:
            from openai import OpenAI

            # Use environment variable if api_key is empty in config
            api_key = self.api_key or os.getenv('OPENAI_API_KEY')
            if not api_key:
                raise ValueError("OpenAI API key not provided in config or environment")

            self.client = OpenAI(api_key=api_key)
        except ImportError:
            raise ImportError("openai package not installed. Run: pip install openai")

    def generate_response(self, prompt: str, context: str, system_prompt: str = "") -> str:
        """Generate response using OpenAI API."""
        try:
            messages = []

            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})

            # Combine context and prompt
            user_message = f"Context:\n{context}\n\nQuestion: {prompt}"
            messages.append({"role": "user", "content": user_message})

            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.7,
                max_tokens=1000
            )

            return response.choices[0].message.content
        except Exception as e:
            raise Exception(f"OpenAI API error: {str(e)}")


class AnthropicLLM(BaseLLM):
    """Anthropic (Claude) LLM implementation."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize Anthropic LLM."""
        super().__init__(config)
        try:
            from anthropic import Anthropic

            api_key = self.api_key or os.getenv('ANTHROPIC_API_KEY')
            if not api_key:
                raise ValueError("Anthropic API key not provided in config or environment")

            self.client = Anthropic(api_key=api_key)
        except ImportError:
            raise ImportError("anthropic package not installed. Run: pip install anthropic")

    def generate_response(self, prompt: str, context: str, system_prompt: str = "") -> str:
        """Generate response using Anthropic API."""
        try:
            # Combine context and prompt
            user_message = f"Context:\n{context}\n\nQuestion: {prompt}"

            response = self.client.messages.create(
                model=self.model,
                max_tokens=1000,
                system=system_prompt if system_prompt else "You are a helpful assistant.",
                messages=[
                    {"role": "user", "content": user_message}
                ]
            )

            return response.content[0].text
        except Exception as e:
            raise Exception(f"Anthropic API error: {str(e)}")


class AzureOpenAILLM(BaseLLM):
    """Azure OpenAI LLM implementation."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize Azure OpenAI LLM."""
        super().__init__(config)
        try:
            from openai import AzureOpenAI

            api_key = self.api_key or os.getenv('AZURE_OPENAI_API_KEY')
            base_url = config.get('base_url', '') or os.getenv('AZURE_OPENAI_ENDPOINT')

            if not api_key or not base_url:
                raise ValueError("Azure OpenAI API key and endpoint required")

            self.client = AzureOpenAI(
                api_key=api_key,
                api_version="2024-02-15-preview",
                azure_endpoint=base_url
            )
        except ImportError:
            raise ImportError("openai package not installed. Run: pip install openai")

    def generate_response(self, prompt: str, context: str, system_prompt: str = "") -> str:
        """Generate response using Azure OpenAI API."""
        try:
            messages = []

            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})

            user_message = f"Context:\n{context}\n\nQuestion: {prompt}"
            messages.append({"role": "user", "content": user_message})

            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.7,
                max_tokens=1000
            )

            return response.choices[0].message.content
        except Exception as e:
            raise Exception(f"Azure OpenAI API error: {str(e)}")


class OllamaLLM(BaseLLM):
    """Ollama (local) LLM implementation."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize Ollama LLM."""
        super().__init__(config)
        self.base_url = config.get('base_url', 'http://localhost:11434')

    def generate_response(self, prompt: str, context: str, system_prompt: str = "") -> str:
        """Generate response using Ollama API."""
        try:
            import requests

            user_message = f"Context:\n{context}\n\nQuestion: {prompt}"

            payload = {
                "model": self.model,
                "prompt": user_message,
                "system": system_prompt if system_prompt else "You are a helpful assistant.",
                "stream": False
            }

            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=60
            )
            response.raise_for_status()

            return response.json().get('response', '')
        except Exception as e:
            raise Exception(f"Ollama API error: {str(e)}")


class LLMFactory:
    """Factory class for creating LLM instances."""

    @staticmethod
    def create_llm(config: Dict[str, Any]) -> BaseLLM:
        """
        Create LLM instance based on provider.

        Args:
            config: LLM configuration dictionary

        Returns:
            BaseLLM instance

        Raises:
            ValueError: If provider is not supported
        """
        provider = config.get('provider', '').lower()

        if provider == 'openai':
            return OpenAILLM(config)
        elif provider == 'anthropic':
            return AnthropicLLM(config)
        elif provider == 'azure_openai':
            return AzureOpenAILLM(config)
        elif provider == 'ollama':
            return OllamaLLM(config)
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")
