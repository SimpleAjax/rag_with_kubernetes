# app/core/llm_client.py

from abc import ABC, abstractmethod
from typing import List, Dict, Any, AsyncGenerator, Optional
import httpx # For making HTTP requests to Ollama
import json # For handling Ollama's streaming response

from app.core.config import settings
from app.api.v1.schemas.chat_schemas import ChatMessage

import logging

logger = logging.getLogger(__name__)

class BaseLLMClient(ABC):
    @abstractmethod
    async def generate_response(
        self, prompt: str, history: Optional[List[ChatMessage]] = None,
        temperature: float = 0.7, max_tokens: int = 1024, **kwargs: Any
    ) -> str:
        pass

    @abstractmethod
    async def generate_streaming_response(
        self, prompt: str, history: Optional[List[ChatMessage]] = None,
        temperature: float = 0.7, max_tokens: int = 1024, **kwargs: Any
    ) -> AsyncGenerator[str, None]:
        if False: yield "" # pragma: no cover
        pass

class MockLLMClient(BaseLLMClient):
    def __init__(self, model_name: str = "mock-llm", api_key: Optional[str] = None):
        self.model_name = model_name
        logger.info(f"MockLLMClient initialized with model: {self.model_name}")

    async def generate_response(
        self, prompt: str, history: Optional[List[ChatMessage]] = None,
        temperature: float = 0.7, max_tokens: int = 150, **kwargs: Any
    ) -> str:
        logger.info(f"MockLLMClient generating non-streaming response for prompt: '{prompt[:50]}...'")
        import asyncio
        await asyncio.sleep(0.1)
        response_text = f"Mock response from {self.model_name} to prompt: '{prompt}'. "
        if history: response_text += f"Considering {len(history)} messages. "
        response_text += f"Params: temp={temperature}, max_tokens={max_tokens}."
        return response_text[:max_tokens*5]

    async def generate_streaming_response(
        self, prompt: str, history: Optional[List[ChatMessage]] = None,
        temperature: float = 0.7, max_tokens: int = 150, **kwargs: Any
    ) -> AsyncGenerator[str, None]:
        logger.info(f"MockLLMClient generating streaming response for prompt: '{prompt[:50]}...'")
        base_response = f"Streaming mock response from {self.model_name} to: '{prompt}'. "
        if history: base_response += f"History considered. "
        words = base_response.split()
        import asyncio
        for i, word in enumerate(words):
            if i * 5 > max_tokens * 5 : break
            yield word + " "
            await asyncio.sleep(0.02)
        yield "\n[End of mock stream]"

class OpenAILLMClient(BaseLLMClient):
    def __init__(self, api_key: str, model_name: str = "gpt-3.5-turbo"):
        try:
            from openai import AsyncOpenAI
            self.client = AsyncOpenAI(api_key=api_key)
            self.model_name = model_name
            logger.info(f"OpenAILLMClient initialized with model: {self.model_name}")
        except ImportError:
            logger.error("OpenAI package not installed. Please install with 'poetry add openai --optional'")
            raise ImportError("OpenAI package not found. Please install it to use OpenAILLMClient.")
        except Exception as e:
            logger.error(f"Error initializing OpenAI client: {e}", exc_info=True)
            raise

    async def generate_response(
        self, prompt: str, history: Optional[List[ChatMessage]] = None,
        temperature: float = 0.7, max_tokens: int = 1024,
        system_message: Optional[str] = None, **kwargs: Any
    ) -> str:
        messages_for_api: List[Dict[str, str]] = []
        if system_message: messages_for_api.append({"role": "system", "content": system_message})
        if history:
            for msg in history: messages_for_api.append({"role": msg.role, "content": msg.content})
        messages_for_api.append({"role": "user", "content": prompt})
        try:
            completion = await self.client.chat.completions.create(
                model=self.model_name, messages=messages_for_api, # type: ignore
                temperature=temperature, max_tokens=max_tokens, **kwargs
            ) # type: ignore
            response_content = completion.choices[0].message.content
            return response_content.strip() if response_content else ""
        except Exception as e:
            logger.error(f"Error calling OpenAI API (generate_response): {e}", exc_info=True)
            raise Exception(f"OpenAI API call failed: {str(e)}") from e

    async def generate_streaming_response(
        self, prompt: str, history: Optional[List[ChatMessage]] = None,
        temperature: float = 0.7, max_tokens: int = 1024,
        system_message: Optional[str] = None, **kwargs: Any
    ) -> AsyncGenerator[str, None]:
        messages_for_api: List[Dict[str, str]] = []
        if system_message: messages_for_api.append({"role": "system", "content": system_message})
        if history:
            for msg in history: messages_for_api.append({"role": msg.role, "content": msg.content})
        messages_for_api.append({"role": "user", "content": prompt})
        try:
            stream = await self.client.chat.completions.create(
                model=self.model_name, messages=messages_for_api, # type: ignore
                temperature=temperature, max_tokens=max_tokens, stream=True, **kwargs
            ) # type: ignore
            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except Exception as e:
            logger.error(f"Error streaming from OpenAI API: {e}", exc_info=True)
            yield f"[LLM_STREAM_ERROR: {str(e)}]"


class OllamaLLMClient(BaseLLMClient):
    """
    LLM Client for interacting with a local Ollama API.
    """
    def __init__(self, base_url: str, model_name: str, request_timeout: int = 120):
        self.base_url = base_url.rstrip('/')
        self.model_name = model_name
        self.request_timeout = request_timeout
        # Create an async HTTP client instance
        self.http_client = httpx.AsyncClient(timeout=self.request_timeout)
        logger.info(f"OllamaLLMClient initialized. Model: {self.model_name}, Base URL: {self.base_url}")

    def _prepare_messages_for_ollama(
        self, prompt: str, history: Optional[List[ChatMessage]] = None, system_message: Optional[str] = None
    ) -> List[Dict[str, str]]:
        messages: List[Dict[str, str]] = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        if history:
            for msg in history:
                messages.append({"role": msg.role, "content": msg.content})
        messages.append({"role": "user", "content": prompt})
        return messages

    async def generate_response(
        self, prompt: str, history: Optional[List[ChatMessage]] = None,
        temperature: float = 0.7, max_tokens: Optional[int] = None, # Ollama uses 'num_predict' for max_tokens
        system_message: Optional[str] = None, **kwargs: Any
    ) -> str:
        messages = self._prepare_messages_for_ollama(prompt, history, system_message)
        payload = {
            "model": self.model_name,
            "messages": messages,
            "stream": False, # For non-streaming
            "options": {
                "temperature": temperature,
            }
        }
        if max_tokens is not None: # Ollama's equivalent is num_predict
            payload["options"]["num_predict"] = max_tokens
        
        # Add any other specific Ollama options from kwargs
        payload["options"].update(kwargs.get("options", {}))


        api_url = f"{self.base_url}/api/chat"
        logger.debug(f"Sending request to Ollama: {api_url}, Payload: {payload}")
        try:
            response = await self.http_client.post(api_url, json=payload)
            response.raise_for_status() # Raise an exception for HTTP errors (4xx or 5xx)
            
            response_data = response.json()
            if response_data.get("message") and response_data["message"].get("content"):
                return response_data["message"]["content"].strip()
            else:
                logger.error(f"Ollama response format unexpected: {response_data}")
                return "[Error: Unexpected Ollama response format]"
        except httpx.HTTPStatusError as e:
            logger.error(f"Ollama API HTTP error (generate_response): {e.response.status_code} - {e.response.text}", exc_info=True)
            raise Exception(f"Ollama API request failed with status {e.response.status_code}: {e.response.text}") from e
        except Exception as e:
            logger.error(f"Error calling Ollama API (generate_response): {e}", exc_info=True)
            raise Exception(f"Ollama API call failed: {str(e)}") from e

    async def generate_streaming_response(
        self, prompt: str, history: Optional[List[ChatMessage]] = None,
        temperature: float = 0.7, max_tokens: Optional[int] = None,
        system_message: Optional[str] = None, **kwargs: Any
    ) -> AsyncGenerator[str, None]:
        messages = self._prepare_messages_for_ollama(prompt, history, system_message)
        payload = {
            "model": self.model_name,
            "messages": messages,
            "stream": True, # Enable streaming
            "options": {
                "temperature": temperature,
            }
        }
        if max_tokens is not None:
            payload["options"]["num_predict"] = max_tokens
        
        payload["options"].update(kwargs.get("options", {}))

        api_url = f"{self.base_url}/api/chat"
        logger.debug(f"Sending streaming request to Ollama: {api_url}, Payload: {payload}")
        try:
            async with self.http_client.stream("POST", api_url, json=payload) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if line:
                        try:
                            chunk_data = json.loads(line)
                            if chunk_data.get("message") and chunk_data["message"].get("content"):
                                yield chunk_data["message"]["content"]
                            if chunk_data.get("done") and chunk_data.get("error"): # Check for error in stream
                                logger.error(f"Ollama stream error: {chunk_data.get('error')}")
                                yield f"[LLM_STREAM_ERROR: {chunk_data.get('error')}]"
                                break
                            if chunk_data.get("done"):
                                break
                        except json.JSONDecodeError:
                            logger.warning(f"Ollama stream: Could not decode JSON line: {line}")
        except httpx.HTTPStatusError as e:
            logger.error(f"Ollama API HTTP error (streaming): {e.response.status_code} - {e.response.text}", exc_info=True)
            yield f"[LLM_STREAM_ERROR: HTTP {e.response.status_code} - {e.response.text}]"
        except Exception as e:
            logger.error(f"Error streaming from Ollama API: {e}", exc_info=True)
            yield f"[LLM_STREAM_ERROR: {str(e)}]"


def get_llm_client() -> BaseLLMClient:
    provider = settings.LLM_PROVIDER.lower() if settings.LLM_PROVIDER else "mock"
    
    if provider == "openai":
        if not settings.OPENAI_API_KEY:
            logger.error("OPENAI_API_KEY not configured. Falling back to MockLLMClient.")
            return MockLLMClient(model_name=settings.OPENAI_MODEL_NAME or "mock-openai-fallback")
        try:
            return OpenAILLMClient(api_key=settings.OPENAI_API_KEY, model_name=settings.OPENAI_MODEL_NAME)
        except Exception as e:
            logger.error(f"Failed to initialize OpenAILLMClient: {e}. Falling back to MockLLMClient.", exc_info=True)
            return MockLLMClient(model_name=settings.OPENAI_MODEL_NAME or "mock-init-fail-fallback")
    
    elif provider == "ollama":
        logger.info(f"Attempting to initialize OllamaLLMClient with base_url='{settings.OLLAMA_BASE_URL}' and model='{settings.OLLAMA_MODEL_NAME}'")
        try:
            return OllamaLLMClient(
                base_url=settings.OLLAMA_BASE_URL,
                model_name=settings.OLLAMA_MODEL_NAME,
                request_timeout=settings.OLLAMA_REQUEST_TIMEOUT
            )
        except Exception as e:
            logger.error(f"Failed to initialize OllamaLLMClient: {e}. Falling back to MockLLMClient.", exc_info=True)
            return MockLLMClient(model_name=settings.OLLAMA_MODEL_NAME or "mock-ollama-fallback")

    elif provider == "mock":
        logger.info(f"Using MockLLMClient for LLM provider: {provider}")
        return MockLLMClient(model_name=settings.OLLAMA_MODEL_NAME or "mock-default-model")
    else:
        logger.warning(f"Unsupported LLM provider '{provider}'. Falling back to MockLLMClient.")
        return MockLLMClient(model_name="fallback-mock-model")

