"""
Unified LLM interface for DeepSeek and Alibaba Cloud Bailian (百炼).

Both providers expose an OpenAI-compatible API, so we use the ``openai`` client
and switch on base URL + model name.
"""

import logging
from typing import Optional

from openai import OpenAI, APIError, RateLimitError, APITimeoutError, APIConnectionError

from src.config import (
    DEEPSEEK_API_KEY,
    DEEPSEEK_BASE_URL,
    DEEPSEEK_MODEL,
    DEEPSEEK_THINKING,
    BAILIAN_API_KEY,
    BAILIAN_BASE_URL,
    BAILIAN_MODEL,
    LLM_PROVIDER,
    LLM_TEMPERATURE,
)

logger = logging.getLogger(__name__)


class LLM:
    """Unified LLM wrapper for DeepSeek and Bailian (百炼)."""

    def __init__(self, provider: str = LLM_PROVIDER, temperature: float = LLM_TEMPERATURE):
        self.provider = provider
        self.temperature = temperature
        self._clients: dict[str, OpenAI] = {}

    def _get_client(self, provider: str) -> OpenAI:
        """Get or create an OpenAI client for the given provider."""
        if provider not in self._clients:
            if provider == "deepseek":
                api_key = DEEPSEEK_API_KEY
                base_url = DEEPSEEK_BASE_URL
            elif provider == "bailian":
                api_key = BAILIAN_API_KEY
                base_url = BAILIAN_BASE_URL
            else:
                raise ValueError(f"Unknown LLM provider: {provider}")

            if not api_key:
                logger.warning(
                    "API key for '%s' is not set. "
                    "Set %s_API_KEY in .env or environment variables.",
                    provider, provider.upper(),
                )

            self._clients[provider] = OpenAI(api_key=api_key, base_url=base_url)
        return self._clients[provider]

    def query(
        self,
        prompt: str,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        thinking_mode: Optional[bool] = None,
        max_retries: int = 2,
    ) -> str:
        """
        Send a prompt to the LLM and return the text response.

        Args:
            prompt: The full prompt (including context + question).
            provider: Override provider (default: self.provider).
            model: Override model name.
            temperature: Override temperature.
            thinking_mode: Enable thinking/reasoning mode (DeepSeek only).
            max_retries: Max retries on transient errors.

        Returns:
            Generated text response.
        """
        provider = provider or self.provider

        if provider == "deepseek":
            model_name = model or DEEPSEEK_MODEL
        elif provider == "bailian":
            model_name = model or BAILIAN_MODEL
        else:
            raise ValueError(f"Unknown LLM provider: {provider}")

        client = self._get_client(provider)
        temp = temperature if temperature is not None else self.temperature
        thinking = thinking_mode if thinking_mode is not None else DEEPSEEK_THINKING

        last_error: Exception | None = None
        for attempt in range(1 + max_retries):
            try:
                kwargs = dict(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temp,
                )
                if thinking and provider == "deepseek":
                    kwargs["extra_body"] = {"thinking": {"type": "enabled"}}
                response = client.chat.completions.create(**kwargs)
                return response.choices[0].message.content

            except RateLimitError as e:
                logger.warning("Rate limited on %s (attempt %d/%d)", provider, attempt + 1, max_retries + 1)
                last_error = e
            except APITimeoutError as e:
                logger.warning("Timeout on %s (attempt %d/%d)", provider, attempt + 1, max_retries + 1)
                last_error = e
            except APIConnectionError as e:
                logger.warning("Connection error on %s (attempt %d/%d)", provider, attempt + 1, max_retries + 1)
                last_error = e
            except APIError as e:
                logger.error("API error on %s: %s", provider, e)
                raise  # Non-retryable API error

        return f"[Error] LLM query failed after {max_retries + 1} attempts: {last_error}"

    def auto_route_query(self, prompt: str, question: str, context: str) -> str:
        """
        Auto-route based on complexity:
        - Simple (short question, small context) → DeepSeek
        - Complex → Bailian (百炼)
        """
        if len(question) < 20 and len(context) < 2000:
            provider = "deepseek"
        else:
            provider = "bailian"
        logger.info("Auto-routing to %s (question=%d chars, context=%d chars)", provider, len(question), len(context))
        return self.query(prompt, provider=provider)
