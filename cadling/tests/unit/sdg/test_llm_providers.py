"""Unit tests for LLM provider initialization and ChatAgent.

Tests all 5 supported providers:
- OpenAI
- Anthropic
- vLLM (OpenAI-compatible)
- Ollama (OpenAI-compatible)
- OpenAI-compatible (generic)
"""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch

from cadling.sdg.qa.base import LlmOptions, LlmProvider


class TestInitializeLlm:
    """Tests for initialize_llm function."""

    @patch("cadling.sdg.qa.utils._OPENAI_AVAILABLE", True)
    @patch("cadling.sdg.qa.utils.openai")
    def test_openai_provider(self, mock_openai):
        """Test OpenAI provider initialization."""
        from cadling.sdg.qa.utils import initialize_llm

        mock_client = MagicMock()
        mock_openai.OpenAI.return_value = mock_client

        options = LlmOptions(
            provider=LlmProvider.OPENAI,
            model_id="gpt-4",
            api_key="sk-test-key",
        )

        client = initialize_llm(options)

        mock_openai.OpenAI.assert_called_once_with(api_key="sk-test-key")
        assert client == mock_client

    @patch("cadling.sdg.qa.utils._ANTHROPIC_AVAILABLE", True)
    @patch("cadling.sdg.qa.utils.anthropic")
    def test_anthropic_provider(self, mock_anthropic):
        """Test Anthropic provider initialization."""
        from cadling.sdg.qa.utils import initialize_llm

        mock_client = MagicMock()
        mock_anthropic.Anthropic.return_value = mock_client

        options = LlmOptions(
            provider=LlmProvider.ANTHROPIC,
            model_id="claude-3-opus",
            api_key="sk-ant-test-key",
        )

        client = initialize_llm(options)

        mock_anthropic.Anthropic.assert_called_once_with(api_key="sk-ant-test-key")
        assert client == mock_client

    @patch("cadling.sdg.qa.utils._OPENAI_AVAILABLE", True)
    @patch("cadling.sdg.qa.utils.openai")
    def test_vllm_provider(self, mock_openai):
        """Test vLLM provider initialization with default URL."""
        from cadling.sdg.qa.utils import initialize_llm

        mock_client = MagicMock()
        mock_openai.OpenAI.return_value = mock_client

        options = LlmOptions(
            provider=LlmProvider.VLLM,
            model_id="meta-llama/Llama-2-70b",
        )

        client = initialize_llm(options)

        mock_openai.OpenAI.assert_called_once_with(
            base_url="http://localhost:8000/v1",
            api_key="EMPTY",
        )
        assert client == mock_client

    @patch("cadling.sdg.qa.utils._OPENAI_AVAILABLE", True)
    @patch("cadling.sdg.qa.utils.openai")
    def test_vllm_provider_custom_url(self, mock_openai):
        """Test vLLM provider with custom URL."""
        from cadling.sdg.qa.utils import initialize_llm

        mock_client = MagicMock()
        mock_openai.OpenAI.return_value = mock_client

        options = LlmOptions(
            provider=LlmProvider.VLLM,
            model_id="custom-model",
            url="http://gpu-server:8080/v1",
            api_key="custom-key",
        )

        client = initialize_llm(options)

        mock_openai.OpenAI.assert_called_once_with(
            base_url="http://gpu-server:8080/v1",
            api_key="custom-key",
        )
        assert client == mock_client

    @patch("cadling.sdg.qa.utils._OPENAI_AVAILABLE", True)
    @patch("cadling.sdg.qa.utils.openai")
    def test_ollama_provider(self, mock_openai):
        """Test Ollama provider initialization with default URL."""
        from cadling.sdg.qa.utils import initialize_llm

        mock_client = MagicMock()
        mock_openai.OpenAI.return_value = mock_client

        options = LlmOptions(
            provider=LlmProvider.OLLAMA,
            model_id="llama2",
        )

        client = initialize_llm(options)

        mock_openai.OpenAI.assert_called_once_with(
            base_url="http://localhost:11434/v1",
            api_key="ollama",
        )
        assert client == mock_client

    @patch("cadling.sdg.qa.utils._OPENAI_AVAILABLE", True)
    @patch("cadling.sdg.qa.utils.openai")
    def test_openai_compatible_provider(self, mock_openai):
        """Test OpenAI-compatible provider requires URL."""
        from cadling.sdg.qa.utils import initialize_llm

        mock_client = MagicMock()
        mock_openai.OpenAI.return_value = mock_client

        options = LlmOptions(
            provider=LlmProvider.OPENAI_COMPATIBLE,
            model_id="custom-model",
            url="http://custom-server/v1",
            api_key="custom-api-key",
        )

        client = initialize_llm(options)

        mock_openai.OpenAI.assert_called_once_with(
            base_url="http://custom-server/v1",
            api_key="custom-api-key",
        )
        assert client == mock_client

    @patch("cadling.sdg.qa.utils._OPENAI_AVAILABLE", True)
    def test_openai_compatible_requires_url(self):
        """Test OpenAI-compatible provider raises without URL."""
        from cadling.sdg.qa.utils import initialize_llm

        options = LlmOptions(
            provider=LlmProvider.OPENAI_COMPATIBLE,
            model_id="custom-model",
        )

        with pytest.raises(ValueError, match="url is required"):
            initialize_llm(options)

    @patch("cadling.sdg.qa.utils._OPENAI_AVAILABLE", False)
    def test_openai_not_available(self):
        """Test error when openai package not installed."""
        from cadling.sdg.qa.utils import initialize_llm

        options = LlmOptions(
            provider=LlmProvider.OPENAI,
            model_id="gpt-4",
        )

        with pytest.raises(ImportError, match="openai package required"):
            initialize_llm(options)

    @patch("cadling.sdg.qa.utils._ANTHROPIC_AVAILABLE", False)
    def test_anthropic_not_available(self):
        """Test error when anthropic package not installed."""
        from cadling.sdg.qa.utils import initialize_llm

        options = LlmOptions(
            provider=LlmProvider.ANTHROPIC,
            model_id="claude-3-opus",
        )

        with pytest.raises(ImportError, match="anthropic package required"):
            initialize_llm(options)


class TestChatAgent:
    """Tests for ChatAgent class."""

    @patch("cadling.sdg.qa.utils._OPENAI_AVAILABLE", True)
    @patch("cadling.sdg.qa.utils.openai")
    def test_from_options(self, mock_openai):
        """Test ChatAgent.from_options factory."""
        from cadling.sdg.qa.utils import ChatAgent

        mock_client = MagicMock()
        mock_openai.OpenAI.return_value = mock_client

        options = LlmOptions(
            provider=LlmProvider.OPENAI,
            model_id="gpt-4",
            temperature=0.5,
            max_tokens=1024,
        )

        agent = ChatAgent.from_options(options)

        assert agent.client == mock_client
        assert agent.model_id == "gpt-4"
        assert agent.provider == LlmProvider.OPENAI
        assert agent.temperature == 0.5
        assert agent.max_tokens == 1024

    def test_ask_openai_compatible(self):
        """Test ask method with OpenAI-compatible provider."""
        from cadling.sdg.qa.utils import ChatAgent

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Hello, world!"
        mock_client.chat.completions.create.return_value = mock_response

        agent = ChatAgent(
            client=mock_client,
            model_id="gpt-4",
            provider=LlmProvider.OPENAI,
            temperature=0.7,
            max_tokens=512,
        )

        result = agent.ask("Test prompt")

        mock_client.chat.completions.create.assert_called_once_with(
            model="gpt-4",
            messages=[{"role": "user", "content": "Test prompt"}],
            temperature=0.7,
            max_tokens=512,
        )
        assert result == "Hello, world!"

    def test_ask_anthropic(self):
        """Test ask method with Anthropic provider."""
        from cadling.sdg.qa.utils import ChatAgent

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.content = [MagicMock()]
        mock_response.content[0].text = "Anthropic response"
        mock_client.messages.create.return_value = mock_response

        agent = ChatAgent(
            client=mock_client,
            model_id="claude-3-opus",
            provider=LlmProvider.ANTHROPIC,
            temperature=0.5,
            max_tokens=1024,
        )

        result = agent.ask("Test prompt")

        mock_client.messages.create.assert_called_once_with(
            model="claude-3-opus",
            max_tokens=1024,
            temperature=0.5,
            messages=[{"role": "user", "content": "Test prompt"}],
        )
        assert result == "Anthropic response"

    def test_ask_max_tokens_override(self):
        """Test ask method with max_tokens override."""
        from cadling.sdg.qa.utils import ChatAgent

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Response"
        mock_client.chat.completions.create.return_value = mock_response

        agent = ChatAgent(
            client=mock_client,
            model_id="gpt-4",
            provider=LlmProvider.OPENAI,
            temperature=0.7,
            max_tokens=512,
        )

        agent.ask("Test prompt", max_tokens=2048)

        mock_client.chat.completions.create.assert_called_once_with(
            model="gpt-4",
            messages=[{"role": "user", "content": "Test prompt"}],
            temperature=0.7,
            max_tokens=2048,  # Override value
        )

    def test_ask_vllm_uses_openai_api(self):
        """Test vLLM provider uses OpenAI-compatible API."""
        from cadling.sdg.qa.utils import ChatAgent

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "vLLM response"
        mock_client.chat.completions.create.return_value = mock_response

        agent = ChatAgent(
            client=mock_client,
            model_id="llama-70b",
            provider=LlmProvider.VLLM,
            temperature=0.8,
            max_tokens=1024,
        )

        result = agent.ask("Test prompt")

        # vLLM should use the same OpenAI-compatible API
        mock_client.chat.completions.create.assert_called_once()
        assert result == "vLLM response"

    def test_ask_ollama_uses_openai_api(self):
        """Test Ollama provider uses OpenAI-compatible API."""
        from cadling.sdg.qa.utils import ChatAgent

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Ollama response"
        mock_client.chat.completions.create.return_value = mock_response

        agent = ChatAgent(
            client=mock_client,
            model_id="llama2",
            provider=LlmProvider.OLLAMA,
            temperature=0.7,
            max_tokens=512,
        )

        result = agent.ask("Test prompt")

        # Ollama should use the same OpenAI-compatible API
        mock_client.chat.completions.create.assert_called_once()
        assert result == "Ollama response"
