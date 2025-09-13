"""
Tests for LangChain API Integration

This module contains comprehensive tests for the LangChain API endpoints,
including authentication, validation, error handling, and rate limiting.
"""

import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient
from fastapi import status

from langflow.api.v1.langchain_router import (
    LangChainRequest,
    LangChainResponse,
    LangChainAPIException,
    validate_langchain_request,
    process_langchain_request
)
from langflow.services.database.models.user.model import UserRead


class TestLangChainRequest:
    """Test LangChain request validation."""
    
    def test_valid_request(self):
        """Test valid request creation."""
        request = LangChainRequest(
            messages=[
                {"role": "user", "content": "Hello"}
            ],
            model="gpt-3.5-turbo",
            temperature=0.7
        )
        assert request.model == "gpt-3.5-turbo"
        assert request.temperature == 0.7
        assert len(request.messages) == 1
    
    def test_invalid_messages(self):
        """Test invalid message validation."""
        with pytest.raises(ValueError):
            LangChainRequest(
                messages=[
                    {"role": "invalid", "content": "Hello"}
                ]
            )
    
    def test_missing_message_fields(self):
        """Test missing message fields validation."""
        with pytest.raises(ValueError):
            LangChainRequest(
                messages=[
                    {"content": "Hello"}  # Missing role
                ]
            )
    
    def test_temperature_validation(self):
        """Test temperature range validation."""
        with pytest.raises(ValueError):
            LangChainRequest(
                messages=[{"role": "user", "content": "Hello"}],
                temperature=3.0  # Too high
            )
    
    def test_max_tokens_validation(self):
        """Test max_tokens validation."""
        with pytest.raises(ValueError):
            LangChainRequest(
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=5000  # Too high
            )


class TestLangChainResponse:
    """Test LangChain response model."""
    
    def test_response_creation(self):
        """Test response model creation."""
        response = LangChainResponse(
            content="Hello, world!",
            model="gpt-3.5-turbo",
            usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
        )
        assert response.content == "Hello, world!"
        assert response.model == "gpt-3.5-turbo"
        assert response.usage["total_tokens"] == 15


class TestRequestValidation:
    """Test request validation functions."""
    
    def test_validate_langchain_request_valid(self):
        """Test validation of valid request."""
        request = LangChainRequest(
            messages=[{"role": "user", "content": "Hello"}],
            model="gpt-3.5-turbo"
        )
        # Should not raise any exception
        validate_langchain_request(request)
    
    def test_validate_langchain_request_too_many_messages(self):
        """Test validation with too many messages."""
        request = LangChainRequest(
            messages=[{"role": "user", "content": "Hello"}] * 101,  # Too many
            model="gpt-3.5-turbo"
        )
        with pytest.raises(LangChainAPIException) as exc_info:
            validate_langchain_request(request)
        assert exc_info.value.error_code == "TOO_MANY_MESSAGES"
    
    def test_validate_langchain_request_message_too_long(self):
        """Test validation with message too long."""
        request = LangChainRequest(
            messages=[{"role": "user", "content": "x" * 10001}],  # Too long
            model="gpt-3.5-turbo"
        )
        with pytest.raises(LangChainAPIException) as exc_info:
            validate_langchain_request(request)
        assert exc_info.value.error_code == "MESSAGE_TOO_LONG"
    
    def test_validate_langchain_request_invalid_model(self):
        """Test validation with invalid model."""
        request = LangChainRequest(
            messages=[{"role": "user", "content": "Hello"}],
            model="invalid-model"
        )
        with pytest.raises(LangChainAPIException) as exc_info:
            validate_langchain_request(request)
        assert exc_info.value.error_code == "INVALID_MODEL"


class TestLangChainAPIException:
    """Test custom exception handling."""
    
    def test_exception_creation(self):
        """Test exception creation with all parameters."""
        exc = LangChainAPIException(
            message="Test error",
            error_code="TEST_ERROR",
            status_code=400,
            details={"key": "value"}
        )
        assert exc.error_code == "TEST_ERROR"
        assert exc.status_code == 400
        assert exc.details == {"key": "value"}
    
    def test_exception_defaults(self):
        """Test exception creation with default parameters."""
        exc = LangChainAPIException("Test error")
        assert exc.error_code == "LANGCHAIN_ERROR"
        assert exc.status_code == 500
        assert exc.details == {}


class TestLangChainAPIEndpoints:
    """Test LangChain API endpoints."""
    
    @pytest.fixture
    def mock_user(self):
        """Create a mock user for testing."""
        return UserRead(
            id="test-user-id",
            username="testuser",
            email="test@example.com",
            is_active=True,
            is_superuser=False
        )
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        from langflow.main import create_app
        app = create_app()
        return TestClient(app)
    
    @patch('langflow.api.v1.langchain_router.api_key_security')
    @patch('langflow.api.v1.langchain_router.process_langchain_request')
    def test_chat_endpoint_success(self, mock_process, mock_auth, client, mock_user):
        """Test successful chat endpoint call."""
        # Setup mocks
        mock_auth.return_value = mock_user
        mock_response = LangChainResponse(
            content="Test response",
            model="gpt-3.5-turbo"
        )
        mock_process.return_value = mock_response
        
        # Make request
        response = client.post(
            "/api/v1/langchain/chat",
            headers={"x-api-key": "test-key"},
            json={
                "messages": [{"role": "user", "content": "Hello"}],
                "model": "gpt-3.5-turbo"
            }
        )
        
        # Assertions
        assert response.status_code == 200
        data = response.json()
        assert data["content"] == "Test response"
        assert data["model"] == "gpt-3.5-turbo"
    
    @patch('langflow.api.v1.langchain_router.api_key_security')
    def test_chat_endpoint_invalid_request(self, mock_auth, client, mock_user):
        """Test chat endpoint with invalid request."""
        mock_auth.return_value = mock_user
        
        response = client.post(
            "/api/v1/langchain/chat",
            headers={"x-api-key": "test-key"},
            json={
                "messages": [{"role": "invalid", "content": "Hello"}],
                "model": "gpt-3.5-turbo"
            }
        )
        
        assert response.status_code == 422  # Validation error
    
    @patch('langflow.api.v1.langchain_router.api_key_security')
    def test_chat_endpoint_missing_auth(self, mock_auth, client):
        """Test chat endpoint without authentication."""
        mock_auth.side_effect = Exception("No API key")
        
        response = client.post(
            "/api/v1/langchain/chat",
            json={
                "messages": [{"role": "user", "content": "Hello"}],
                "model": "gpt-3.5-turbo"
            }
        )
        
        assert response.status_code == 403
    
    @patch('langflow.api.v1.langchain_router.api_key_security')
    def test_models_endpoint(self, mock_auth, client, mock_user):
        """Test models endpoint."""
        mock_auth.return_value = mock_user
        
        response = client.get(
            "/api/v1/langchain/models",
            headers={"x-api-key": "test-key"}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert "gpt-3.5-turbo" in data
        assert "gpt-4" in data
    
    @patch('langflow.api.v1.langchain_router.api_key_security')
    def test_health_endpoint(self, mock_auth, client, mock_user):
        """Test health endpoint."""
        mock_auth.return_value = mock_user
        
        response = client.get(
            "/api/v1/langchain/health",
            headers={"x-api-key": "test-key"}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["service"] == "langchain_integration"


class TestRateLimiting:
    """Test rate limiting functionality."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        from langflow.main import create_app
        app = create_app()
        return TestClient(app)
    
    @patch('langflow.api.v1.langchain_router.api_key_security')
    def test_rate_limiting_chat_endpoint(self, mock_auth, client, mock_user):
        """Test rate limiting on chat endpoint."""
        mock_auth.return_value = mock_user
        
        # Make multiple requests quickly
        for i in range(35):  # Exceed the 30/minute limit
            response = client.post(
                "/api/v1/langchain/chat",
                headers={"x-api-key": "test-key"},
                json={
                    "messages": [{"role": "user", "content": f"Request {i}"}],
                    "model": "gpt-3.5-turbo"
                }
            )
            
            if i >= 30:
                # Should be rate limited
                assert response.status_code == 429
            else:
                # Should succeed (mocked)
                assert response.status_code in [200, 500]  # 500 due to mocking


class TestStreamingEndpoint:
    """Test streaming functionality."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        from langflow.main import create_app
        app = create_app()
        return TestClient(app)
    
    @patch('langflow.api.v1.langchain_router.api_key_security')
    @patch('langflow.api.v1.langchain_router.process_langchain_stream')
    def test_streaming_endpoint(self, mock_stream, mock_auth, client, mock_user):
        """Test streaming endpoint."""
        mock_auth.return_value = mock_user
        
        # Mock streaming response
        async def mock_stream_generator():
            yield {"content": "Hello", "is_final": False}
            yield {"content": " world", "is_final": False}
            yield {"content": "", "is_final": True}
        
        mock_stream.return_value = mock_stream_generator()
        
        response = client.post(
            "/api/v1/langchain/chat/stream",
            headers={"x-api-key": "test-key"},
            json={
                "messages": [{"role": "user", "content": "Hello"}],
                "model": "gpt-3.5-turbo",
                "stream": True
            }
        )
        
        assert response.status_code == 200
        assert response.headers["content-type"] == "text/event-stream; charset=utf-8"


class TestErrorHandling:
    """Test error handling scenarios."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        from langflow.main import create_app
        app = create_app()
        return TestClient(app)
    
    @patch('langflow.api.v1.langchain_router.api_key_security')
    @patch('langflow.api.v1.langchain_router.process_langchain_request')
    def test_processing_error(self, mock_process, mock_auth, client, mock_user):
        """Test processing error handling."""
        mock_auth.return_value = mock_user
        mock_process.side_effect = Exception("Processing error")
        
        response = client.post(
            "/api/v1/langchain/chat",
            headers={"x-api-key": "test-key"},
            json={
                "messages": [{"role": "user", "content": "Hello"}],
                "model": "gpt-3.5-turbo"
            }
        )
        
        assert response.status_code == 500
        data = response.json()
        assert "error" in data
        assert data["error_code"] == "UNEXPECTED_ERROR"
    
    @patch('langflow.api.v1.langchain_router.api_key_security')
    def test_validation_error(self, mock_auth, client, mock_user):
        """Test validation error handling."""
        mock_auth.return_value = mock_user
        
        response = client.post(
            "/api/v1/langchain/chat",
            headers={"x-api-key": "test-key"},
            json={
                "messages": [{"role": "user", "content": "x" * 10001}],  # Too long
                "model": "gpt-3.5-turbo"
            }
        )
        
        assert response.status_code == 400
        data = response.json()
        assert "error" in data
        assert data["error_code"] == "MESSAGE_TOO_LONG"


class TestOpenAIConfig:
    """Test OpenAI configuration."""
    
    @patch('langflow.api.v1.langchain_router.ChatOpenAI')
    def test_create_llm_success(self, mock_chat_openai):
        """Test successful LLM creation."""
        from langflow.api.v1.langchain_router import OpenAIConfig
        
        mock_instance = MagicMock()
        mock_chat_openai.return_value = mock_instance
        
        llm = OpenAIConfig.create_llm(
            model="gpt-3.5-turbo",
            temperature=0.7,
            max_tokens=100,
            api_key="test-key"
        )
        
        assert llm == mock_instance
        mock_chat_openai.assert_called_once_with(
            model="gpt-3.5-turbo",
            temperature=0.7,
            max_tokens=100,
            openai_api_key="test-key",
            streaming=True
        )
    
    @patch('langflow.api.v1.langchain_router.ChatOpenAI')
    def test_create_llm_failure(self, mock_chat_openai):
        """Test LLM creation failure."""
        from langflow.api.v1.langchain_router import OpenAIConfig
        
        mock_chat_openai.side_effect = Exception("API key invalid")
        
        with pytest.raises(Exception) as exc_info:
            OpenAIConfig.create_llm(api_key="invalid-key")
        
        assert "Failed to initialize OpenAI model" in str(exc_info.value)


if __name__ == "__main__":
    pytest.main([__file__])


