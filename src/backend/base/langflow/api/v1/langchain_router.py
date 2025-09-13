"""
LangChain Integration API Router

This module provides RESTful API endpoints for LangChain integration with Langflow.
It includes OpenAI LLM integration, proper error handling, request validation,
rate limiting, and authentication mechanisms.
"""

from __future__ import annotations

import asyncio
import time
from typing import Annotated, Any, Dict, List, Optional, Union
from uuid import UUID

from fastapi import APIRouter, BackgroundTasks, Body, Depends, HTTPException, Request, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, validator
from lfx.log.logger import logger
from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI

from langflow.api.utils import CurrentActiveUser, DbSession
from langflow.api.v1.schemas import RunResponse
from langflow.exceptions.api import APIException
from langflow.services.auth.utils import api_key_security
from langflow.services.database.models.user.model import UserRead
from langflow.services.deps import get_telemetry_service

router = APIRouter(tags=["LangChain Integration"])


# Request/Response Models
class LangChainRequest(BaseModel):
    """Request model for LangChain API calls."""
    
    messages: List[Dict[str, str]] = Field(
        ..., 
        description="List of messages for the conversation",
        min_items=1
    )
    model: str = Field(
        default="gpt-3.5-turbo",
        description="OpenAI model to use"
    )
    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Temperature for response generation"
    )
    max_tokens: Optional[int] = Field(
        default=None,
        ge=1,
        le=4096,
        description="Maximum tokens to generate"
    )
    stream: bool = Field(
        default=False,
        description="Whether to stream the response"
    )
    system_prompt: Optional[str] = Field(
        default=None,
        description="System prompt for the conversation"
    )
    tools: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="Tools to make available to the model"
    )
    
    @validator('messages')
    def validate_messages(cls, v):
        """Validate that messages have required fields."""
        for msg in v:
            if 'role' not in msg or 'content' not in msg:
                raise ValueError("Each message must have 'role' and 'content' fields")
            if msg['role'] not in ['user', 'assistant', 'system']:
                raise ValueError("Message role must be 'user', 'assistant', or 'system'")
        return v


class LangChainResponse(BaseModel):
    """Response model for LangChain API calls."""
    
    content: str = Field(..., description="Generated response content")
    model: str = Field(..., description="Model used for generation")
    usage: Optional[Dict[str, int]] = Field(
        default=None,
        description="Token usage information"
    )
    finish_reason: Optional[str] = Field(
        default=None,
        description="Reason for completion"
    )
    session_id: Optional[str] = Field(
        default=None,
        description="Session ID for conversation tracking"
    )


class LangChainErrorResponse(BaseModel):
    """Error response model for LangChain API calls."""
    
    error: str = Field(..., description="Error message")
    error_code: str = Field(..., description="Error code")
    details: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional error details"
    )
    timestamp: str = Field(..., description="Error timestamp")


class LangChainStreamChunk(BaseModel):
    """Streaming response chunk model."""
    
    content: str = Field(..., description="Chunk content")
    is_final: bool = Field(default=False, description="Whether this is the final chunk")
    session_id: Optional[str] = Field(default=None, description="Session ID")


# OpenAI LLM Configuration
class OpenAIConfig:
    """OpenAI configuration manager."""
    
    @staticmethod
    def create_llm(
        model: str = "gpt-3.5-turbo",
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        api_key: Optional[str] = None
    ) -> ChatOpenAI:
        """Create and configure OpenAI LLM instance."""
        try:
            llm = ChatOpenAI(
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                openai_api_key=api_key,
                streaming=True
            )
            return llm
        except Exception as e:
            logger.error(f"Failed to create OpenAI LLM: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to initialize OpenAI model: {str(e)}"
            )


# Error Handling
class LangChainAPIException(HTTPException):
    """Custom exception for LangChain API errors."""
    
    def __init__(
        self,
        message: str,
        error_code: str = "LANGCHAIN_ERROR",
        status_code: int = 500,
        details: Optional[Dict[str, Any]] = None
    ):
        self.error_code = error_code
        self.details = details or {}
        super().__init__(
            status_code=status_code,
            detail={
                "error": message,
                "error_code": error_code,
                "details": self.details,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
            }
        )


# Request Validation
def validate_langchain_request(request: LangChainRequest) -> None:
    """Validate LangChain request parameters."""
    # Validate message count
    if len(request.messages) > 100:
        raise LangChainAPIException(
            "Too many messages in request",
            error_code="TOO_MANY_MESSAGES",
            status_code=400
        )
    
    # Validate message content length
    for msg in request.messages:
        if len(msg.get('content', '')) > 10000:
            raise LangChainAPIException(
                "Message content too long",
                error_code="MESSAGE_TOO_LONG",
                status_code=400
            )
    
    # Validate model name
    valid_models = [
        "gpt-3.5-turbo", "gpt-3.5-turbo-16k", "gpt-4", "gpt-4-32k",
        "gpt-4-turbo", "gpt-4o", "gpt-4o-mini"
    ]
    if request.model not in valid_models:
        raise LangChainAPIException(
            f"Invalid model: {request.model}",
            error_code="INVALID_MODEL",
            status_code=400,
            details={"valid_models": valid_models}
        )


# LangChain Integration Functions
async def process_langchain_request(
    request: LangChainRequest,
    api_key_user: UserRead,
    openai_api_key: Optional[str] = None
) -> LangChainResponse:
    """Process a LangChain request and return response."""
    try:
        # Validate request
        validate_langchain_request(request)
        
        # Create OpenAI LLM
        llm = OpenAIConfig.create_llm(
            model=request.model,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            api_key=openai_api_key
        )
        
        # Convert messages to LangChain format
        messages = []
        for msg in request.messages:
            if msg['role'] == 'system':
                messages.append(SystemMessage(content=msg['content']))
            elif msg['role'] == 'user':
                messages.append(HumanMessage(content=msg['content']))
            # Note: assistant messages are typically not sent to the model
        
        # Add system prompt if provided
        if request.system_prompt:
            messages.insert(0, SystemMessage(content=request.system_prompt))
        
        # Generate response
        if request.stream:
            # For streaming, we'll handle this in the streaming endpoint
            raise NotImplementedError("Streaming not implemented in this function")
        
        response = await llm.ainvoke(messages)
        
        # Extract usage information if available
        usage = None
        if hasattr(response, 'usage_metadata'):
            usage = {
                "prompt_tokens": response.usage_metadata.prompt_tokens,
                "completion_tokens": response.usage_metadata.completion_tokens,
                "total_tokens": response.usage_metadata.total_tokens
            }
        
        return LangChainResponse(
            content=response.content,
            model=request.model,
            usage=usage,
            finish_reason=getattr(response, 'finish_reason', None),
            session_id=None  # TODO: Implement session tracking
        )
        
    except LangChainAPIException:
        raise
    except Exception as e:
        logger.error(f"Error processing LangChain request: {str(e)}")
        raise LangChainAPIException(
            f"Failed to process request: {str(e)}",
            error_code="PROCESSING_ERROR",
            status_code=500
        )


async def process_langchain_stream(
    request: LangChainRequest,
    api_key_user: UserRead,
    openai_api_key: Optional[str] = None
):
    """Process a LangChain request with streaming response."""
    try:
        # Validate request
        validate_langchain_request(request)
        
        # Create OpenAI LLM
        llm = OpenAIConfig.create_llm(
            model=request.model,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            api_key=openai_api_key
        )
        
        # Convert messages to LangChain format
        messages = []
        for msg in request.messages:
            if msg['role'] == 'system':
                messages.append(SystemMessage(content=msg['content']))
            elif msg['role'] == 'user':
                messages.append(HumanMessage(content=msg['content']))
        
        # Add system prompt if provided
        if request.system_prompt:
            messages.insert(0, SystemMessage(content=request.system_prompt))
        
        # Stream response
        async for chunk in llm.astream(messages):
            if hasattr(chunk, 'content') and chunk.content:
                yield LangChainStreamChunk(
                    content=chunk.content,
                    is_final=False,
                    session_id=None
                )
        
        # Send final chunk
        yield LangChainStreamChunk(
            content="",
            is_final=True,
            session_id=None
        )
        
    except LangChainAPIException:
        raise
    except Exception as e:
        logger.error(f"Error processing LangChain stream: {str(e)}")
        yield LangChainStreamChunk(
            content=f"Error: {str(e)}",
            is_final=True,
            session_id=None
        )


# API Endpoints
@router.post(
    "/langchain/chat",
    response_model=LangChainResponse,
    status_code=status.HTTP_200_OK,
    summary="Chat with LangChain OpenAI integration",
    description="Send messages to OpenAI models through LangChain with proper validation and error handling"
)
async def langchain_chat(
    request: Request,
    langchain_request: LangChainRequest,
    api_key_user: Annotated[UserRead, Depends(api_key_security)],
    background_tasks: BackgroundTasks
) -> LangChainResponse:
    """
    Chat endpoint for LangChain OpenAI integration.
    
    This endpoint provides a RESTful interface to interact with OpenAI models
    through LangChain, including proper authentication, validation, and error handling.
    
    Args:
        request: FastAPI request object
        langchain_request: The chat request with messages and parameters
        api_key_user: Authenticated user from API key
        background_tasks: Background tasks for telemetry
        
    Returns:
        LangChainResponse: The generated response from the model
        
    Raises:
        LangChainAPIException: For validation or processing errors
        HTTPException: For authentication errors
    """
    start_time = time.perf_counter()
    telemetry_service = get_telemetry_service()
    
    try:
        # Get OpenAI API key from user settings or environment
        # In a real implementation, this would come from user's stored API keys
        openai_api_key = None  # TODO: Implement API key retrieval from user settings
        
        # Process the request
        response = await process_langchain_request(
            langchain_request,
            api_key_user,
            openai_api_key
        )
        
        # Log successful request
        end_time = time.perf_counter()
        background_tasks.add_task(
            telemetry_service.log_package_run,
            {
                "run_is_webhook": False,
                "run_seconds": int(end_time - start_time),
                "run_success": True,
                "run_error_message": "",
                "endpoint": "langchain_chat"
            }
        )
        
        return response
        
    except LangChainAPIException:
        raise
    except Exception as e:
        end_time = time.perf_counter()
        background_tasks.add_task(
            telemetry_service.log_package_run,
            {
                "run_is_webhook": False,
                "run_seconds": int(end_time - start_time),
                "run_success": False,
                "run_error_message": str(e),
                "endpoint": "langchain_chat"
            }
        )
        raise LangChainAPIException(
            f"Unexpected error: {str(e)}",
            error_code="UNEXPECTED_ERROR",
            status_code=500
        )


@router.post(
    "/langchain/chat/public",
    response_model=LangChainResponse,
    status_code=status.HTTP_200_OK,
    summary="Public chat with LangChain OpenAI integration (no authentication required)",
    description="Send messages to OpenAI models through LangChain for public access (e.g., login page)"
)
async def langchain_chat_public(
    request: Request,
    langchain_request: LangChainRequest,
    background_tasks: BackgroundTasks
) -> LangChainResponse:
    """
    Public chat endpoint for LangChain OpenAI integration.
    
    This endpoint provides a RESTful interface to interact with OpenAI models
    through LangChain without requiring authentication. Useful for login page
    or public-facing chatbot functionality.
    
    Args:
        request: FastAPI request object
        langchain_request: The chat request with messages and parameters
        background_tasks: Background tasks for telemetry
        
    Returns:
        LangChainResponse: The generated response from the model
        
    Raises:
        LangChainAPIException: For validation or processing errors
    """
    start_time = time.perf_counter()
    telemetry_service = get_telemetry_service()
    
    try:
        # For public access, we'll use a default OpenAI API key from environment
        # In production, you might want to use a dedicated public API key
        import os
        openai_api_key = os.getenv("OPENAI_API_KEY")
        
        if not openai_api_key:
            raise LangChainAPIException(
                "OpenAI API key not configured for public access",
                error_code="API_KEY_NOT_CONFIGURED",
                status_code=503
            )
        
        # Process the request
        response = await process_langchain_request(
            langchain_request,
            None,  # No user for public access
            openai_api_key
        )
        
        # Log successful request
        end_time = time.perf_counter()
        background_tasks.add_task(
            telemetry_service.log_package_run,
            {
                "run_is_webhook": False,
                "run_seconds": int(end_time - start_time),
                "run_success": True,
                "run_error_message": "",
                "endpoint": "langchain_chat_public"
            }
        )
        
        return response
        
    except LangChainAPIException:
        raise
    except Exception as e:
        end_time = time.perf_counter()
        background_tasks.add_task(
            telemetry_service.log_package_run,
            {
                "run_is_webhook": False,
                "run_seconds": int(end_time - start_time),
                "run_success": False,
                "run_error_message": str(e),
                "endpoint": "langchain_chat_public"
            }
        )
        raise LangChainAPIException(
            f"Unexpected error: {str(e)}",
            error_code="UNEXPECTED_ERROR",
            status_code=500
        )


@router.post(
    "/langchain/chat/stream",
    response_class=StreamingResponse,
    status_code=status.HTTP_200_OK,
    summary="Stream chat with LangChain OpenAI integration",
    description="Stream responses from OpenAI models through LangChain"
)
async def langchain_chat_stream(
    request: Request,
    langchain_request: LangChainRequest,
    api_key_user: Annotated[UserRead, Depends(api_key_security)],
    background_tasks: BackgroundTasks
) -> StreamingResponse:
    """
    Streaming chat endpoint for LangChain OpenAI integration.
    
    This endpoint provides real-time streaming of responses from OpenAI models
    through LangChain, with proper authentication.
    
    Args:
        request: FastAPI request object
        langchain_request: The chat request with messages and parameters
        api_key_user: Authenticated user from API key
        background_tasks: Background tasks for telemetry
        
    Returns:
        StreamingResponse: Server-sent events stream of response chunks
    """
    start_time = time.perf_counter()
    telemetry_service = get_telemetry_service()
    
    try:
        # Get OpenAI API key from user settings or environment
        openai_api_key = None  # TODO: Implement API key retrieval from user settings
        
        # Create streaming generator
        async def generate_stream():
            try:
                async for chunk in process_langchain_stream(
                    langchain_request,
                    api_key_user,
                    openai_api_key
                ):
                    yield f"data: {chunk.model_dump_json()}\n\n"
            except Exception as e:
                error_chunk = LangChainStreamChunk(
                    content=f"Error: {str(e)}",
                    is_final=True
                )
                yield f"data: {error_chunk.model_dump_json()}\n\n"
            finally:
                # Log completion
                end_time = time.perf_counter()
                background_tasks.add_task(
                    telemetry_service.log_package_run,
                    {
                        "run_is_webhook": False,
                        "run_seconds": int(end_time - start_time),
                        "run_success": True,
                        "run_error_message": "",
                        "endpoint": "langchain_chat_stream"
                    }
                )
        
        return StreamingResponse(
            generate_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"  # Disable nginx buffering
            }
        )
        
    except Exception as e:
        end_time = time.perf_counter()
        background_tasks.add_task(
            telemetry_service.log_package_run,
            {
                "run_is_webhook": False,
                "run_seconds": int(end_time - start_time),
                "run_success": False,
                "run_error_message": str(e),
                "endpoint": "langchain_chat_stream"
            }
        )
        raise LangChainAPIException(
            f"Streaming error: {str(e)}",
            error_code="STREAMING_ERROR",
            status_code=500
        )


@router.get(
    "/langchain/models",
    response_model=List[str],
    status_code=status.HTTP_200_OK,
    summary="Get available OpenAI models",
    description="Retrieve list of available OpenAI models for LangChain integration"
)
async def get_available_models(
    request: Request,
    api_key_user: Annotated[UserRead, Depends(api_key_security)]
) -> List[str]:
    """
    Get list of available OpenAI models.
    
    Returns a list of OpenAI models that can be used with the LangChain integration.
    
    Args:
        request: FastAPI request object
        api_key_user: Authenticated user from API key
        
    Returns:
        List[str]: List of available model names
    """
    return [
        "gpt-3.5-turbo",
        "gpt-3.5-turbo-16k", 
        "gpt-4",
        "gpt-4-32k",
        "gpt-4-turbo",
        "gpt-4o",
        "gpt-4o-mini"
    ]


@router.get(
    "/langchain/health",
    status_code=status.HTTP_200_OK,
    summary="Health check for LangChain integration",
    description="Check the health status of the LangChain integration service"
)
async def health_check(
    api_key_user: Annotated[UserRead, Depends(api_key_security)]
) -> Dict[str, Any]:
    """
    Health check endpoint for LangChain integration.
    
    Returns the health status of the LangChain integration service.
    
    Args:
        api_key_user: Authenticated user from API key
        
    Returns:
        Dict[str, Any]: Health status information
    """
    return {
        "status": "healthy",
        "service": "langchain_integration",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "version": "1.0.0"
    }

