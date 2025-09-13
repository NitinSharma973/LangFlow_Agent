# LangChain API Integration

This document provides comprehensive documentation for the LangChain API integration endpoints in Langflow.

## Overview

The LangChain API integration provides RESTful endpoints for interacting with OpenAI models through LangChain, including:

- **Chat endpoint**: Send messages and receive responses from OpenAI models
- **Streaming endpoint**: Real-time streaming of responses
- **Models endpoint**: List available OpenAI models
- **Health endpoint**: Check service health status

## Features

### ✅ Implemented Features

- **RESTful API Design**: Clean, RESTful endpoints following OpenAPI standards
- **OpenAI LLM Integration**: Full integration with OpenAI models through LangChain
- **Authentication**: API key and JWT token authentication
- **Request Validation**: Comprehensive input validation with detailed error messages
- **Rate Limiting**: Configurable rate limiting per endpoint
- **Error Handling**: Structured error responses with error codes and details
- **Streaming Support**: Real-time streaming responses for better user experience
- **Telemetry**: Request tracking and performance monitoring
- **Documentation**: Comprehensive API documentation with examples

### 🔧 Configuration

The API supports the following OpenAI models:
- `gpt-3.5-turbo`
- `gpt-3.5-turbo-16k`
- `gpt-4`
- `gpt-4-32k`
- `gpt-4-turbo`
- `gpt-4o`
- `gpt-4o-mini`

## API Endpoints

### 1. Chat Endpoint

**POST** `/api/v1/langchain/chat`

Send messages to OpenAI models and receive responses.

#### Request Body

```json
{
  "messages": [
    {
      "role": "system",
      "content": "You are a helpful AI assistant."
    },
    {
      "role": "user",
      "content": "What is the capital of France?"
    }
  ],
  "model": "gpt-3.5-turbo",
  "temperature": 0.7,
  "max_tokens": 200,
  "stream": false,
  "system_prompt": "You are a geography expert."
}
```

#### Response

```json
{
  "content": "The capital of France is Paris.",
  "model": "gpt-3.5-turbo",
  "usage": {
    "prompt_tokens": 45,
    "completion_tokens": 8,
    "total_tokens": 53
  },
  "finish_reason": "stop",
  "session_id": null
}
```

#### Rate Limiting
- **Limit**: 30 requests per minute
- **Headers**: `X-RateLimit-Limit`, `X-RateLimit-Remaining`, `X-RateLimit-Reset`

### 2. Streaming Chat Endpoint

**POST** `/api/v1/langchain/chat/stream`

Stream responses from OpenAI models in real-time.

#### Request Body

Same as chat endpoint, but `stream` should be `true`.

#### Response

Server-sent events stream:

```
data: {"content": "The", "is_final": false, "session_id": null}

data: {"content": " capital", "is_final": false, "session_id": null}

data: {"content": "", "is_final": true, "session_id": null}
```

#### Rate Limiting
- **Limit**: 10 requests per minute
- **Headers**: `X-RateLimit-Limit`, `X-RateLimit-Remaining`, `X-RateLimit-Reset`

### 3. Models Endpoint

**GET** `/api/v1/langchain/models`

Get list of available OpenAI models.

#### Response

```json
[
  "gpt-3.5-turbo",
  "gpt-3.5-turbo-16k",
  "gpt-4",
  "gpt-4-32k",
  "gpt-4-turbo",
  "gpt-4o",
  "gpt-4o-mini"
]
```

#### Rate Limiting
- **Limit**: 60 requests per minute

### 4. Health Endpoint

**GET** `/api/v1/langchain/health`

Check the health status of the LangChain integration service.

#### Response

```json
{
  "status": "healthy",
  "service": "langchain_integration",
  "timestamp": "2024-01-15 10:30:45 UTC",
  "version": "1.0.0"
}
```

#### Rate Limiting
- **Limit**: No rate limiting

## Authentication

The API supports multiple authentication methods:

### API Key Authentication

Include your API key in the request header:

```bash
curl -H "x-api-key: YOUR_API_KEY" \
     -H "Content-Type: application/json" \
     -X POST http://localhost:7860/api/v1/langchain/chat \
     -d '{"messages": [{"role": "user", "content": "Hello"}]}'
```

### JWT Token Authentication

Include your JWT token in the Authorization header:

```bash
curl -H "Authorization: Bearer YOUR_JWT_TOKEN" \
     -H "Content-Type: application/json" \
     -X POST http://localhost:7860/api/v1/langchain/chat \
     -d '{"messages": [{"role": "user", "content": "Hello"}]}'
```

## Error Handling

The API returns structured error responses with the following format:

```json
{
  "error": "Error message",
  "error_code": "ERROR_CODE",
  "details": {
    "additional": "information"
  },
  "timestamp": "2024-01-15 10:30:45 UTC"
}
```

### Common Error Codes

| Error Code | Status Code | Description |
|------------|-------------|-------------|
| `TOO_MANY_MESSAGES` | 400 | Request contains more than 100 messages |
| `MESSAGE_TOO_LONG` | 400 | Individual message content exceeds 10,000 characters |
| `INVALID_MODEL` | 400 | Specified model is not supported |
| `PROCESSING_ERROR` | 500 | Internal error during request processing |
| `STREAMING_ERROR` | 500 | Error during streaming response |
| `UNEXPECTED_ERROR` | 500 | Unexpected server error |

## Usage Examples

### Python Client

```python
import requests
import json

API_BASE_URL = "http://localhost:7860/api/v1"
API_KEY = "YOUR_API_KEY"
HEADERS = {
    "Content-Type": "application/json",
    "x-api-key": API_KEY
}

def chat_with_langchain(messages, model="gpt-3.5-turbo"):
    url = f"{API_BASE_URL}/langchain/chat"
    payload = {
        "messages": messages,
        "model": model,
        "temperature": 0.7
    }
    
    response = requests.post(url, headers=HEADERS, json=payload)
    return response.json()

# Example usage
messages = [
    {"role": "user", "content": "What is artificial intelligence?"}
]
response = chat_with_langchain(messages)
print(response["content"])
```

### JavaScript Client

```javascript
const axios = require('axios');

const API_BASE_URL = 'http://localhost:7860/api/v1';
const API_KEY = 'YOUR_API_KEY';

const headers = {
    'Content-Type': 'application/json',
    'x-api-key': API_KEY
};

async function chatWithLangChain(messages, model = 'gpt-3.5-turbo') {
    try {
        const response = await axios.post(`${API_BASE_URL}/langchain/chat`, {
            messages,
            model,
            temperature: 0.7
        }, { headers });
        
        return response.data;
    } catch (error) {
        throw new Error(`API Error: ${error.response?.status} - ${error.response?.data}`);
    }
}

// Example usage
const messages = [
    { role: 'user', content: 'What is machine learning?' }
];

chatWithLangChain(messages)
    .then(response => console.log(response.content))
    .catch(error => console.error(error.message));
```

### cURL Examples

#### Simple Chat

```bash
curl -X POST "http://localhost:7860/api/v1/langchain/chat" \
  -H "Content-Type: application/json" \
  -H "x-api-key: YOUR_API_KEY" \
  -d '{
    "messages": [
      {
        "role": "user",
        "content": "What is machine learning?"
      }
    ],
    "model": "gpt-3.5-turbo",
    "temperature": 0.7
  }'
```

#### Streaming Chat

```bash
curl -X POST "http://localhost:7860/api/v1/langchain/chat/stream" \
  -H "Content-Type: application/json" \
  -H "x-api-key: YOUR_API_KEY" \
  -d '{
    "messages": [
      {
        "role": "user",
        "content": "Explain quantum computing"
      }
    ],
    "model": "gpt-4",
    "stream": true
  }'
```

## Rate Limiting

The API implements rate limiting to ensure fair usage and system stability:

| Endpoint | Rate Limit | Description |
|----------|------------|-------------|
| `/chat` | 30/minute | Standard chat endpoint |
| `/chat/stream` | 10/minute | Streaming endpoint (lower limit due to resource usage) |
| `/models` | 60/minute | Models list endpoint |
| `/health` | No limit | Health check endpoint |

Rate limit headers are included in responses:
- `X-RateLimit-Limit`: Maximum requests allowed
- `X-RateLimit-Remaining`: Remaining requests in current window
- `X-RateLimit-Reset`: Time when the rate limit resets

## Best Practices

### Request Optimization

1. **Use appropriate temperature values** (0.0-2.0)
2. **Set reasonable max_tokens limits** to control costs
3. **Keep messages concise and relevant**
4. **Use system prompts for context** instead of including in every message

### Error Handling

1. **Always check response status codes**
2. **Implement retry logic for 5xx errors**
3. **Handle rate limiting gracefully** with exponential backoff
4. **Log errors for debugging** and monitoring

### Security

1. **Never expose API keys in client-side code**
2. **Use environment variables for API keys**
3. **Implement proper authentication**
4. **Monitor API usage and costs**

### Performance

1. **Use streaming for long responses** to improve perceived performance
2. **Implement connection pooling** for high-volume applications
3. **Cache responses when appropriate**
4. **Monitor response times** and optimize accordingly

## Testing

The API includes comprehensive tests covering:

- Request validation
- Authentication
- Error handling
- Rate limiting
- Streaming functionality
- Response formatting

Run tests with:

```bash
pytest src/backend/base/tests/test_langchain_api.py -v
```

## Monitoring and Telemetry

The API automatically tracks:

- Request success/failure rates
- Response times
- Error rates by type
- Rate limiting events
- Usage patterns

Telemetry data is sent to the configured telemetry service for monitoring and analysis.

## Troubleshooting

### Common Issues

1. **Authentication Errors**
   - Verify API key is correct
   - Check API key permissions
   - Ensure proper header format

2. **Rate Limiting**
   - Implement exponential backoff
   - Consider upgrading rate limits
   - Monitor usage patterns

3. **Model Errors**
   - Verify model name is supported
   - Check model availability
   - Review model-specific parameters

4. **Streaming Issues**
   - Check network connectivity
   - Verify client supports server-sent events
   - Handle connection interruptions gracefully

### Debug Mode

Enable debug logging by setting the log level to DEBUG in your Langflow configuration.

## Contributing

When contributing to the LangChain API integration:

1. Follow the existing code patterns
2. Add comprehensive tests for new features
3. Update documentation
4. Ensure backward compatibility
5. Follow security best practices

## License

This API integration is part of the Langflow project and follows the same license terms.


