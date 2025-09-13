"""
LangChain API Integration Examples

This module provides comprehensive examples and documentation for using the
LangChain API integration endpoints in Langflow.
"""

from typing import Dict, List, Any
import json


# Example request payloads
EXAMPLE_CHAT_REQUEST = {
    "messages": [
        {
            "role": "system",
            "content": "You are a helpful AI assistant that provides accurate and concise responses."
        },
        {
            "role": "user", 
            "content": "What is the capital of France?"
        }
    ],
    "model": "gpt-3.5-turbo",
    "temperature": 0.7,
    "max_tokens": 150,
    "stream": False,
    "system_prompt": "You are a knowledgeable geography expert."
}

EXAMPLE_STREAMING_REQUEST = {
    "messages": [
        {
            "role": "user",
            "content": "Write a short story about a robot learning to paint."
        }
    ],
    "model": "gpt-4",
    "temperature": 0.8,
    "max_tokens": 500,
    "stream": True
}

EXAMPLE_CONVERSATION_REQUEST = {
    "messages": [
        {
            "role": "system",
            "content": "You are a helpful coding assistant."
        },
        {
            "role": "user",
            "content": "How do I create a Python function?"
        },
        {
            "role": "assistant",
            "content": "To create a Python function, you use the 'def' keyword followed by the function name and parameters."
        },
        {
            "role": "user",
            "content": "Can you show me an example?"
        }
    ],
    "model": "gpt-3.5-turbo",
    "temperature": 0.3
}

# Example response formats
EXAMPLE_CHAT_RESPONSE = {
    "content": "The capital of France is Paris. Paris is located in the north-central part of France and is the country's largest city and economic center.",
    "model": "gpt-3.5-turbo",
    "usage": {
        "prompt_tokens": 45,
        "completion_tokens": 28,
        "total_tokens": 73
    },
    "finish_reason": "stop",
    "session_id": None
}

EXAMPLE_STREAMING_CHUNK = {
    "content": "Once upon a time, there was a robot named",
    "is_final": False,
    "session_id": None
}

EXAMPLE_ERROR_RESPONSE = {
    "error": "Invalid model: invalid-model-name",
    "error_code": "INVALID_MODEL",
    "details": {
        "valid_models": [
            "gpt-3.5-turbo",
            "gpt-3.5-turbo-16k",
            "gpt-4",
            "gpt-4-32k",
            "gpt-4-turbo",
            "gpt-4o",
            "gpt-4o-mini"
        ]
    },
    "timestamp": "2024-01-15 10:30:45 UTC"
}

# cURL examples
CURL_CHAT_EXAMPLE = """
curl -X POST "http://localhost:7860/api/v1/langchain/chat" \\
  -H "Content-Type: application/json" \\
  -H "x-api-key: YOUR_API_KEY" \\
  -d '{
    "messages": [
      {
        "role": "user",
        "content": "What is machine learning?"
      }
    ],
    "model": "gpt-3.5-turbo",
    "temperature": 0.7,
    "max_tokens": 200
  }'
"""

CURL_STREAMING_EXAMPLE = """
curl -X POST "http://localhost:7860/api/v1/langchain/chat/stream" \\
  -H "Content-Type: application/json" \\
  -H "x-api-key: YOUR_API_KEY" \\
  -d '{
    "messages": [
      {
        "role": "user",
        "content": "Explain quantum computing in simple terms"
      }
    ],
    "model": "gpt-4",
    "temperature": 0.5,
    "stream": true
  }'
"""

CURL_MODELS_EXAMPLE = """
curl -X GET "http://localhost:7860/api/v1/langchain/models" \\
  -H "x-api-key: YOUR_API_KEY"
"""

CURL_HEALTH_EXAMPLE = """
curl -X GET "http://localhost:7860/api/v1/langchain/health" \\
  -H "x-api-key: YOUR_API_KEY"
"""

# Python client examples
PYTHON_CLIENT_EXAMPLE = """
import requests
import json

# Configuration
API_BASE_URL = "http://localhost:7860/api/v1"
API_KEY = "YOUR_API_KEY"
HEADERS = {
    "Content-Type": "application/json",
    "x-api-key": API_KEY
}

def chat_with_langchain(messages, model="gpt-3.5-turbo", temperature=0.7):
    \"\"\"Send a chat request to the LangChain API.\"\"\"
    url = f"{API_BASE_URL}/langchain/chat"
    payload = {
        "messages": messages,
        "model": model,
        "temperature": temperature,
        "max_tokens": 200
    }
    
    response = requests.post(url, headers=HEADERS, json=payload)
    
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"API Error: {response.status_code} - {response.text}")

def stream_chat_with_langchain(messages, model="gpt-4"):
    \"\"\"Stream a chat response from the LangChain API.\"\"\"
    url = f"{API_BASE_URL}/langchain/chat/stream"
    payload = {
        "messages": messages,
        "model": model,
        "stream": True
    }
    
    response = requests.post(url, headers=HEADERS, json=payload, stream=True)
    
    if response.status_code == 200:
        for line in response.iter_lines():
            if line:
                line = line.decode('utf-8')
                if line.startswith('data: '):
                    data = json.loads(line[6:])  # Remove 'data: ' prefix
                    yield data
    else:
        raise Exception(f"API Error: {response.status_code} - {response.text}")

# Example usage
if __name__ == "__main__":
    # Simple chat
    messages = [
        {"role": "user", "content": "What is artificial intelligence?"}
    ]
    
    try:
        response = chat_with_langchain(messages)
        print("Response:", response["content"])
    except Exception as e:
        print("Error:", e)
    
    # Streaming chat
    print("\\nStreaming response:")
    try:
        for chunk in stream_chat_with_langchain(messages):
            if not chunk["is_final"]:
                print(chunk["content"], end="", flush=True)
            else:
                print("\\n[Stream complete]")
                break
    except Exception as e:
        print("Error:", e)
"""

# JavaScript/Node.js client examples
JAVASCRIPT_CLIENT_EXAMPLE = """
const axios = require('axios');

// Configuration
const API_BASE_URL = 'http://localhost:7860/api/v1';
const API_KEY = 'YOUR_API_KEY';

const headers = {
    'Content-Type': 'application/json',
    'x-api-key': API_KEY
};

async function chatWithLangChain(messages, model = 'gpt-3.5-turbo', temperature = 0.7) {
    try {
        const response = await axios.post(`${API_BASE_URL}/langchain/chat`, {
            messages,
            model,
            temperature,
            max_tokens: 200
        }, { headers });
        
        return response.data;
    } catch (error) {
        throw new Error(`API Error: ${error.response?.status} - ${error.response?.data}`);
    }
}

async function streamChatWithLangChain(messages, model = 'gpt-4') {
    try {
        const response = await axios.post(`${API_BASE_URL}/langchain/chat/stream`, {
            messages,
            model,
            stream: true
        }, { 
            headers,
            responseType: 'stream'
        });
        
        return new Promise((resolve, reject) => {
            let fullResponse = '';
            
            response.data.on('data', (chunk) => {
                const lines = chunk.toString().split('\\n');
                for (const line of lines) {
                    if (line.startsWith('data: ')) {
                        try {
                            const data = JSON.parse(line.slice(6));
                            if (!data.is_final) {
                                process.stdout.write(data.content);
                                fullResponse += data.content;
                            } else {
                                console.log('\\n[Stream complete]');
                                resolve(fullResponse);
                            }
                        } catch (e) {
                            // Ignore parsing errors for incomplete chunks
                        }
                    }
                }
            });
            
            response.data.on('error', reject);
        });
    } catch (error) {
        throw new Error(`API Error: ${error.response?.status} - ${error.response?.data}`);
    }
}

// Example usage
async function main() {
    const messages = [
        { role: 'user', content: 'What is machine learning?' }
    ];
    
    try {
        // Simple chat
        const response = await chatWithLangChain(messages);
        console.log('Response:', response.content);
        
        // Streaming chat
        console.log('\\nStreaming response:');
        await streamChatWithLangChain(messages);
    } catch (error) {
        console.error('Error:', error.message);
    }
}

main();
"""

# Rate limiting information
RATE_LIMITING_INFO = {
    "chat_endpoint": {
        "limit": "30 requests per minute",
        "description": "Standard chat endpoint with moderate rate limiting"
    },
    "streaming_endpoint": {
        "limit": "10 requests per minute", 
        "description": "Streaming endpoint with lower rate limit due to resource usage"
    },
    "models_endpoint": {
        "limit": "60 requests per minute",
        "description": "Models list endpoint with higher rate limit"
    },
    "health_endpoint": {
        "limit": "No rate limiting",
        "description": "Health check endpoint with no rate limiting"
    }
}

# Error codes documentation
ERROR_CODES = {
    "TOO_MANY_MESSAGES": {
        "status_code": 400,
        "description": "Request contains more than 100 messages",
        "solution": "Reduce the number of messages in your request"
    },
    "MESSAGE_TOO_LONG": {
        "status_code": 400,
        "description": "Individual message content exceeds 10,000 characters",
        "solution": "Shorten the message content"
    },
    "INVALID_MODEL": {
        "status_code": 400,
        "description": "Specified model is not supported",
        "solution": "Use one of the supported models from the /models endpoint"
    },
    "PROCESSING_ERROR": {
        "status_code": 500,
        "description": "Internal error during request processing",
        "solution": "Retry the request or contact support"
    },
    "STREAMING_ERROR": {
        "status_code": 500,
        "description": "Error during streaming response",
        "solution": "Retry the request or use the non-streaming endpoint"
    },
    "UNEXPECTED_ERROR": {
        "status_code": 500,
        "description": "Unexpected server error",
        "solution": "Retry the request or contact support"
    }
}

# Authentication examples
AUTHENTICATION_EXAMPLES = {
    "api_key_header": {
        "description": "Include API key in request header",
        "example": "x-api-key: YOUR_API_KEY"
    },
    "api_key_query": {
        "description": "Include API key as query parameter",
        "example": "?x-api-key=YOUR_API_KEY"
    },
    "jwt_token": {
        "description": "Use JWT token for authentication (if configured)",
        "example": "Authorization: Bearer YOUR_JWT_TOKEN"
    }
}

# Best practices
BEST_PRACTICES = {
    "request_optimization": [
        "Use appropriate temperature values (0.0-2.0)",
        "Set reasonable max_tokens limits",
        "Keep messages concise and relevant",
        "Use system prompts for context"
    ],
    "error_handling": [
        "Always check response status codes",
        "Implement retry logic for 5xx errors",
        "Handle rate limiting gracefully",
        "Log errors for debugging"
    ],
    "security": [
        "Never expose API keys in client-side code",
        "Use environment variables for API keys",
        "Implement proper authentication",
        "Monitor API usage and costs"
    ],
    "performance": [
        "Use streaming for long responses",
        "Implement connection pooling",
        "Cache responses when appropriate",
        "Monitor response times"
    ]
}


