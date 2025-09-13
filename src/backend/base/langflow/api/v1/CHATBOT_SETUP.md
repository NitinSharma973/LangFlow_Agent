# Chatbot Setup Guide

This guide explains how to set up the chatbot functionality that appears on the login page.

## Overview

The chatbot integration includes:
- **Frontend**: A floating chatbot widget on the login page
- **Backend**: LangChain API endpoints for OpenAI integration
- **Public Access**: No authentication required for basic chat functionality

## Setup Instructions

### 1. Environment Configuration

Add the following environment variable to your `.env` file:

```bash
# OpenAI API Key for chatbot functionality
OPENAI_API_KEY=your_openai_api_key_here
```

### 2. Backend Setup

The backend automatically includes the LangChain API endpoints:
- `/api/v1/langchain/chat/public` - Public chat endpoint (no auth required)
- `/api/v1/langchain/chat` - Authenticated chat endpoint
- `/api/v1/langchain/chat/stream` - Streaming chat endpoint
- `/api/v1/langchain/models` - Available models list
- `/api/v1/langchain/health` - Health check

### 3. Frontend Integration

The chatbot is automatically integrated into the login page (`/login`) and includes:
- Floating chat widget in the bottom-right corner
- Minimize/maximize functionality
- Real-time messaging with OpenAI models
- Error handling and loading states

## Features

### Chatbot Widget
- **Position**: Fixed bottom-right corner
- **Size**: 320px wide, 384px tall (when expanded)
- **Minimizable**: Can be minimized to just the bot icon
- **Responsive**: Adapts to different screen sizes

### Chat Functionality
- **Real-time messaging**: Send and receive messages instantly
- **Context awareness**: Maintains conversation history
- **Error handling**: Graceful error messages for connection issues
- **Loading states**: Visual feedback during message processing

### Security
- **Public endpoint**: No authentication required for basic chat
- **Rate limiting**: Built-in protection against abuse
- **Input validation**: Sanitized user inputs
- **Error handling**: No sensitive information exposed in errors

## Usage

### For Users
1. Navigate to the login page (`/login`)
2. Click the chatbot icon in the bottom-right corner
3. Start chatting with the AI assistant
4. Use minimize/maximize buttons to control the widget

### For Developers
The chatbot can be easily customized by modifying:
- `src/frontend/src/components/chatbot/LoginChatbot.tsx` - Frontend component
- `src/backend/base/langflow/api/v1/langchain_router.py` - Backend endpoints

## API Endpoints

### Public Chat Endpoint
```http
POST /api/v1/langchain/chat/public
Content-Type: application/json

{
  "messages": [
    {
      "role": "system",
      "content": "You are a helpful AI assistant for Langflow."
    },
    {
      "role": "user",
      "content": "Hello, how can you help me?"
    }
  ],
  "model": "gpt-3.5-turbo",
  "temperature": 0.7,
  "max_tokens": 500
}
```

### Response Format
```json
{
  "content": "Hello! I'm your AI assistant for Langflow...",
  "model": "gpt-3.5-turbo",
  "usage": {
    "prompt_tokens": 45,
    "completion_tokens": 28,
    "total_tokens": 73
  },
  "finish_reason": "stop",
  "session_id": null
}
```

## Troubleshooting

### Common Issues

1. **"OpenAI API key not configured"**
   - Ensure `OPENAI_API_KEY` is set in your `.env` file
   - Restart the backend server after adding the environment variable

2. **"Connection refused" errors**
   - Make sure the backend server is running on port 7860
   - Check that the frontend can reach the backend API

3. **Chatbot not appearing**
   - Verify the frontend is running on port 3000
   - Check browser console for JavaScript errors
   - Ensure all dependencies are installed

### Debug Mode

To enable debug logging, set the log level to DEBUG in your Langflow configuration.

## Customization

### Styling
The chatbot uses Tailwind CSS classes and can be customized by modifying the component styles in `LoginChatbot.tsx`.

### Behavior
- **System prompt**: Modify the system message in the frontend component
- **Model selection**: Change the default model in the API call
- **Response limits**: Adjust `max_tokens` parameter

### Integration
The chatbot can be easily added to other pages by importing and using the `LoginChatbot` component.

## Security Considerations

- The public endpoint uses a shared OpenAI API key
- Consider implementing rate limiting for production use
- Monitor API usage to prevent abuse
- Consider implementing user-specific API keys for authenticated users

## Production Deployment

For production deployment:
1. Set up a dedicated OpenAI API key for the chatbot
2. Implement proper rate limiting
3. Add monitoring and logging
4. Consider implementing user authentication for advanced features
5. Set up proper error handling and fallback responses

