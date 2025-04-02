# API Documentation

## Overview

The Activity Recommendation System provides a RESTful API for emotion detection and activity recommendations. This document describes the available endpoints, request/response formats, and authentication requirements.

## Base URL

```
http://localhost:8501/api/v1
```

## Authentication

Currently, the API does not require authentication. However, rate limiting is implemented to prevent abuse.

## Rate Limiting

- 100 requests per minute per IP address
- Rate limit headers are included in the response:
  - `X-RateLimit-Limit`: Maximum number of requests per minute
  - `X-RateLimit-Remaining`: Number of requests remaining in the current time window
  - `X-RateLimit-Reset`: Time when the rate limit resets (Unix timestamp)

## Endpoints

### Detect Face Emotion

```http
POST /emotion/face
```

Detects emotions from a facial image.

#### Request

- Content-Type: `multipart/form-data`
- Body:
  - `image`: Image file (JPEG, PNG) - max 5MB

#### Response

```json
{
  "emotion": "happy",
  "confidence": 0.95,
  "recommendations": [
    "Go for a walk in the park",
    "Listen to upbeat music",
    "Call a friend"
  ]
}
```

#### Error Responses

- `400 Bad Request`: Invalid image format or size
- `500 Internal Server Error`: Server-side error

### Detect Speech Emotion

```http
POST /emotion/speech
```

Detects emotions from a speech audio file.

#### Request

- Content-Type: `multipart/form-data`
- Body:
  - `audio`: Audio file (WAV) - max 10MB

#### Response

```json
{
  "emotion": "sad",
  "confidence": 0.88,
  "recommendations": [
    "Watch a comedy movie",
    "Take a warm bath",
    "Write in a journal"
  ]
}
```

#### Error Responses

- `400 Bad Request`: Invalid audio format or size
- `500 Internal Server Error`: Server-side error

### Get Recommendations

```http
GET /recommendations/{emotion}
```

Get activity recommendations for a specific emotion.

#### Request

- Path Parameters:
  - `emotion`: The emotion to get recommendations for (e.g., "happy", "sad")

#### Response

```json
{
  "emotion": "happy",
  "recommendations": [
    "Go for a walk in the park",
    "Listen to upbeat music",
    "Call a friend"
  ]
}
```

#### Error Responses

- `400 Bad Request`: Invalid emotion
- `404 Not Found`: Emotion not found in database

## Error Response Format

All error responses follow this format:

```json
{
  "error": {
    "code": "ERROR_CODE",
    "message": "Human-readable error message",
    "details": {
      "field": "Additional error details if available"
    }
  }
}
```

## Common Error Codes

- `INVALID_INPUT`: The request contains invalid data
- `FILE_TOO_LARGE`: The uploaded file exceeds the size limit
- `INVALID_FILE_TYPE`: The file type is not supported
- `PROCESSING_ERROR`: Error occurred while processing the input
- `NOT_FOUND`: The requested resource was not found
- `INTERNAL_ERROR`: An unexpected error occurred

## Examples

### cURL

Detect face emotion:
```bash
curl -X POST \
  http://localhost:8501/api/v1/emotion/face \
  -H 'Content-Type: multipart/form-data' \
  -F 'image=@/path/to/image.jpg'
```

Get recommendations:
```bash
curl -X GET \
  http://localhost:8501/api/v1/recommendations/happy
```

### Python

```python
import requests

# Detect face emotion
files = {'image': open('image.jpg', 'rb')}
response = requests.post('http://localhost:8501/api/v1/emotion/face', files=files)
print(response.json())

# Get recommendations
response = requests.get('http://localhost:8501/api/v1/recommendations/happy')
print(response.json())
```

## Support

For API support or to report issues, please contact:
- Email: support@activity-recommendation-system.com
- GitHub Issues: https://github.com/yourusername/activity-recommendation-system/issues
