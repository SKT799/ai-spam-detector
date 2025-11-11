# Spam Email Classifier Backend

Flask API for spam email classification using fine-tuned transformer model.

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the server:
```bash
python app.py
```

Server will start at `http://localhost:5000`

## API Endpoints

### Health Check
```
GET /health
```

### Classify Email
```
POST /classify
Content-Type: application/json

{
  "text": "your email text here"
}
```

Response:
```json
{
  "success": true,
  "spam_probability": 0.85,
  "is_spam": true
}
```

## Model
- **Name**: satyam2025/spam-email-classifier
- **Type**: Fine-tuned transformer for spam classification
- **Input**: Email text (max 512 tokens)
- **Output**: Spam probability (0-1)
