# AI-Powered Email Spam Detector

An intelligent email classification system that distinguishes spam from legitimate emails (ham) using machine learning. This project combines a Flask backend with scikit-learn and a modern React frontend to deliver real-time spam detection with explainable results.

> **Live Demo**: [Try it here](#) | **Repository**: [github.com/SKT799/ai-spam-detector](https://github.com/SKT799/ai-spam-detector)

---

## Table of Contents
- [Overview](#overview)
- [The Theory Behind It](#the-theory-behind-it)
- [Mathematical Foundation](#mathematical-foundation)
- [Backend Architecture](#backend-architecture)
- [Frontend Design](#frontend-design)
- [Getting Started](#getting-started)
- [Deployment](#deployment)
- [Future Improvements](#future-improvements)

---

## Overview

Email spam detection is a classic text classification problem. Instead of relying on simple keyword matching (which spammers can easily evade), this project uses **statistical machine learning** to understand the underlying patterns in email text.

The system processes incoming email text through several stages:
1. **Text cleaning** - removes noise (URLs, emails, HTML)
2. **Feature extraction** - converts text into numerical vectors using TF-IDF
3. **Classification** - applies a trained ML model to predict spam probability
4. **Explainability** - shows which words influenced the decision

---

## The Theory Behind It

### Why TF-IDF?

When we read an email, certain words immediately stand out. Words like "winner", "free", "click here" might signal spam, while words like "meeting", "report", "attached" suggest legitimate communication. But how do we teach a computer to recognize these patterns?

Simple word counting doesn't work well because:
- Common words like "the", "and", "is" appear everywhere
- Rare words that are actually meaningful get lost in the noise
- Document length varies wildly

**TF-IDF (Term Frequency - Inverse Document Frequency)** solves this elegantly. It's a numerical statistic that reflects how important a word is to a document within a collection of documents.

### The Mathematics

Let's break down TF-IDF step by step.

#### 1. Term Frequency (TF)

**TF** measures how often a word appears in a document. The intuition: if a word appears many times, it's probably important to that document.

```
TF(t, d) = (Number of times term t appears in document d) / (Total number of terms in document d)
```

**Example**: In an email with 100 words, if "lottery" appears 5 times:
```
TF("lottery") = 5/100 = 0.05
```

#### 2. Inverse Document Frequency (IDF)

**IDF** measures how rare a word is across all documents. The intuition: rare words are more informative than common ones.

```
IDF(t) = log(Total number of documents / Number of documents containing term t)
```

**Example**: If we have 1000 emails and "lottery" appears in only 10 of them:
```
IDF("lottery") = log(1000/10) = log(100) ≈ 4.605
```

But if "the" appears in 999 emails:
```
IDF("the") = log(1000/999) ≈ 0.001
```

#### 3. TF-IDF Score

The final TF-IDF score is simply the product:

```
TF-IDF(t, d) = TF(t, d) × IDF(t)
```

**Why this works**: 
- Words that appear frequently in a document (high TF) but rarely across documents (high IDF) get high scores
- Words that appear everywhere (low IDF) get low scores, even if they're frequent in one document
- This naturally filters out common "stop words" while highlighting distinctive terms

**Continuing our example**:
```
TF-IDF("lottery") = 0.05 × 4.605 ≈ 0.230
TF-IDF("the") = 0.20 × 0.001 ≈ 0.0002
```

Even though "the" appears 20 times (TF = 0.20) and "lottery" only 5 times, "lottery" gets a much higher TF-IDF score because it's more distinctive.

#### 4. Vector Representation

Once we calculate TF-IDF scores for every word, we can represent any email as a **numerical vector**:

```
Email = [TF-IDF(word₁), TF-IDF(word₂), ..., TF-IDF(wordₙ)]
```

This transforms text (which machines can't directly process) into numbers (which they can).

### Classification Model

After vectorization, we feed these numerical vectors into a **supervised learning classifier** (trained on labeled spam/ham examples). Common choices include:

- **Logistic Regression**: Models the probability P(spam|email) using a linear decision boundary
- **Naive Bayes**: Uses Bayes' theorem assuming word independence
- **Random Forest**: Ensemble of decision trees voting on the classification
- **Support Vector Machine**: Finds the optimal hyperplane separating spam from ham

In this project, the model is pre-trained and stored in Hugging Face. It outputs **probabilities**:
```
P(spam|email) and P(ham|email) where P(spam) + P(ham) = 1
```

### Decision Threshold

We don't just classify based on "spam probability > 0.5". Instead, we use a **configurable threshold** (default 0.8) to reduce false positives:

```python
if P(spam|email) >= 0.8:
    classification = "SPAM"
else:
    classification = "HAM"
```

This means we only mark something as spam if we're quite confident (≥80% probability). This is tunable based on your tolerance for false positives vs. false negatives.

### Evaluation Metrics

When training/evaluating the model, we use:

**Confusion Matrix**:
```
                 Predicted
                 HAM   SPAM
Actual HAM     [ TN  |  FP ]
       SPAM    [ FN  |  TP ]
```

**Metrics**:
- **Accuracy** = (TP + TN) / Total
- **Precision** = TP / (TP + FP) — "Of emails marked spam, how many actually were?"
- **Recall** = TP / (TP + FN) — "Of actual spam, how many did we catch?"
- **F1 Score** = 2 × (Precision × Recall) / (Precision + Recall) — harmonic mean
- **ROC-AUC** — measures the model's ability to separate classes across all thresholds

For spam detection, **high precision** is often prioritized (we don't want to mark legitimate emails as spam), even if it means lower recall (some spam gets through).

---

## Backend Architecture

### Tech Stack
- **Flask** - lightweight Python web framework
- **scikit-learn** - ML model and TF-IDF vectorizer
- **Hugging Face Hub** - model artifact storage
- **pickle** - model serialization

### File Structure
```
backend/
├── app.py              # Main Flask application
├── requirements.txt    # Python dependencies
└── README.md          # Backend-specific docs
```

### Key Components

#### 1. Model Loading
```python
# Download pre-trained artifacts from Hugging Face
model_path = hf_hub_download(repo_id=MODEL_REPO, 
                              filename="spam_classifier_model.pkl",
                              token=HF_TOKEN)

vectorizer_path = hf_hub_download(repo_id=MODEL_REPO,
                                  filename="tfidf_vectorizer.pkl",
                                  token=HF_TOKEN)
```

The model and vectorizer are loaded once at startup and kept in memory for fast predictions.

#### 2. Text Preprocessing
```python
def clean_email_text(text):
    text = text.lower()                          # Normalize case
    text = re.sub(r'\S*@\S*\s?', '', text)       # Remove emails
    text = re.sub(r'http\S+|www\.\S+', '', text) # Remove URLs
    text = re.sub(r'<.*?>', '', text)            # Remove HTML
    text = re.sub(r'[^a-zA-Z\s!?.]', ' ', text)  # Keep only letters
    text = ' '.join(text.split())                # Normalize whitespace
    return text
```

This cleaning is crucial because:
- URLs and emails are spam giveaways that would leak into training
- HTML tags add noise
- Case normalization ensures "FREE" and "free" are treated the same

#### 3. Prediction Pipeline
```python
def predict_spam(text, threshold=0.8):
    # Clean the text
    cleaned = clean_email_text(text)
    
    # Transform to TF-IDF vector
    text_vector = vectorizer.transform([cleaned])
    
    # Get probabilities from model
    probabilities = model.predict_proba(text_vector)[0]
    spam_prob = probabilities[1]
    
    # Apply threshold
    is_spam = spam_prob >= threshold
    
    return {
        'spam_probability': spam_prob,
        'is_spam': is_spam,
        'classification': 'SPAM' if is_spam else 'HAM',
        ...
    }
```

#### 4. Explainability
The backend returns the **top TF-IDF features** (words) from the input:
```python
# Get TF-IDF scores for this email
feature_names = vectorizer.get_feature_names_out()
feature_values = text_vector.toarray()[0]

# Find top scoring words
top_indices = feature_values.argsort()[-10:][::-1]
top_features = [
    {'word': feature_names[idx], 'tfidf_score': feature_values[idx]}
    for idx in top_indices if feature_values[idx] > 0
]
```

This helps users understand **why** the model made its decision.

### API Endpoints

**POST /api/classify**
```json
Request:
{
    "text": "Congratulations! You've won a FREE iPhone...",
    "threshold": 0.8  // optional
}

Response:
{
    "success": true,
    "spam_probability": 0.95,
    "ham_probability": 0.05,
    "is_spam": true,
    "classification": "SPAM",
    "top_features": [
        {"word": "free", "tfidf_score": 0.456},
        {"word": "won", "tfidf_score": 0.389}
    ],
    "text_stats": {...}
}
```

**GET /api/health**
```json
{
    "status": "ok",
    "model": "satyam2025/spam-email-classifier"
}
```

---

## Frontend Design

The frontend is built with **React + Vite + TypeScript** for a modern, fast, and type-safe development experience.

### Tech Stack
- **React 18** - component-based UI
- **TypeScript** - type safety
- **Vite** - blazingly fast dev server and build tool
- **Tailwind CSS** - utility-first styling
- **Lucide Icons** - clean iconography

### User Experience

The interface is designed to be **intuitive and visually engaging**:

1. **Input area**: Large textarea for pasting email content
2. **Real-time analysis**: Submit button triggers API call
3. **Visual feedback**: 
   - Animated gauge showing spam probability
   - Color-coded results (red for spam, green for ham)
   - Floating animated blobs for visual interest
4. **Explainability panel**: Shows top contributing words with their TF-IDF scores

### Key Components

#### SpamGauge.tsx
A semicircular gauge that visually represents spam probability:
```tsx
// Converts probability (0-1) to angle (0-180°)
const angle = probability * 180
```
The needle animates smoothly and the color transitions from green → yellow → red as probability increases.

#### FloatingBlob.tsx
CSS-animated gradient blobs that add visual depth without distracting from content. Uses `@keyframes` for smooth floating motion.

#### API Integration
```typescript
const response = await fetch('/api/classify', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text: emailText, threshold: 0.8 })
})

const data = await response.json()
setSpamProbability(data.spam_probability)
setTopFeatures(data.top_features)
```

### Build Output
The frontend compiles to static files in `frontend/build/`:
```
build/
├── index.html
└── assets/
    ├── index-[hash].js
    └── index-[hash].css
```

These can be deployed to **any static host** (Netlify, Vercel, GitHub Pages, S3, etc.).

---

## Getting Started

### Prerequisites
- Python 3.8+
- Node.js 16+ (for frontend development)
- Git

### Backend Setup

1. **Clone the repository**
```bash
git clone https://github.com/SKT799/ai-spam-detector.git
cd ai-spam-detector
```

2. **Create virtual environment**
```bash
cd backend
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Set environment variables**

Create a `backend/.env` file:
```
HF_TOKEN=your_huggingface_token_here
PORT=5000
```

Or export directly:
```bash
# Windows PowerShell
$env:HF_TOKEN="your_token"

# Linux/Mac
export HF_TOKEN="your_token"
```

5. **Run the server**
```bash
python app.py
```

The backend will start on `http://localhost:5000`.

### Frontend Setup (Development)

1. **Navigate to frontend**
```bash
cd frontend
```

2. **Install dependencies**
```bash
npm install
```

3. **Start dev server**
```bash
npm run dev
```

The frontend will start on `http://localhost:5173` and proxy API requests to the backend.

4. **Build for production**
```bash
npm run build
```

Output will be in `frontend/build/`.

---

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, open an issue first to discuss what you'd like to change.

---

## License

MIT License - feel free to use this project for learning or commercial purposes.

---

## Acknowledgments

- **scikit-learn** team for excellent ML tools
- **Hugging Face** for model hosting infrastructure
- The open-source community for inspiration and tools

---

Project Link: [github.com/SKT799/ai-spam-detector](https://github.com/SKT799/ai-spam-detector)

---

*Built with ❤️ and lots of ☕*
