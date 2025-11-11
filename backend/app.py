from flask import Flask, request, jsonify, send_from_directory
import os
from huggingface_hub import hf_hub_download
import pickle
import re

# Get the parent directory (project root)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRONTEND_BUILD = os.path.join(BASE_DIR, 'frontend', 'build')

app = Flask(__name__, static_folder=FRONTEND_BUILD, static_url_path='')

# Model config
MODEL_REPO = "satyam2025/spam-email-classifier"
HF_TOKEN = os.environ.get("HF_TOKEN", None)  # Set HF_TOKEN environment variable if repo is private

print("=" * 70)
print("Loading Spam Classifier Model from Hugging Face")
print("=" * 70)
print(f"Repository: {MODEL_REPO}")
print()

try:
    # Download the sklearn model and TF-IDF vectorizer from HuggingFace
    model_path = hf_hub_download(repo_id=MODEL_REPO, filename="spam_classifier_model.pkl", token=HF_TOKEN)
    vectorizer_path = hf_hub_download(repo_id=MODEL_REPO, filename="tfidf_vectorizer.pkl", token=HF_TOKEN)
    
    # Load the model and vectorizer using pickle
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    with open(vectorizer_path, 'rb') as f:
        vectorizer = pickle.load(f)
    
    print("✓ Model loaded (spam_classifier_model.pkl)")
    print("✓ Vectorizer loaded (tfidf_vectorizer.pkl)")
    print()
    print("=" * 70)
    print("Model ready for predictions!")
    print("=" * 70)
    print()
except Exception as e:
    import traceback
    print(f"Error loading model: {e}")
    traceback.print_exc()
    print()
    print("Note: Make sure spam_classifier_model.pkl and tfidf_vectorizer.pkl exist in your HuggingFace repo")
    exit(1)

def clean_email_text(text):
    """Clean and preprocess email text"""
    text = text.lower()
    text = re.sub(r'\S*@\S*\s?', '', text)
    text = re.sub(r'http\S+|www\.\S+|https\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z\s!?.]', ' ', text)
    text = ' '.join(text.split())
    return text

def predict_spam(text, threshold=0.8):
    """Predict if email is spam using sklearn model with detailed calculations"""
    # Step 1: Clean the text
    cleaned_text = clean_email_text(text)
    
    # Step 2: Transform text using the vectorizer
    text_vectorized = vectorizer.transform([cleaned_text])
    
    # Step 3: Get probability predictions
    probabilities = model.predict_proba(text_vectorized)[0]
    ham_prob = probabilities[0]  # probability of ham (not spam)
    spam_prob = probabilities[1]  # probability of spam
    
    # Step 4: Determine classification based on threshold
    is_spam = spam_prob >= threshold
    classification = 'SPAM' if is_spam else 'HAM'
    
    # Step 5: Get feature information
    feature_names = vectorizer.get_feature_names_out()
    feature_values = text_vectorized.toarray()[0]
    
    # Get top features (words with highest TF-IDF scores)
    top_feature_indices = feature_values.argsort()[-10:][::-1]
    top_features = [
        {
            'word': feature_names[idx],
            'tfidf_score': float(feature_values[idx])
        }
        for idx in top_feature_indices if feature_values[idx] > 0
    ]
    
    return {
        'spam_probability': float(spam_prob),
        'ham_probability': float(ham_prob),
        'is_spam': bool(is_spam),
        'classification': classification,
        'threshold': threshold,
        'cleaned_text': cleaned_text,
        'original_length': len(text),
        'cleaned_length': len(cleaned_text),
        'top_features': top_features[:5],  # Return top 5 features
        'total_features': int(text_vectorized.nnz)  # Number of non-zero features
    }

@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({'status': 'ok', 'model': MODEL_REPO})

@app.route('/api/classify', methods=['POST'])
def classify():
    """Classify email as spam or not"""
    try:
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({'error': 'No text provided'}), 400
        
        email_text = data['text'].strip()
        
        if not email_text:
            return jsonify({'error': 'Empty text'}), 400
        
        # Get threshold from request, default to 0.8
        threshold = data.get('threshold', 0.8)
        
        print(f"\nAnalyzing email ({len(email_text)} chars)...")
        
        # Get detailed prediction results
        result = predict_spam(email_text, threshold=threshold)
        
        print(f"Result: {result['spam_probability']:.2%} spam probability")
        
        return jsonify({
            'success': True,
            'spam_probability': result['spam_probability'],
            'ham_probability': result['ham_probability'],
            'is_spam': result['is_spam'],
            'classification': result['classification'],
            'threshold': result['threshold'],
            'text_stats': {
                'original_length': result['original_length'],
                'cleaned_length': result['cleaned_length'],
                'total_features': result['total_features']
            },
            'top_features': result['top_features'],
            'cleaned_text': result['cleaned_text'][:200]  # Return first 200 chars of cleaned text
        })
    
    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({'error': f'Classification failed: {str(e)}'}), 500

@app.route('/')
def serve_frontend():
    """Serve the React frontend"""
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/<path:path>')
def serve_static(path):
    """Serve static files"""
    if os.path.exists(os.path.join(app.static_folder, path)):
        return send_from_directory(app.static_folder, path)
    else:
        return send_from_directory(app.static_folder, 'index.html')

if __name__ == '__main__':
    print("\n" + "=" * 70)
    print("Starting Spam Detector Server...")
    print("=" * 70)
    print()
    print("🌐 Server: http://localhost:5000")
    print("📱 Access from your browser at the URL above")
    print()
    print("API Endpoints:")
    print("  • GET  /api/health   - Health check")
    print("  • POST /api/classify - Classify email")
    print()
    print("⏸️  Press Ctrl+C to stop the server")
    print("=" * 70)
    print()
    
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host='0.0.0.0', port=port)
