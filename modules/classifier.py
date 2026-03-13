"""
ToxiClean - Toxicity Detection Model
======================================
Trains and uses a machine learning classifier to detect whether
input text is toxic, and what type of toxicity it contains.

Supports:
- Logistic Regression (fast, interpretable)
- Naive Bayes (simple, works well for text)
- LSTM (deep learning, most accurate)
"""

import os
import numpy as np
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, 
    recall_score, f1_score, classification_report
)

# Import our preprocessing module
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from modules.preprocessor import preprocess_text, preprocess_batch


# ─── Label Columns (from Jigsaw dataset) ────────────────────────────────────
# Each comment can have multiple toxicity types simultaneously
TOXICITY_LABELS = [
    'toxic',          # General toxicity
    'severe_toxic',   # Extremely offensive content
    'obscene',        # Vulgar/obscene language
    'threat',         # Threats of violence
    'insult',         # Personal insults
    'identity_hate'   # Hate based on identity (race, religion, etc.)
]


# ─── Feature Engineering ────────────────────────────────────────────────────
def build_tfidf_vectorizer(max_features: int = 50000,
                            ngram_range: tuple = (1, 2)) -> TfidfVectorizer:
    """
    Build a TF-IDF vectorizer to convert text into numerical features.
    
    TF-IDF (Term Frequency-Inverse Document Frequency):
    - TF: How often a word appears in a document
    - IDF: How rare a word is across all documents
    - Toxic words that appear frequently in bad comments get high scores
    
    Parameters:
    -----------
    max_features : int
        Maximum number of unique words/phrases to track
    ngram_range : tuple
        (1,2) means use single words AND two-word phrases
        Example: "stupid idiot" is captured as a phrase
    
    Returns:
    --------
    TfidfVectorizer configured and ready to fit
    """
    return TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,       # Capture word pairs for context
        sublinear_tf=True,             # Apply log to term frequency
        strip_accents='unicode',       # Handle accented characters
        analyzer='word',               # Analyze at word level
        token_pattern=r'\w{2,}',      # Minimum 2-character tokens
        min_df=3,                      # Word must appear in at least 3 docs
        max_df=0.9                     # Ignore words in >90% of documents
    )


# ─── Model Training ─────────────────────────────────────────────────────────
class ToxicityClassifier:
    """
    Multi-label toxicity classifier that can detect 6 types of toxic content.
    
    Uses a pipeline: Text → TF-IDF Features → Logistic Regression
    
    Example Usage:
    --------------
    >>> clf = ToxicityClassifier()
    >>> clf.train(df)  # df has 'comment_text' and label columns
    >>> results = clf.predict("You are so stupid!")
    >>> print(results)
    {'is_toxic': True, 'toxic': 0.95, 'insult': 0.87, ...}
    """
    
    def __init__(self, model_type: str = 'logistic_regression'):
        """
        Initialize the classifier.
        
        Parameters:
        -----------
        model_type : str
            'logistic_regression' - Fast, good performance
            'naive_bayes'         - Very fast, decent performance
        """
        self.model_type = model_type
        self.models = {}        # One model per toxicity label
        self.vectorizer = None  # Shared TF-IDF vectorizer
        self.is_trained = False
        
    def _get_base_model(self):
        """Return the ML algorithm based on selected type."""
        if self.model_type == 'logistic_regression':
            return LogisticRegression(
                C=1.0,              # Regularization strength
                max_iter=200,       # Training iterations
                solver='lbfgs',     # Optimization algorithm
                class_weight='balanced'  # Handle imbalanced data
            )
        elif self.model_type == 'naive_bayes':
            return MultinomialNB(alpha=0.1)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def train(self, df: pd.DataFrame, 
              text_col: str = 'comment_text',
              test_size: float = 0.2) -> dict:
        """
        Train the toxicity classifier on labeled data.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Dataset with text column and toxicity label columns
        text_col : str
            Name of the column containing text
        test_size : float
            Fraction of data to use for evaluation
            
        Returns:
        --------
        dict : Evaluation metrics for each label
        """
        print("=" * 60)
        print("🤖 Training ToxiClean Toxicity Classifier")
        print("=" * 60)
        
        # Step 1: Preprocess all text
        print("\n📝 Step 1: Preprocessing text...")
        df['clean_text'] = preprocess_batch(df[text_col].fillna('').tolist())
        
        # Step 2: Split into train/test
        print("📊 Step 2: Splitting data into train/test sets...")
        train_df, test_df = train_test_split(
            df, test_size=test_size, random_state=42
        )
        print(f"   Training samples: {len(train_df):,}")
        print(f"   Testing samples:  {len(test_df):,}")
        
        # Step 3: Build TF-IDF features
        print("\n🔢 Step 3: Building TF-IDF features...")
        self.vectorizer = build_tfidf_vectorizer()
        X_train = self.vectorizer.fit_transform(train_df['clean_text'])
        X_test = self.vectorizer.transform(test_df['clean_text'])
        print(f"   Feature matrix shape: {X_train.shape}")
        
        # Step 4: Train one model per toxicity label
        print(f"\n🏋️ Step 4: Training {self.model_type} models...")
        print("   (One model per toxicity category)")
        
        metrics = {}
        available_labels = [l for l in TOXICITY_LABELS if l in df.columns]
        
        for label in available_labels:
            y_train = train_df[label].values
            y_test = test_df[label].values
            
            # Train model
            model = self._get_base_model()
            model.fit(X_train, y_train)
            
            # Evaluate
            y_pred = model.predict(X_test)
            
            metrics[label] = {
                'accuracy':  accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, zero_division=0),
                'recall':    recall_score(y_test, y_pred, zero_division=0),
                'f1':        f1_score(y_test, y_pred, zero_division=0)
            }
            
            self.models[label] = model
            print(f"   ✅ {label:<15} | F1: {metrics[label]['f1']:.3f} | "
                  f"Acc: {metrics[label]['accuracy']:.3f}")
        
        self.is_trained = True
        print(f"\n✨ Training complete! {len(self.models)} models trained.")
        return metrics
    
    def predict(self, text: str, threshold: float = 0.3) -> dict:
        """
        Predict toxicity for a single text input.
        
        Parameters:
        -----------
        text : str
            Raw input text to analyze
        threshold : float
            Probability cutoff for positive classification (0.0 - 1.0)
            Lower = more sensitive to toxicity
            
        Returns:
        --------
        dict with structure:
        {
            'is_toxic': bool,          # Overall toxicity verdict
            'confidence': float,       # Highest toxicity probability
            'labels': {
                'toxic': float,        # Probability for each category
                'insult': float,
                ...
            },
            'detected_types': list     # List of detected toxicity types
        }
        """
        if not self.is_trained:
            raise RuntimeError("Model not trained! Call train() first.")
        
        # Preprocess the input text
        clean = preprocess_text(text)
        
        # Transform to TF-IDF features
        features = self.vectorizer.transform([clean])
        
        # Get predictions for each toxicity type
        label_probs = {}
        detected_types = []
        
        for label, model in self.models.items():
            try:
                # Get probability of being toxic (class = 1)
                prob = model.predict_proba(features)[0][1]
                label_probs[label] = round(float(prob), 4)
                
                if prob >= threshold:
                    detected_types.append(label)
            except Exception:
                label_probs[label] = 0.0
        
        # Overall toxicity = any label exceeds threshold
        is_toxic = len(detected_types) > 0
        confidence = max(label_probs.values()) if label_probs else 0.0
        
        return {
            'is_toxic': is_toxic,
            'confidence': round(confidence, 4),
            'labels': label_probs,
            'detected_types': detected_types,
            'clean_text': clean
        }
    
    def predict_batch(self, texts: list, **kwargs) -> list:
        """Predict toxicity for multiple texts."""
        return [self.predict(text, **kwargs) for text in texts]
    
    def save(self, path: str = 'models/toxiclean_model.pkl'):
        """Save the trained model to disk."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump({
            'models': self.models,
            'vectorizer': self.vectorizer,
            'model_type': self.model_type,
            'labels': TOXICITY_LABELS
        }, path)
        print(f"💾 Model saved to: {path}")
    
    @classmethod
    def load(cls, path: str = 'models/toxiclean_model.pkl'):
        """Load a pre-trained model from disk."""
        data = joblib.load(path)
        clf = cls(model_type=data['model_type'])
        clf.models = data['models']
        clf.vectorizer = data['vectorizer']
        clf.is_trained = True
        print(f"✅ Model loaded from: {path}")
        return clf
    
    def get_feature_importance(self, label: str = 'toxic', 
                               top_n: int = 20) -> pd.DataFrame:
        """
        Get the most important words/features for detecting a specific
        toxicity type. Useful for understanding what the model learned.
        
        Returns:
        --------
        DataFrame with columns: ['word', 'importance']
        """
        if label not in self.models:
            raise ValueError(f"Label '{label}' not found. Choose from: {list(self.models.keys())}")
        
        model = self.models[label]
        feature_names = self.vectorizer.get_feature_names_out()
        
        # For Logistic Regression: use coefficients as importance
        if hasattr(model, 'coef_'):
            importance = model.coef_[0]
        else:
            raise ValueError("Feature importance not available for this model type")
        
        # Get top positive features (most indicative of toxicity)
        top_idx = np.argsort(importance)[-top_n:][::-1]
        
        return pd.DataFrame({
            'word': [feature_names[i] for i in top_idx],
            'importance': [importance[i] for i in top_idx]
        })


# ─── Evaluation Helper ───────────────────────────────────────────────────────
def print_evaluation_report(metrics: dict):
    """Print a formatted evaluation report for all labels."""
    print("\n" + "=" * 65)
    print("📊 EVALUATION REPORT")
    print("=" * 65)
    print(f"{'Label':<18} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print("-" * 65)
    
    for label, m in metrics.items():
        print(f"{label:<18} {m['accuracy']:>10.3f} {m['precision']:>10.3f} "
              f"{m['recall']:>10.3f} {m['f1']:>10.3f}")
    
    # Average
    avg = {k: np.mean([m[k] for m in metrics.values()]) for k in ['accuracy', 'precision', 'recall', 'f1']}
    print("-" * 65)
    print(f"{'AVERAGE':<18} {avg['accuracy']:>10.3f} {avg['precision']:>10.3f} "
          f"{avg['recall']:>10.3f} {avg['f1']:>10.3f}")
    print("=" * 65)


# ─── Demo without real dataset ───────────────────────────────────────────────
if __name__ == "__main__":
    # Create a small synthetic dataset for demonstration
    sample_data = pd.DataFrame({
        'comment_text': [
            "You are a complete idiot!",
            "I will find you and hurt you badly",
            "This is such an obscene and disgusting comment",
            "Have a wonderful day everyone!",
            "Thank you for your help",
            "The weather is nice today",
            "You stupid moron, go away",
            "Great job on the project!",
            "I hate people like you so much",
            "This was very helpful, thanks!"
        ] * 50,  # Repeat to make a larger demo dataset
        'toxic':         [1,1,1,0,0,0,1,0,1,0] * 50,
        'severe_toxic':  [0,1,0,0,0,0,0,0,0,0] * 50,
        'obscene':       [0,0,1,0,0,0,0,0,0,0] * 50,
        'threat':        [0,1,0,0,0,0,0,0,0,0] * 50,
        'insult':        [1,0,0,0,0,0,1,0,0,0] * 50,
        'identity_hate': [0,0,0,0,0,0,0,0,1,0] * 50,
    })
    
    # Train classifier
    clf = ToxicityClassifier(model_type='logistic_regression')
    metrics = clf.train(sample_data)
    
    # Print report
    print_evaluation_report(metrics)
    
    # Test predictions
    test_texts = [
        "You are such a stupid idiot!",
        "I hope you have a great day!",
        "I will destroy everything you love.",
    ]
    
    print("\n🔍 PREDICTION EXAMPLES:")
    print("-" * 50)
    for text in test_texts:
        result = clf.predict(text)
        emoji = "🔴" if result['is_toxic'] else "🟢"
        print(f"\n{emoji} Text: \"{text}\"")
        print(f"   Toxic: {result['is_toxic']} | Confidence: {result['confidence']:.1%}")
        if result['detected_types']:
            print(f"   Types: {', '.join(result['detected_types'])}")
