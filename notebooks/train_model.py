"""
ToxiClean - Dataset Exploration & Model Training Script
=========================================================
This script shows how to work with the Jigsaw Toxic Comment
Classification dataset and train the toxicity classifier.

Dataset: https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge

How to get the dataset:
1. Go to the Kaggle link above
2. Download train.csv
3. Place it in the 'data/' folder as 'data/train.csv'
4. Run this script

If you don't have the dataset, this script creates a demo dataset
to show you how everything works.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from modules.classifier import ToxicityClassifier, TOXICITY_LABELS, print_evaluation_report
from modules.preprocessor import preprocess_batch

# ─── Configuration ────────────────────────────────────────────────────────────
DATA_PATH = 'data/train.csv'
MODEL_SAVE_PATH = 'models/toxiclean_model.pkl'
SAMPLE_SIZE = 50000  # Use a subset for faster training (set to None for full dataset)


# ─── Step 1: Load or Create Dataset ──────────────────────────────────────────
def load_dataset():
    """
    Load the Jigsaw dataset or create a synthetic demo dataset.
    
    The Jigsaw dataset format:
    - id: unique identifier
    - comment_text: the actual text comment
    - toxic: 0 or 1 (general toxicity)
    - severe_toxic: 0 or 1 (very offensive)
    - obscene: 0 or 1 (vulgar language)
    - threat: 0 or 1 (violent threats)
    - insult: 0 or 1 (personal insults)
    - identity_hate: 0 or 1 (hate based on identity)
    
    Note: A comment can have multiple labels simultaneously!
    For example, a comment can be both toxic AND an insult.
    """
    
    if os.path.exists(DATA_PATH):
        print(f"📂 Loading dataset from {DATA_PATH}...")
        df = pd.read_csv(DATA_PATH)
        
        if SAMPLE_SIZE and len(df) > SAMPLE_SIZE:
            print(f"   Using {SAMPLE_SIZE:,} samples (full dataset has {len(df):,})")
            # Keep balanced sample
            toxic = df[df['toxic'] == 1].sample(min(SAMPLE_SIZE//4, len(df[df['toxic']==1])))
            clean = df[df['toxic'] == 0].sample(SAMPLE_SIZE - len(toxic))
            df = pd.concat([toxic, clean]).sample(frac=1, random_state=42)
    else:
        print("⚠️  Dataset not found. Creating synthetic demo dataset...")
        print(f"   (Download real data from Kaggle: jigsaw-toxic-comment-classification-challenge)")
        df = create_demo_dataset()
    
    print(f"\n📊 Dataset shape: {df.shape}")
    print(f"   Columns: {list(df.columns)}")
    return df


def create_demo_dataset(n_samples: int = 2000) -> pd.DataFrame:
    """
    Create a synthetic dataset for demonstration.
    In a real project, use the actual Jigsaw dataset!
    """
    
    # Templates for generating synthetic toxic comments
    toxic_templates = [
        "You are a complete {insult1} and I {hate} you!",
        "What a {insult1} {insult2} you are!",
        "I {hate} people like you, you're {adj1}.",
        "You {insult1}! Just {command}!",
        "This is {adj1} and {adj2} work!",
        "You're so {adj1}, it's {adj2}.",
    ]
    
    clean_templates = [
        "Have a wonderful day!",
        "Thank you for your help with this.",
        "I strongly disagree with your approach.",
        "Could you please explain your reasoning?",
        "I think there might be a better way.",
        "Great work on the project!",
        "I'm not sure I understand your point.",
        "Let's discuss this further.",
    ]
    
    insults = ['idiot', 'moron', 'fool', 'loser', 'clown']
    hate_verbs = ['hate', 'despise', 'loathe']
    adj_negative = ['stupid', 'pathetic', 'worthless', 'useless', 'horrible']
    commands = ['go away', 'shut up', 'leave', 'stop']
    
    data = []
    
    # Generate toxic examples
    for i in range(n_samples // 2):
        template = toxic_templates[i % len(toxic_templates)]
        text = template.format(
            insult1=insults[i % len(insults)],
            insult2=insults[(i+1) % len(insults)],
            hate=hate_verbs[i % len(hate_verbs)],
            adj1=adj_negative[i % len(adj_negative)],
            adj2=adj_negative[(i+1) % len(adj_negative)],
            command=commands[i % len(commands)]
        )
        
        is_insult = any(w in text for w in insults)
        is_hate = any(w in text for w in hate_verbs)
        is_threat = 'destroy' in text or 'kill' in text or 'hurt' in text
        
        data.append({
            'comment_text': text,
            'toxic': 1,
            'severe_toxic': 1 if i % 10 == 0 else 0,
            'obscene': 1 if i % 3 == 0 else 0,
            'threat': int(is_threat),
            'insult': int(is_insult),
            'identity_hate': 1 if i % 8 == 0 else 0
        })
    
    # Generate clean examples
    for i in range(n_samples // 2):
        text = clean_templates[i % len(clean_templates)]
        data.append({
            'comment_text': text,
            'toxic': 0,
            'severe_toxic': 0,
            'obscene': 0,
            'threat': 0,
            'insult': 0,
            'identity_hate': 0
        })
    
    df = pd.DataFrame(data)
    return df.sample(frac=1, random_state=42).reset_index(drop=True)


# ─── Step 2: Exploratory Data Analysis ──────────────────────────────────────
def explore_dataset(df: pd.DataFrame):
    """
    Analyze the dataset to understand its characteristics.
    This is crucial before training any ML model.
    """
    
    print("\n" + "=" * 60)
    print("📊 DATASET EXPLORATION")
    print("=" * 60)
    
    # Basic stats
    print(f"\n📈 Total comments: {len(df):,}")
    print(f"   Average comment length: {df['comment_text'].str.len().mean():.0f} characters")
    
    # Label distribution
    print("\n📊 Label Distribution:")
    print("-" * 45)
    available_labels = [l for l in TOXICITY_LABELS if l in df.columns]
    
    for label in available_labels:
        count = df[label].sum()
        pct = count / len(df) * 100
        bar = '█' * int(pct / 2) + '░' * (50 - int(pct / 2))
        print(f"  {label:<15} {count:>6,} ({pct:>5.1f}%) {bar[:20]}")
    
    # Multi-label analysis
    if available_labels:
        df['label_count'] = df[available_labels].sum(axis=1)
        print(f"\n🏷️  Comments with multiple labels: {(df['label_count'] > 1).sum():,}")
        print(f"   Max labels per comment: {df['label_count'].max()}")
        
        # Clean vs toxic
        clean_count = (df['label_count'] == 0).sum()
        toxic_count = (df['label_count'] > 0).sum()
        print(f"\n   Clean comments: {clean_count:,} ({clean_count/len(df):.1%})")
        print(f"   Toxic comments: {toxic_count:,} ({toxic_count/len(df):.1%})")
    
    return df


# ─── Step 3: Train the Model ─────────────────────────────────────────────────
def train_model(df: pd.DataFrame) -> ToxicityClassifier:
    """Train the toxicity classifier on the dataset."""
    
    print("\n" + "=" * 60)
    print("🤖 MODEL TRAINING")
    print("=" * 60)
    
    # Initialize classifier
    clf = ToxicityClassifier(model_type='logistic_regression')
    
    # Train and get evaluation metrics
    metrics = clf.train(df)
    
    # Print detailed report
    print_evaluation_report(metrics)
    
    # Save model for later use
    os.makedirs('models', exist_ok=True)
    clf.save(MODEL_SAVE_PATH)
    
    return clf


# ─── Step 4: Test Predictions ────────────────────────────────────────────────
def test_predictions(clf: ToxicityClassifier):
    """Test the trained classifier on new examples."""
    
    print("\n" + "=" * 60)
    print("🔍 PREDICTION TESTS")
    print("=" * 60)
    
    test_cases = [
        "You are such a stupid idiot!",
        "I hate everything about you.",
        "I will find and hurt you badly.",
        "Have a wonderful day, everyone!",
        "Thank you for your contribution.",
        "I strongly disagree with your position.",
        "This is the most pathetic thing I've ever seen.",
        "Great work! Keep it up!",
    ]
    
    print(f"\n{'Text':<45} {'Toxic':>6} {'Conf':>6} {'Types'}")
    print("-" * 80)
    
    for text in test_cases:
        result = clf.predict(text)
        status = "🔴 YES" if result['is_toxic'] else "🟢 NO"
        types = ', '.join(result['detected_types'][:2]) or '-'
        
        print(f"{text[:44]:<45} {status:>6} {result['confidence']:>5.0%} {types}")


# ─── Step 5: Feature Analysis ────────────────────────────────────────────────
def analyze_features(clf: ToxicityClassifier):
    """Show which words the model learned are most toxic."""
    
    print("\n" + "=" * 60)
    print("🔬 TOP TOXIC FEATURES (Words Most Associated with Toxicity)")
    print("=" * 60)
    
    try:
        top_features = clf.get_feature_importance('toxic', top_n=20)
        
        print("\n🔴 Top words/phrases indicating TOXIC content:")
        print("-" * 40)
        for _, row in top_features.head(10).iterrows():
            bar = '█' * int(row['importance'] * 5)
            print(f"  {row['word']:<20} {bar} ({row['importance']:.3f})")
        
    except Exception as e:
        print(f"Feature importance not available: {e}")


# ─── Main Runner ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 65)
    print("🧹 ToxiClean — Model Training Pipeline")
    print("=" * 65)
    
    # Step 1: Load data
    df = load_dataset()
    
    # Step 2: Explore data
    df = explore_dataset(df)
    
    # Step 3: Train model
    clf = train_model(df)
    
    # Step 4: Test predictions
    test_predictions(clf)
    
    # Step 5: Feature analysis
    analyze_features(clf)
    
    print("\n✅ Training pipeline complete!")
    print(f"💾 Model saved to: {MODEL_SAVE_PATH}")
    print("\n🚀 To launch the web app:")
    print("   streamlit run app/streamlit_app.py")
