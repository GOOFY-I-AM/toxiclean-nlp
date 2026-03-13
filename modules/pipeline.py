"""
ToxiClean - Main Pipeline
==========================
The complete end-to-end pipeline that:
1. Takes raw text input
2. Detects if it's toxic
3. Identifies toxic words
4. Generates a neutral alternative
5. Returns detailed results

This is the main entry point for the ToxiClean system.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from modules.preprocessor import preprocess_text
from modules.word_detector import ToxicWordDetector
from modules.neutralizer import TextNeutralizer


class ToxiCleanPipeline:
    """
    Complete ToxiClean pipeline for toxic speech neutralization.
    
    This class ties together all components:
    - Preprocessing: Clean the input text
    - Detection: Check if text is toxic (rule-based when model isn't available)
    - Word Detection: Find which words are toxic
    - Neutralization: Convert to neutral text
    
    Example Usage:
    --------------
    >>> pipeline = ToxiCleanPipeline()
    >>> result = pipeline.analyze("You are such a stupid idiot!")
    >>> print(result['neutral_text'])
    "Your behavior seems quite mistaken!"
    >>> print(result['is_toxic'])
    True
    """
    
    def __init__(self, classifier=None, neutralization_strategy: str = 'combined'):
        """
        Initialize the pipeline.
        
        Parameters:
        -----------
        classifier : ToxicityClassifier or None
            Pre-trained ML classifier. If None, uses rule-based detection.
        neutralization_strategy : str
            How to neutralize: 'word_replacement', 'rule_based', 'combined', 'transformer'
        """
        self.classifier = classifier
        self.word_detector = ToxicWordDetector()
        self.neutralizer = TextNeutralizer(strategy=neutralization_strategy)
    
    def analyze(self, text: str) -> dict:
        """
        Full pipeline: analyze and neutralize a text input.
        
        Parameters:
        -----------
        text : str
            Raw user input text
            
        Returns:
        --------
        dict with complete analysis:
        {
            'original_text': str,       # What the user typed
            'clean_text': str,          # After preprocessing
            'is_toxic': bool,           # Is the text toxic?
            'confidence': float,        # How confident (0-1)
            'toxicity_types': list,     # Which types were detected
            'toxic_words': list,        # Specific toxic words found
            'highlighted_text': str,    # HTML with highlights
            'neutral_text': str,        # Neutralized version
            'changes_made': list,       # What was changed
            'intensity_score': float,   # How toxic (0-1)
        }
        """
        if not text or not text.strip():
            return self._empty_result(text)
        
        # ── Stage 1: Preprocessing ──────────────────────────────────────
        clean_text = preprocess_text(text, for_model=True)
        
        # ── Stage 2: Toxicity Detection ─────────────────────────────────
        if self.classifier and self.classifier.is_trained:
            # Use ML classifier if available
            detection = self.classifier.predict(text)
            is_toxic = detection['is_toxic']
            confidence = detection['confidence']
            toxicity_types = detection['detected_types']
        else:
            # Fallback: use word dictionary for detection
            word_analysis = self.word_detector.detect(text)
            is_toxic = word_analysis['toxic_count'] > 0
            confidence = min(1.0, word_analysis['intensity_score'] * 2)
            toxicity_types = word_analysis['categories_found']
        
        # ── Stage 3: Toxic Word Detection ───────────────────────────────
        word_analysis = self.word_detector.detect(text)
        
        # ── Stage 4: Neutralization ─────────────────────────────────────
        if is_toxic:
            neutralization = self.neutralizer.neutralize(
                text, 
                toxic_words=word_analysis['toxic_words']
            )
            neutral_text = neutralization['neutral_text']
            changes = neutralization['changes_made']
        else:
            # Text is already clean - no changes needed
            neutral_text = text
            changes = []
        
        return {
            'original_text': text,
            'clean_text': clean_text,
            'is_toxic': is_toxic,
            'confidence': round(confidence, 4),
            'toxicity_types': toxicity_types,
            'toxic_words': word_analysis['toxic_words'],
            'highlighted_text': word_analysis['highlighted_text'],
            'neutral_text': neutral_text,
            'changes_made': changes,
            'intensity_score': word_analysis['intensity_score'],
            'toxic_count': word_analysis['toxic_count'],
        }
    
    def _empty_result(self, text: str) -> dict:
        """Return empty/default result for empty input."""
        return {
            'original_text': text or '',
            'clean_text': '',
            'is_toxic': False,
            'confidence': 0.0,
            'toxicity_types': [],
            'toxic_words': [],
            'highlighted_text': text or '',
            'neutral_text': text or '',
            'changes_made': [],
            'intensity_score': 0.0,
            'toxic_count': 0,
        }


def demonstrate_pipeline():
    """Run a complete demonstration of the ToxiClean pipeline."""
    
    pipeline = ToxiCleanPipeline()
    
    # Example input/output pairs
    examples = [
        # (input_text, description)
        ("You are such a stupid idiot, I hate you!",
         "Multiple insults"),
        
        ("I will destroy you if you don't listen!",
         "Threat-based toxicity"),
        
        ("This is absolutely pathetic and worthless work!",
         "Negative evaluation"),
        
        ("Shut up and get lost, you moron!",
         "Multiple aggressive phrases"),
        
        ("I hate people like you, you're so disgusting!",
         "Hate speech"),
        
        ("Have a wonderful day! Your work is excellent.",
         "Clean/positive text"),
        
        ("I strongly disagree with your approach, let's discuss.",
         "Assertive but respectful"),
        
        ("You dumb fool, why would you do that?!",
         "Rhetorical insult"),
    ]
    
    print("=" * 75)
    print("🧹 ToxiClean: Toxic Speech Neutralizer — Pipeline Demo")
    print("=" * 75)
    
    for text, description in examples:
        print(f"\n📌 Example: {description}")
        print("─" * 75)
        
        result = pipeline.analyze(text)
        
        status = "🔴 TOXIC" if result['is_toxic'] else "🟢 CLEAN"
        print(f"{'Status':<16}: {status}")
        print(f"{'Original':<16}: {result['original_text']}")
        
        if result['toxic_words']:
            words = [w['word'] for w in result['toxic_words']]
            print(f"{'Toxic Words':<16}: {', '.join(words)}")
        
        if result['toxicity_types']:
            print(f"{'Types':<16}: {', '.join(result['toxicity_types'])}")
        
        if result['is_toxic']:
            print(f"{'Neutralized':<16}: {result['neutral_text']}")
            print(f"{'Intensity':<16}: {result['intensity_score']:.0%}")
    
    print("\n" + "=" * 75)
    print("✅ Pipeline demonstration complete!")
    print("=" * 75)


if __name__ == "__main__":
    demonstrate_pipeline()
