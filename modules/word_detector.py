"""
ToxiClean - Toxic Word Detector
================================
Identifies WHICH specific words in a sentence are toxic.
This is the "explainability" layer — it shows users exactly
what parts of their text triggered the toxicity flag.

Methods:
1. Dictionary-based: Uses a curated toxic word list
2. TF-IDF weight-based: Uses model feature importance scores
"""

import re
from typing import List, Dict, Tuple


# ─── Toxic Word Dictionary ────────────────────────────────────────────────────
# A curated list of words/phrases organized by toxicity category.
# In a production system, this would be much larger and dynamically maintained.
# Words here are intentionally mild examples for demonstration.
TOXIC_WORD_DICTIONARY = {
    'insult': [
        'stupid', 'idiot', 'moron', 'dumb', 'fool', 'loser', 'worthless',
        'pathetic', 'useless', 'brainless', 'imbecile', 'dimwit', 'buffoon',
        'clown', 'ignorant', 'incompetent', 'failure', 'retard', 'dumbass',
        'jackass', 'asshole', 'jerk', 'bastard', 'bitch', 'idiot',
        'scumbag', 'lowlife', 'piece of shit', 'pos'
    ],
    'threat': [
        'kill', 'hurt', 'destroy', 'attack', 'harm', 'murder', 'threaten',
        'punish', 'eliminate', 'end you', 'find you', 'come for you',
        'beat you', 'smash you', 'crush you', 'finish you'
    ],
    'hate': [
        'hate', 'despise', 'detest', 'loathe', 'abhor', 'disgusting',
        'repulsive', 'vile', 'horrible', 'terrible', 'awful', 'nasty'
    ],
    'obscene': [
        'fuck', 'fucking', 'fucked', 'fucker', 'fuck off', 'wtf',
        'shit', 'bullshit', 'shitty', 'crap', 'ass', 'arse',
        'damn', 'hell', 'cunt', 'dick', 'cock', 'pussy',
        'piss', 'pissed', 'bitch', 'whore', 'slut', 'bastard'
    ],
    'aggressive': [
        'shut up', 'go away', 'get lost', 'leave me alone', 'stay away',
        'go to hell', 'drop dead', 'get out', 'back off', 'piss off',
        'screw you', 'screw off', 'kiss my ass', 'up yours'
    ]
}

# Flatten dictionary to a quick lookup set
ALL_TOXIC_WORDS = {
    word: category 
    for category, words in TOXIC_WORD_DICTIONARY.items() 
    for word in words
}

# Intensity modifiers that make words more toxic
INTENSITY_MODIFIERS = {'very', 'extremely', 'so', 'completely', 'totally', 'utterly', 'absolutely'}

# Negation words that can flip meaning
NEGATION_WORDS = {'not', "n't", 'never', 'no', 'neither', 'nor'}


class ToxicWordDetector:
    """
    Detects and highlights toxic words within a sentence.
    
    Example Usage:
    --------------
    >>> detector = ToxicWordDetector()
    >>> result = detector.detect("You are a stupid idiot!")
    >>> print(result['toxic_words'])
    [{'word': 'stupid', 'category': 'insult', 'position': 3},
     {'word': 'idiot', 'category': 'insult', 'position': 4}]
    """
    
    def __init__(self, custom_words: Dict[str, List[str]] = None):
        """
        Initialize the detector with the default dictionary.
        
        Parameters:
        -----------
        custom_words : dict
            Optional additional words to add to the dictionary.
            Format: {'category': ['word1', 'word2', ...]}
        """
        self.word_dict = dict(ALL_TOXIC_WORDS)
        self.toxic_phrases = []  # Multi-word phrases to check
        
        # Load multi-word toxic phrases
        for category, words in TOXIC_WORD_DICTIONARY.items():
            for word in words:
                if ' ' in word:  # Multi-word phrase
                    self.toxic_phrases.append((word, category))
        
        # Add custom words if provided
        if custom_words:
            for category, words in custom_words.items():
                for word in words:
                    self.word_dict[word.lower()] = category
    
    def detect(self, text: str) -> Dict:
        """
        Analyze text and identify all toxic words and their positions.
        
        Parameters:
        -----------
        text : str
            Input text to analyze
            
        Returns:
        --------
        dict with structure:
        {
            'original_text': str,
            'toxic_words': list of dicts with word info,
            'toxic_count': int,
            'categories_found': list,
            'highlighted_text': str (HTML with highlights),
            'intensity_score': float (0-1 toxicity intensity)
        }
        """
        tokens = text.lower().split()
        toxic_words = []
        toxic_positions = set()
        
        # Step 1: Check for multi-word phrases first
        text_lower = text.lower()
        for phrase, category in self.toxic_phrases:
            if phrase in text_lower:
                start_idx = text_lower.index(phrase)
                toxic_words.append({
                    'word': phrase,
                    'category': category,
                    'position': -1,  # Phrase, not single token
                    'is_phrase': True,
                    'has_intensifier': False,
                    'is_negated': False
                })
        
        # Step 2: Check individual tokens
        for i, token in enumerate(tokens):
            # Clean token (remove punctuation)
            clean_token = re.sub(r'[^\w]', '', token)
            
            if clean_token in self.word_dict:
                category = self.word_dict[clean_token]
                
                # Check for intensity modifier before the word
                has_intensifier = (
                    i > 0 and 
                    re.sub(r'[^\w]', '', tokens[i-1]) in INTENSITY_MODIFIERS
                )
                
                # Check for negation (word before or two before)
                is_negated = False
                for j in range(max(0, i-3), i):
                    if re.sub(r'[^\w]', '', tokens[j]) in NEGATION_WORDS:
                        is_negated = True
                        break
                
                toxic_words.append({
                    'word': clean_token,
                    'category': category,
                    'position': i,
                    'is_phrase': False,
                    'has_intensifier': has_intensifier,
                    'is_negated': is_negated
                })
                toxic_positions.add(i)
        
        # Step 3: Calculate intensity score
        intensity_score = self._calculate_intensity(toxic_words, len(tokens))
        
        # Step 4: Generate highlighted text (HTML)
        highlighted = self._generate_highlighted_text(text, toxic_words)
        
        # Step 5: Get unique categories
        categories = list(set(w['category'] for w in toxic_words))
        
        return {
            'original_text': text,
            'toxic_words': toxic_words,
            'toxic_count': len(toxic_words),
            'categories_found': categories,
            'highlighted_text': highlighted,
            'intensity_score': round(intensity_score, 3),
            'non_toxic_words': [
                tokens[i] for i in range(len(tokens)) 
                if i not in toxic_positions
            ]
        }
    
    def _calculate_intensity_score(self, toxic_words: list, total_tokens: int) -> float:
        """Calculate overall toxicity intensity (0-1 scale)."""
        return self._calculate_intensity(toxic_words, total_tokens)
    
    def _calculate_intensity(self, toxic_words: list, total_tokens: int) -> float:
        """
        Calculate a toxicity intensity score from 0 to 1.
        
        Factors:
        - Ratio of toxic words to total words
        - Presence of intensity modifiers
        - Threat category (highest weight)
        """
        if not toxic_words or total_tokens == 0:
            return 0.0
        
        base_score = len(toxic_words) / max(total_tokens, 1)
        
        # Boost for intensity modifiers
        intensifier_boost = sum(0.1 for w in toxic_words if w['has_intensifier'])
        
        # Boost for threats (most serious)
        threat_boost = sum(0.2 for w in toxic_words if w['category'] == 'threat')
        
        # Reduce for negated words
        negation_reduction = sum(0.15 for w in toxic_words if w['is_negated'])
        
        score = base_score + intensifier_boost + threat_boost - negation_reduction
        return min(1.0, max(0.0, score))
    
    def _generate_highlighted_text(self, text: str, toxic_words: list) -> str:
        """
        Generate HTML with toxic words highlighted in red.
        
        Returns an HTML string like:
        "You are a <mark style='color:red'>stupid</mark> <mark>idiot</mark>!"
        """
        result = text
        
        # Sort by word length (longer first) to avoid partial replacements
        sorted_toxic = sorted(
            [w['word'] for w in toxic_words if not w.get('is_phrase')],
            key=len, reverse=True
        )
        
        for word in sorted_toxic:
            # Case-insensitive replacement
            pattern = re.compile(re.escape(word), re.IGNORECASE)
            result = pattern.sub(
                f'<mark class="toxic-word" style="background-color:#ffcccc;'
                f'border-radius:3px;padding:1px 3px;">{word}</mark>',
                result
            )
        
        return result
    
    def get_toxic_word_list(self) -> List[str]:
        """Return all toxic words in the dictionary."""
        return list(self.word_dict.keys())
    
    def add_words(self, words: List[str], category: str = 'custom'):
        """Add new words to the toxic dictionary."""
        for word in words:
            self.word_dict[word.lower()] = category
        print(f"✅ Added {len(words)} words to category '{category}'")
    
    def get_category_summary(self) -> Dict[str, int]:
        """Get count of words per category."""
        from collections import Counter
        return dict(Counter(self.word_dict.values()))


# ─── Demo ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    detector = ToxicWordDetector()
    
    test_sentences = [
        "You are such a stupid moron!",
        "I will hurt you if you don't stop.",
        "This is absolutely pathetic and worthless.",
        "Have a great day, you're doing amazing!",
        "I hate everything about this, it's disgusting.",
    ]
    
    print("=" * 65)
    print("🔍 ToxiClean - Toxic Word Detection Demo")
    print("=" * 65)
    
    for sentence in test_sentences:
        print(f"\n📝 Input: {sentence}")
        result = detector.detect(sentence)
        
        if result['toxic_words']:
            words = [f"'{w['word']}' ({w['category']})" 
                    for w in result['toxic_words']]
            print(f"🔴 Toxic words found: {', '.join(words)}")
            print(f"   Intensity score: {result['intensity_score']:.1%}")
        else:
            print("🟢 No toxic words detected!")
    
    print("\n📊 Category Summary:")
    for cat, count in detector.get_category_summary().items():
        print(f"   {cat:<15}: {count} words")
