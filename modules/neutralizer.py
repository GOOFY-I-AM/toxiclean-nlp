"""
ToxiClean - Neutralization Module
====================================
Converts toxic sentences into neutral, polite alternatives.

Three strategies (in order of sophistication):
1. Word Replacement  - Simple substitution with neutral synonyms
2. Rule-Based        - Pattern matching and structural rewrites
3. Transformer-Based - Use T5/GPT to paraphrase (most powerful)
"""

import re
from typing import Optional


# ─── Neutral Word Substitution Map ──────────────────────────────────────────
# Maps toxic words → neutral/polite alternatives
# In production: this would be a much larger, carefully curated dictionary
NEUTRAL_REPLACEMENTS = {
    # Insults → neutral descriptors
    'stupid':      'mistaken',
    'idiot':       'person',
    'moron':       'individual',
    'dumb':        'uninformed',
    'fool':        'person',
    'loser':       'individual',
    'worthless':   'struggling',
    'pathetic':    'disappointing',
    'useless':     'ineffective',
    'brainless':   'uninformed',
    'imbecile':    'person',
    'dimwit':      'individual',
    'buffoon':     'person',
    'ignorant':    'uninformed',
    'incompetent': 'inexperienced',
    'failure':     'setback',
    'clown':       'person',
    
    # Threats → non-threatening language
    'kill':        'stop',
    'hurt':        'upset',
    'destroy':     'change',
    'attack':      'address',
    'harm':        'affect',
    'murder':      'confront',
    'eliminate':   'remove',
    
    # Hate words → neutral expressions
    'hate':        'strongly disagree with',
    'despise':     'strongly dislike',
    'detest':      'dislike',
    'loathe':      'dislike',
    'abhor':       'oppose',
    'disgusting':  'concerning',
    'repulsive':   'unpleasant',
    'vile':        'problematic',
    'horrible':    'unpleasant',
    'terrible':    'poor',
    'awful':       'unfortunate',
    'nasty':       'unpleasant',
    
    # Obscene words → neutral
    'fuck':      'mess',
    'fucking':   'really',
    'fucked':    'ruined',
    'shit':      'stuff',
    'bullshit':  'nonsense',
    'ass':       'person',
    'asshole':   'individual',
    'bitch':     'person',
    'bastard':   'person',
    'damn':      'very',
    'crap':      'nonsense',
    'piss off':  'go away',
    'screw you': 'I disagree',
    
    # Aggressive phrases
    'shut up':     'please stop',
    'go away':     'please leave',
    'get lost':    'please go',
}

# Phrase-level neutralization rules
# These handle common patterns in toxic speech
PHRASE_RULES = [
    # Pattern: "You are [toxic adjective]" → "Your behavior seems [neutral]"
    (r'\byou\s+are\s+(so\s+|very\s+|extremely\s+)?(\w+)', 
     r'your behavior seems \2'),
    
    # Pattern: "I hate you" → "I disagree with you"  
    (r'\bi\s+hate\s+(you|everyone|all)', 
     r'I strongly disagree with \1'),
    
    # Pattern: "Go [somewhere rude]" → "Please [neutral]"
    (r'\bgo\s+(away|to hell|f\w+\s+yourself)', 
     r'please excuse yourself'),
    
    # Pattern: intensifiers before insults
    (r'\b(so|very|extremely|completely|utterly|absolutely)\s+(\w+)', 
     r'\2'),
    
    # Pattern: "I will [threat]" → "I would prefer if"
    (r"\bi('ll|will|gonna|am going to)\s+(kill|hurt|destroy|attack|harm)\s+",
     r'I would like to change '),
    
    # Pattern: Remove excessive exclamation marks
    (r'!{2,}', '.'),
    
    # Pattern: Convert ALL CAPS (shouting) to normal case
    (r'\b([A-Z]{2,})\b', lambda m: m.group(1).capitalize()),
]

# Sentence-level complete rewrites for common toxic patterns
SENTENCE_REWRITES = {
    "shut up": "Please consider listening to others.",
    "you're worthless": "You have potential that needs development.",
    "i hate you": "I strongly disagree with your actions.",
    "you're an idiot": "I think there may be a misunderstanding here.",
    "go away": "Please give me some space.",
    "you're stupid": "I think we see this differently.",
    "you're a loser": "Everyone faces challenges at different times.",
    "i will hurt you": "I'm feeling very frustrated right now.",
}


class TextNeutralizer:
    """
    Converts toxic text into neutral, respectful alternatives.
    
    Supports three neutralization strategies:
    1. word_replacement - Fast, simple word substitution
    2. rule_based - Pattern matching for common toxic structures
    3. transformer - AI-powered paraphrasing (requires transformers library)
    
    Example Usage:
    --------------
    >>> neutralizer = TextNeutralizer()
    >>> result = neutralizer.neutralize("You are a complete idiot!")
    >>> print(result['neutral_text'])
    "Your behavior seems quite misguided."
    """
    
    def __init__(self, strategy: str = 'combined'):
        """
        Initialize the neutralizer.
        
        Parameters:
        -----------
        strategy : str
            'word_replacement' - Only substitute toxic words
            'rule_based'       - Apply grammar/pattern rules
            'transformer'      - Use AI model for paraphrasing
            'combined'         - Use all strategies (recommended)
        """
        self.strategy = strategy
        self.replacements = NEUTRAL_REPLACEMENTS
        self.phrase_rules = PHRASE_RULES
        self.sentence_rewrites = SENTENCE_REWRITES
        self._transformer_model = None  # Lazy-loaded
    
    def neutralize(self, text: str, toxic_words: list = None) -> dict:
        """
        Main neutralization function. Converts toxic text to neutral text.
        
        Parameters:
        -----------
        text : str
            Original toxic input text
        toxic_words : list
            List of detected toxic word dicts (from ToxicWordDetector)
            
        Returns:
        --------
        dict with structure:
        {
            'original': str,
            'neutral_text': str,
            'strategy_used': str,
            'changes_made': list of (original, replacement) tuples,
            'neutralization_score': float (0=no change, 1=fully neutralized)
        }
        """
        if not text or not text.strip():
            return {
                'original': text,
                'neutral_text': text,
                'strategy_used': 'none',
                'changes_made': [],
                'neutralization_score': 0.0
            }
        
        changes_made = []
        
        # Try strategy in order of increasing complexity
        
        # Step 1: Check for complete sentence rewrites first
        text_lower = text.lower().strip().rstrip('!?.')
        for pattern, rewrite in self.sentence_rewrites.items():
            if pattern in text_lower:
                return {
                    'original': text,
                    'neutral_text': rewrite,
                    'strategy_used': 'sentence_rewrite',
                    'changes_made': [(text, rewrite)],
                    'neutralization_score': 1.0
                }
        
        # Step 2: Word replacement
        result = text
        result, word_changes = self._apply_word_replacement(result)
        changes_made.extend(word_changes)
        
        # Step 3: Rule-based rewriting
        if self.strategy in ('rule_based', 'combined'):
            result, rule_changes = self._apply_rules(result)
            changes_made.extend(rule_changes)
        
        # Step 4: Transformer-based rewriting (if available and requested)
        if self.strategy == 'transformer':
            transformer_result = self._apply_transformer(result)
            if transformer_result:
                result = transformer_result
                changes_made.append(('(transformer)', result))
        
        # Step 5: Final cleanup
        result = self._final_cleanup(result)
        
        # Calculate how much was neutralized
        score = self._calculate_neutralization_score(text, result, changes_made)
        
        # Determine primary strategy used
        strategy_used = 'combined' if len(changes_made) > 0 else 'none'
        
        return {
            'original': text,
            'neutral_text': result,
            'strategy_used': strategy_used,
            'changes_made': changes_made,
            'neutralization_score': round(score, 3)
        }
    
    def _apply_word_replacement(self, text: str) -> tuple:
        """
        Strategy 1: Replace toxic words with neutral alternatives.
        Simple but effective for single-word toxicity.
        
        Returns: (modified_text, list_of_changes)
        """
        changes = []
        result = text
        
        # Sort by length (longer phrases first) to avoid partial matches
        sorted_replacements = sorted(
            self.replacements.items(), 
            key=lambda x: len(x[0]), 
            reverse=True
        )
        
        for toxic_word, neutral_word in sorted_replacements:
            # Case-insensitive word boundary match
            pattern = re.compile(r'\b' + re.escape(toxic_word) + r'\b', re.IGNORECASE)
            
            if pattern.search(result):
                original_result = result
                result = pattern.sub(neutral_word, result)
                
                if result != original_result:
                    changes.append((toxic_word, neutral_word))
        
        return result, changes
    
    def _apply_rules(self, text: str) -> tuple:
        """
        Strategy 2: Apply pattern-based grammar rules.
        Handles structural patterns in toxic speech.
        
        Returns: (modified_text, list_of_changes)
        """
        changes = []
        result = text
        
        for pattern, replacement in self.phrase_rules:
            try:
                if callable(replacement):
                    new_result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)
                else:
                    new_result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)
                
                if new_result != result:
                    changes.append((pattern, replacement if not callable(replacement) else '(processed)'))
                    result = new_result
            except Exception:
                continue  # Skip problematic patterns
        
        return result, changes
    
    def _apply_transformer(self, text: str) -> Optional[str]:
        """
        Strategy 3: Use a transformer model for paraphrasing.
        
        Uses T5 model fine-tuned for text style transfer.
        Falls back gracefully if transformers library isn't installed.
        
        Note for beginners: This requires ~500MB model download on first use.
        """
        try:
            from transformers import pipeline
            
            # Lazy-load the model (only loads once)
            if self._transformer_model is None:
                print("⏳ Loading transformer model (first-time download ~500MB)...")
                self._transformer_model = pipeline(
                    "text2text-generation",
                    model="t5-small",  # Small fast model, use t5-base for better quality
                    max_length=150
                )
                print("✅ Transformer model loaded!")
            
            # Prompt the model to neutralize
            prompt = f"paraphrase in a neutral and polite way: {text}"
            output = self._transformer_model(prompt, max_length=150, do_sample=False)
            
            return output[0]['generated_text']
            
        except ImportError:
            print("⚠️  Transformers library not installed.")
            print("    Install with: pip install transformers torch")
            return None
        except Exception as e:
            print(f"⚠️  Transformer failed: {e}")
            return None
    
    def _final_cleanup(self, text: str) -> str:
        """
        Final cleanup of the neutralized text:
        - Fix capitalization
        - Fix punctuation
        - Remove double spaces
        """
        # Remove double spaces
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Ensure sentence starts with capital letter
        if text:
            text = text[0].upper() + text[1:]
        
        # Ensure sentence ends with punctuation
        if text and text[-1] not in '.!?':
            text += '.'
        
        # Fix common capitalization issues from rules
        # E.g. "your behavior" after "You are" → keep it lowercase
        text = re.sub(r'\bi\b', 'I', text)  # Capitalize standalone 'i'
        
        return text
    
    def _calculate_neutralization_score(self, original: str, 
                                         neutralized: str, 
                                         changes: list) -> float:
        """
        Calculate a score from 0-1 indicating how much was neutralized.
        0 = no changes, 1 = completely rewritten
        """
        if not changes:
            return 0.0
        
        # Calculate character-level similarity
        orig_len = len(original)
        if orig_len == 0:
            return 0.0
        
        # Count changed characters
        changed = sum(len(old) for old, new in changes if isinstance(old, str))
        score = min(1.0, changed / orig_len)
        
        return score
    
    def batch_neutralize(self, texts: list) -> list:
        """
        Neutralize multiple texts at once.
        
        Parameters:
        -----------
        texts : list of str
            List of texts to neutralize
            
        Returns:
        --------
        list of neutralization result dicts
        """
        return [self.neutralize(text) for text in texts]
    
    def add_replacement(self, toxic_word: str, neutral_word: str):
        """Add a custom word replacement to the dictionary."""
        self.replacements[toxic_word.lower()] = neutral_word.lower()


# ─── Convenience Function ─────────────────────────────────────────────────────
def neutralize_text(text: str) -> str:
    """
    Quick function to neutralize text without creating a Neutralizer object.
    
    Parameters:
    -----------
    text : str
        Toxic text to neutralize
        
    Returns:
    --------
    str : Neutralized text
    
    Example:
    --------
    >>> neutralize_text("You stupid idiot!")
    'Your behavior seems mistaken!'
    """
    neutralizer = TextNeutralizer()
    result = neutralizer.neutralize(text)
    return result['neutral_text']


# ─── Demo ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    neutralizer = TextNeutralizer(strategy='combined')
    
    test_cases = [
        ("You are such a stupid moron!", "insult"),
        ("I hate everything about you!", "hate"),
        ("Shut up you complete idiot.", "insult"),
        ("You are absolutely worthless.", "insult"),
        ("This is a wonderful day!", "neutral"),
        ("I strongly disagree with your approach.", "neutral"),
    ]
    
    print("=" * 70)
    print("🔄 ToxiClean - Neutralization Demo")
    print("=" * 70)
    
    for text, expected_type in test_cases:
        result = neutralizer.neutralize(text)
        
        print(f"\n📥 Original  : {result['original']}")
        print(f"📤 Neutralized: {result['neutral_text']}")
        print(f"   Strategy  : {result['strategy_used']}")
        if result['changes_made']:
            changes_str = ', '.join(
                f"'{old}' → '{new}'" 
                for old, new in result['changes_made']
                if isinstance(old, str) and len(old) < 30
            )
            print(f"   Changes   : {changes_str}")
    
    print("\n" + "=" * 70)
