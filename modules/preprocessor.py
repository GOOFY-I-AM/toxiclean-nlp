"""
ToxiClean - Text Preprocessing Module
======================================
Handles all text cleaning and normalization steps before feeding
text into the toxicity detection model.
"""

import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Download required NLTK data (run once)
def download_nltk_resources():
    """Download all required NLTK resources."""
    resources = ['stopwords', 'wordnet', 'punkt', 'averaged_perceptron_tagger']
    for resource in resources:
        try:
            nltk.download(resource, quiet=True)
        except Exception as e:
            print(f"Warning: Could not download {resource}: {e}")

download_nltk_resources()

# Initialize the lemmatizer (converts words to their base form)
lemmatizer = WordNetLemmatizer()

# Load English stopwords (common words like "the", "is", "at")
try:
    STOP_WORDS = set(stopwords.words('english'))
except LookupError:
    STOP_WORDS = set()


def lowercase_text(text: str) -> str:
    """
    Step 1: Convert all text to lowercase.
    Example: "Hello World" → "hello world"
    """
    return text.lower()


def remove_urls(text: str) -> str:
    """
    Step 2: Remove URLs from text.
    Example: "Visit https://example.com now" → "Visit  now"
    """
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub('', text)


def remove_html_tags(text: str) -> str:
    """
    Step 3: Remove HTML tags.
    Example: "<b>Hello</b>" → "Hello"
    """
    html_pattern = re.compile(r'<.*?>')
    return html_pattern.sub('', text)


def remove_special_characters(text: str) -> str:
    """
    Step 4: Remove special characters and numbers, keep only letters and spaces.
    Example: "Hello!! World123" → "Hello  World"
    """
    # Keep only alphabetic characters and spaces
    clean = re.sub(r'[^a-zA-Z\s]', ' ', text)
    # Remove extra whitespace
    clean = re.sub(r'\s+', ' ', clean).strip()
    return clean


def remove_punctuation(text: str) -> str:
    """
    Step 5: Remove punctuation marks.
    Example: "Hello, World!" → "Hello World"
    """
    translator = str.maketrans('', '', string.punctuation)
    return text.translate(translator)


def tokenize_text(text: str) -> list:
    """
    Step 6: Tokenization - split text into individual words (tokens).
    Example: "hello world" → ["hello", "world"]
    """
    return word_tokenize(text)


def remove_stopwords(tokens: list) -> list:
    """
    Step 7: Remove stopwords - common words that don't carry much meaning.
    Example: ["i", "am", "very", "angry"] → ["angry"]
    
    Note: For toxicity detection, we keep some stopwords like "not", "very"
    as they can change meaning. Here we remove only the safest ones.
    """
    # Keep negation words as they affect meaning
    keep_words = {'not', 'no', 'nor', 'very', 'too', 'more', 'most'}
    
    filtered = [
        word for word in tokens 
        if word not in STOP_WORDS or word in keep_words
    ]
    return filtered


def lemmatize_tokens(tokens: list) -> list:
    """
    Step 8: Lemmatization - convert words to their base/dictionary form.
    Example: ["running", "hated", "dogs"] → ["run", "hate", "dog"]
    
    This helps the model recognize different forms of the same word.
    """
    return [lemmatizer.lemmatize(token) for token in tokens]


def preprocess_text(text: str, 
                    for_model: bool = True,
                    remove_stops: bool = False) -> str:
    """
    Full preprocessing pipeline combining all steps above.
    
    Parameters:
    -----------
    text : str
        Raw input text from user
    for_model : bool
        If True, returns cleaned string ready for vectorization
        If False, returns lightly cleaned text (preserves more original content)
    remove_stops : bool
        Whether to remove stopwords (False by default for toxicity detection)
    
    Returns:
    --------
    str : Cleaned and preprocessed text
    
    Example:
    --------
    >>> preprocess_text("You are SO STUPID!! Visit www.spam.com")
    'you stupid'
    """
    if not isinstance(text, str):
        return ""
    
    # Apply each preprocessing step
    text = lowercase_text(text)         # Step 1
    text = remove_urls(text)            # Step 2
    text = remove_html_tags(text)       # Step 3
    
    if for_model:
        text = remove_special_characters(text)  # Step 4
        tokens = tokenize_text(text)            # Step 6
        
        if remove_stops:
            tokens = remove_stopwords(tokens)   # Step 7
        
        tokens = lemmatize_tokens(tokens)       # Step 8
        text = ' '.join(tokens)
    else:
        text = remove_punctuation(text)         # Step 5 (lighter cleaning)
    
    return text.strip()


def preprocess_batch(texts: list, **kwargs) -> list:
    """
    Preprocess a list of texts (batch processing for datasets).
    
    Parameters:
    -----------
    texts : list
        List of raw text strings
    
    Returns:
    --------
    list : List of preprocessed text strings
    """
    return [preprocess_text(text, **kwargs) for text in texts]


# ─── Quick Demo ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    sample_texts = [
        "You are SO STUPID and worthless!!!",
        "I hate everything about you, you idiot!",
        "Have a wonderful day, everyone! 😊",
        "Visit www.spam.com for FREE MONEY!!!",
    ]
    
    print("=" * 60)
    print("ToxiClean - Preprocessing Demo")
    print("=" * 60)
    
    for text in sample_texts:
        cleaned = preprocess_text(text)
        print(f"\nOriginal : {text}")
        print(f"Cleaned  : {cleaned}")
    
    print("\n" + "=" * 60)
