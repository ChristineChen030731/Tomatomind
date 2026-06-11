import re
import nltk
from nltk.corpus import stopwords

_stopwords_cache = None


def _get_stopwords():
    global _stopwords_cache
    if _stopwords_cache is None:
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords', quiet=True)
        _stopwords_cache = set(stopwords.words('english'))
    return _stopwords_cache


def clean_text(text):
    text = text.lower()
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    stop_words = _get_stopwords()
    words = [w for w in text.split() if w not in stop_words and len(w) > 1]
    return ' '.join(words)


def get_keywords(text, top_n=8):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    stop_words = _get_stopwords()
    words = [w for w in text.split() if w not in stop_words and len(w) > 2]
    from collections import Counter
    return [w for w, _ in Counter(words).most_common(top_n)]
