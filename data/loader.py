import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARCHIVE_DIR = os.path.join(BASE_DIR, 'archive')

MOVIES_PATH = os.path.join(ARCHIVE_DIR, 'rotten_tomatoes_movies.csv')
REVIEWS_PATH = os.path.join(ARCHIVE_DIR, 'rotten_tomatoes_critic_reviews.csv')

_movies_cache = None
_reviews_cache = None
_sampled_reviews_cache = None


def load_movies(force=False):
    global _movies_cache
    if _movies_cache is not None and not force:
        return _movies_cache
    _movies_cache = pd.read_csv(MOVIES_PATH, encoding='utf-8', on_bad_lines='skip')
    _movies_cache.columns = _movies_cache.columns.str.strip()
    return _movies_cache


def load_all_reviews(force=False):
    global _reviews_cache
    if _reviews_cache is not None and not force:
        return _reviews_cache
    _reviews_cache = pd.read_csv(REVIEWS_PATH, encoding='utf-8', on_bad_lines='skip')
    _reviews_cache.columns = _reviews_cache.columns.str.strip()
    return _reviews_cache


def load_and_sample_reviews(n=100000, random_state=42, force=False):
    global _sampled_reviews_cache
    if _sampled_reviews_cache is not None and not force:
        return _sampled_reviews_cache

    df = load_all_reviews(force)
    valid = df[df['review_type'].isin(['Fresh', 'Rotten']) & df['review_content'].notna()]
    valid = valid[valid['review_content'].str.strip().str.len() > 10]
    n = min(n, len(valid))
    _sampled_reviews_cache = valid.sample(n=n, random_state=random_state).copy()
    return _sampled_reviews_cache
