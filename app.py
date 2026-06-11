import os
import json
import pickle
import numpy as np
import pandas as pd
from collections import Counter
from flask import Flask, render_template, request, jsonify

from data.loader import load_and_sample_reviews, load_movies, load_all_reviews
from data.preprocess import clean_text, get_keywords
from model.sentiment_model import SentimentModel
from model.recommender import MovieRecommender
from model.review_generator import template_generate, llm_generate, STYLE_BANKS

app = Flask(__name__)
app.secret_key = 'tomato-mind-secret-key-2024'

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'model', 'saved')
os.makedirs(MODEL_DIR, exist_ok=True)

SENTIMENT_MODEL_PATH = os.path.join(MODEL_DIR, 'sentiment_model.pkl')
TRANSFORMER_MODEL_DIR = os.path.join(MODEL_DIR, 'transformer')
RECOMMENDER_PATH = os.path.join(MODEL_DIR, 'recommender.pkl')

sentiment_model = None
movie_recommender = None

# Default training sample size.
_DEFAULT_SAMPLE_SIZE = int(os.environ.get('SENTIMENT_SAMPLE_SIZE', 100_000))


def get_sentiment_model():
    global sentiment_model
    if sentiment_model is not None:
        return sentiment_model

    # ── Auto-detect: use Transformer if Colab-exported model exists ──
    if _has_transformer_model():
        from model.transformer import TransformerSentimentModel
        sentiment_model = TransformerSentimentModel.load(TRANSFORMER_MODEL_DIR)
        return sentiment_model

    # ── Fallback: sklearn model ──
    if os.path.exists(SENTIMENT_MODEL_PATH):
        sentiment_model = SentimentModel.load(SENTIMENT_MODEL_PATH)
        print(f'Loaded saved sentiment model: {sentiment_model.summary()}')
    else:
        print(f'Training sentiment model on {_DEFAULT_SAMPLE_SIZE:,} reviews '
              f'(4 models, auto-select best)...')
        sentiment_model = SentimentModel(max_features=30000, ngram_range=(1, 3))
        reviews_df = load_and_sample_reviews(n=_DEFAULT_SAMPLE_SIZE)
        sentiment_model.train(reviews_df)
        sentiment_model.save(SENTIMENT_MODEL_PATH)
        sentiment_model.print_eval()
    return sentiment_model


def _has_transformer_model():
    """Check if a Colab-exported transformer model exists."""
    return (
        os.path.isdir(TRANSFORMER_MODEL_DIR)
        and os.path.isdir(os.path.join(TRANSFORMER_MODEL_DIR, 'tokenizer'))
        and os.path.isdir(os.path.join(TRANSFORMER_MODEL_DIR, 'model'))
    )


def get_recommender():
    global movie_recommender
    if movie_recommender is None:
        if os.path.exists(RECOMMENDER_PATH):
            movie_recommender = MovieRecommender.load(RECOMMENDER_PATH)
        else:
            movie_recommender = MovieRecommender()
            reviews_df = load_all_reviews()
            movies_df = load_movies()
            movie_recommender.fit(reviews_df, movies_df)
            movie_recommender.save(RECOMMENDER_PATH)
    return movie_recommender


# ── Whisper model (lazy load) ────────────────────────────────

_whisper_model = None

def get_whisper_model():
    """Lazy-load the Whisper base model for speech-to-text."""
    global _whisper_model
    if _whisper_model is None:
        import whisper
        _whisper_model = whisper.load_model('base')
        print(f'Whisper base model loaded on {_whisper_model.device}')
    return _whisper_model


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/ai-review')
def ai_review_page():
    return render_template('ai_review.html', styles=STYLE_BANKS)


@app.route('/recommendation')
def recommendation_page():
    return render_template('recommendation.html')


@app.route('/dashboard')
def dashboard_page():
    return render_template('dashboard.html')


@app.route('/about')
def about_page():
    return render_template('about.html')


# --- API: Model Evaluation ---

@app.route('/api/model/evaluation')
def api_model_evaluation():
    """Return the sentiment model's evaluation metrics."""
    model = get_sentiment_model()
    e = model.eval_
    if not e.get('accuracy'):
        return jsonify({'error': 'Model has not been evaluated yet.'}), 404

    result = {
        'accuracy': e['accuracy'],
        'f1': e['f1'],
        'precision': e['precision'],
        'recall': e['recall'],
        'classification_report': e['classification_report'],
        'train_size': e['train_size'],
        'test_size': e['test_size'],
        'class_distribution': e['class_distribution'],
    }

    # sklearn model extras
    result['selected_model'] = e.get('selected_model', e.get('model', 'unknown'))
    result['candidate_results'] = e.get('candidate_results', [])
    result['tfidf_config'] = e.get('tfidf_config', {})

    # transformer model extras
    result['model_name'] = e.get('model', e.get('selected_model', 'unknown'))
    result['hf_name'] = e.get('hf_name', '')
    result['hyperparams'] = e.get('hyperparams', {})
    result['best_epoch'] = e.get('best_epoch', None)

    return jsonify(result)


@app.route('/api/model/info')
def api_model_info():
    """Return which model is currently active."""
    model = get_sentiment_model()
    return jsonify({
        'type': type(model).__name__,
        'summary': model.summary(),
        'accuracy': model.eval_.get('accuracy'),
        'f1': model.eval_.get('f1'),
    })


# --- API: Recommendation ---

@app.route('/api/recommend/evaluation')
def api_recommend_evaluation():
    """Return the recommender's evaluation metrics."""
    rec = get_recommender()
    e = rec.eval_
    if not e:
        return jsonify({'error': 'No evaluation data yet. Run evaluate_all() first.'}), 404

    result = {}
    loo = e.get('leave_one_out')
    if loo:
        result['leave_one_out'] = {
            'n_users': loo['n_users'],
            'hit_rate': loo['hit_rate'],
            'precision': loo['precision'],
            'recall': loo['recall'],
        }
    div = e.get('diversity')
    if div:
        result['diversity'] = {
            'diversity': div['diversity'],
            'ils': div['ils'],
            'n_samples': div['n_samples'],
            'top_n': div['top_n'],
        }
    return jsonify(result)


@app.route('/api/recommend/evaluate', methods=['POST'])
def api_recommend_run_evaluation():
    """Trigger a new evaluation run for the recommender."""
    data = request.get_json() or {}
    n_users = int(data.get('n_users', 200))
    k_values = tuple(int(k) for k in data.get('k_values', [5, 10, 20]))
    n_diversity = int(data.get('n_diversity', 300))

    rec = get_recommender()
    result = rec.evaluate_all(
        n_users=n_users,
        k_values=k_values,
        n_diversity_samples=n_diversity,
    )
    rec.save(RECOMMENDER_PATH)
    return jsonify({
        'leave_one_out': result['leave_one_out'],
        'diversity': result['diversity'],
    })


# --- API: Recommendation ---

@app.route('/api/recommend/by-movies', methods=['POST'])
def api_recommend_by_movies():
    data = request.get_json()
    movie_names = [m.strip() for m in data.get('movie_names', []) if m.strip()]
    if not movie_names:
        return jsonify({'error': 'At least one movie name is required'}), 400

    rec = get_recommender()
    recommendations = rec.recommend_by_movies(movie_names)
    return jsonify({'recommendations': recommendations})


@app.route('/api/recommend/by-user', methods=['POST'])
def api_recommend_by_user():
    data = request.get_json()
    user_id = data.get('user_id', '').strip()
    if not user_id:
        return jsonify({'error': 'User ID is required'}), 400

    rec = get_recommender()
    recommendations = rec.recommend_by_user(user_id)
    return jsonify({'recommendations': recommendations})


@app.route('/api/recommend/by-mood', methods=['POST'])
def api_recommend_by_mood():
    data = request.get_json()
    mood_text = data.get('mood', '').strip().lower()
    if not mood_text:
        return jsonify({'error': 'Please describe your mood.'}), 400

    rec = get_recommender()
    recommendations = rec.recommend_by_mood(mood_text)
    return jsonify({'recommendations': recommendations})


@app.route('/api/users/top')
def api_top_users():
    rec = get_recommender()
    top_users = rec.get_top_users(200)
    return jsonify({'users': top_users})


@app.route('/api/movies/search')
def api_movies_search():
    q = request.args.get('q', '').strip().lower()
    if len(q) < 2:
        return jsonify({'movies': []})

    movies_df = load_movies()
    matches = movies_df[movies_df['movie_title'].str.lower().str.contains(q, na=False)]
    results = matches[['rotten_tomatoes_link', 'movie_title']].head(15)
    return jsonify({
        'movies': [{'id': r['rotten_tomatoes_link'], 'title': r['movie_title']}
                   for _, r in results.iterrows()]
    })


@app.route('/api/movies/all')
def api_all_movies():
    movies_df = load_movies()
    titles = movies_df[['movie_title']].dropna().drop_duplicates().sort_values('movie_title')
    return jsonify({
        'movies': [{'title': m['movie_title']} for _, m in titles.iterrows()]
    })


# --- API: Voice Transcription (Whisper) ---

@app.route('/api/transcribe', methods=['POST'])
def api_transcribe():
    """Accept raw WAV bytes, return transcribed text via Whisper.

    WAV is decoded in pure Python (struct) — no ffmpeg required.
    16 kHz, 16-bit, mono WAV expected from the browser recorder.
    """
    raw_data = request.get_data()
    if not raw_data:
        return jsonify({'error': 'No audio data provided'}), 400

    # Parse WAV header (44 bytes) → extract raw PCM samples
    import struct
    try:
        # Read fmt chunk to find audio format params
        # Standard WAV: RIFF(12) + fmt (24) + data(8) + samples
        sample_rate = struct.unpack_from('<I', raw_data, 24)[0]
        bits_per_sample = struct.unpack_from('<H', raw_data, 34)[0]
        data_offset = 44  # standard PCM WAV header size
        # PCM 16-bit → int16 numpy array
        pcm_bytes = raw_data[data_offset:]
        if bits_per_sample == 16:
            samples = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        elif bits_per_sample == 8:
            samples = np.frombuffer(pcm_bytes, dtype=np.uint8).astype(np.float32) / 128.0 - 1.0
        else:
            return jsonify({'error': f'Unsupported bit depth: {bits_per_sample}'}), 400
    except Exception as e:
        return jsonify({'error': f'WAV parse failed: {str(e)}'}), 400

    if len(samples) < sample_rate * 0.5:  # less than 0.5s is likely silent
        return jsonify({'text': ''})

    try:
        model = get_whisper_model()
        result = model.transcribe(samples, fp16=False, language='en')
        text = result['text'].strip()
        return jsonify({'text': text, 'language': result.get('language', 'en')})
    except Exception as e:
        return jsonify({'error': f'Transcription failed: {str(e)}'}), 500


# --- API: User Review Submission ---

@app.route('/api/review/add', methods=['POST'])
def api_review_add():
    """Submit a user-written review + get sentiment analysis back."""
    data = request.get_json()
    review_text = data.get('review_text', '').strip()
    movie_id = data.get('movie_id', '').strip()
    critic_name = data.get('critic_name', 'You').strip() or 'You'

    if not review_text or len(review_text) < 10:
        return jsonify({'error': 'Review must be at least 10 characters.'}), 400

    # Sentiment prediction
    model = get_sentiment_model()
    result = model.predict(review_text)
    keywords = get_keywords(review_text, top_n=5)

    return jsonify({
        'movie_id': movie_id,
        'critic_name': critic_name,
        'content': review_text[:500],
        'sentiment': result['sentiment'],
        'confidence': round(result['confidence'] * 100, 1),
        'keywords': keywords,
        'review_type': 'User-Submitted',
    })


# --- Movie Detail & Review Analysis ---

@app.route('/movie')
def movie_page():
    return render_template('movie.html')


@app.route('/api/movie/<path:movie_id>')
def api_movie_detail(movie_id):
    movies_df = load_movies()
    movie = movies_df[movies_df['rotten_tomatoes_link'] == movie_id]
    if movie.empty:
        return jsonify({'error': 'Movie not found'}), 404

    m = movie.iloc[0]
    reviews_df = load_all_reviews()
    movie_reviews = reviews_df[reviews_df['rotten_tomatoes_link'] == movie_id].copy()

    fresh_count = int((movie_reviews['review_type'] == 'Fresh').sum())
    rotten_count = int((movie_reviews['review_type'] == 'Rotten').sum())
    total_reviews = len(movie_reviews)
    top_critics_count = int((movie_reviews['top_critic'] == True).sum())

    # ── Batch prediction (10-50× faster than per-review loop) ──
    valid_reviews = []
    review_texts = []
    for _, rev in movie_reviews.iterrows():
        content = str(rev['review_content']) if pd.notna(rev['review_content']) else ''
        if len(content) < 10:
            continue
        valid_reviews.append(rev)
        review_texts.append(content)

    model = get_sentiment_model()
    batch_results = model.predict_batch(review_texts) if review_texts else []

    # ── Build results ──
    analyzed_reviews = []
    pos_count = 0
    neg_count = 0
    all_keywords = []
    for rev, result in zip(valid_reviews, batch_results):
        content = str(rev['review_content']) if pd.notna(rev['review_content']) else ''
        keywords = get_keywords(content, top_n=5)
        all_keywords.extend(keywords)
        if result['sentiment'].startswith('Positive') or result['sentiment'].startswith('Mixed-Positive'):
            pos_count += 1
        else:
            neg_count += 1
        analyzed_reviews.append({
            'critic_name': str(rev['critic_name']) if pd.notna(rev['critic_name']) else 'Anonymous',
            'top_critic': bool(rev['top_critic']) if pd.notna(rev['top_critic']) else False,
            'publisher': str(rev['publisher_name']) if pd.notna(rev['publisher_name']) else '',
            'review_type': str(rev['review_type']),
            'review_score': str(rev['review_score']) if pd.notna(rev['review_score']) else '',
            'review_date': str(rev['review_date']) if pd.notna(rev['review_date']) else '',
            'content': content[:400],
            'sentiment': result['sentiment'],
            'confidence': round(result['confidence'] * 100, 1)
        })

    from collections import Counter
    keyword_freq = Counter(all_keywords).most_common(15)

    movie_title = str(m['movie_title'])

    # ── Recommended movies (by this movie) ──
    rec = get_recommender()
    recommended = rec.recommend_by_movies([movie_title], top_n=6)
    # Add movie_id to each recommendation so frontend can link
    for r in recommended:
        mid = rec._find_movie_id(r['title'])
        r['id'] = mid if mid else ''

    return jsonify({
        'movie': {
            'id': movie_id,
            'title': movie_title,
            'info': str(m['movie_info']) if pd.notna(m['movie_info']) else '',
            'consensus': str(m['critics_consensus']) if pd.notna(m['critics_consensus']) else '',
            'rating': str(m['content_rating']) if pd.notna(m['content_rating']) else 'NR',
            'genres': str(m['genres']) if pd.notna(m['genres']) else '',
            'directors': str(m['directors']) if pd.notna(m['directors']) else '',
            'actors': ', '.join(str(m['actors']).split(',')[:5]) if pd.notna(m['actors']) else '',
            'release_date': str(m['original_release_date']) if pd.notna(m['original_release_date']) else '',
            'runtime': str(m['runtime']) if pd.notna(m['runtime']) else 'N/A',
            'tomatometer_rating': int(m['tomatometer_rating']) if pd.notna(m['tomatometer_rating']) else 0,
            'tomatometer_status': str(m['tomatometer_status']) if pd.notna(m['tomatometer_status']) else '',
            'audience_rating': int(m['audience_rating']) if pd.notna(m['audience_rating']) else 0,
            'audience_status': str(m['audience_status']) if pd.notna(m['audience_status']) else '',
        },
        'stats': {
            'total_reviews': total_reviews,
            'fresh_count': fresh_count,
            'rotten_count': rotten_count,
            'fresh_pct': round(fresh_count / total_reviews * 100, 1) if total_reviews > 0 else 0,
            'top_critics_count': top_critics_count,
            'model_pos_count': pos_count,
            'model_neg_count': neg_count,
            'model_pos_pct': round(pos_count / (pos_count + neg_count) * 100, 1) if (pos_count + neg_count) > 0 else 0,
        },
        'keywords': [{'text': w, 'count': c} for w, c in keyword_freq],
        'reviews': analyzed_reviews[:30],
        'recommended': recommended,
    })


# --- API: Movie Comparison ---

@app.route('/api/movies/compare')
def api_movies_compare():
    """Return side-by-side comparison data for two movies."""
    movie_id_a = request.args.get('a', '').strip()
    movie_id_b = request.args.get('b', '').strip()
    if not movie_id_a or not movie_id_b:
        return jsonify({'error': 'Both movie IDs (a and b) are required'}), 400

    movies_df = load_movies()
    model = get_sentiment_model()

    results = []
    for mid in [movie_id_a, movie_id_b]:
        movie = movies_df[movies_df['rotten_tomatoes_link'] == mid]
        if movie.empty:
            results.append(None)
            continue
        m = movie.iloc[0]
        reviews_df = load_all_reviews()
        movie_reviews = reviews_df[reviews_df['rotten_tomatoes_link'] == mid]

        # Batch sentiment
        valid_texts = []
        for _, rev in movie_reviews.iterrows():
            c = str(rev['review_content']) if pd.notna(rev['review_content']) else ''
            if len(c) >= 10:
                valid_texts.append(c)
        batch_results = model.predict_batch(valid_texts[:50]) if valid_texts else []

        pos_count = sum(1 for r in batch_results if
                       r['sentiment'].startswith('Positive') or r['sentiment'].startswith('Mixed-Positive'))
        neg_count = len(batch_results) - pos_count

        # Keywords
        all_kw = []
        for t in valid_texts[:50]:
            all_kw.extend(get_keywords(t, top_n=3))
        kw_freq = Counter(all_kw).most_common(10)

        results.append({
            'id': mid,
            'title': str(m['movie_title']),
            'tomatometer_rating': int(m['tomatometer_rating']) if pd.notna(m['tomatometer_rating']) else 0,
            'tomatometer_status': str(m['tomatometer_status']) if pd.notna(m['tomatometer_status']) else '',
            'audience_rating': int(m['audience_rating']) if pd.notna(m['audience_rating']) else 0,
            'genres': str(m['genres']) if pd.notna(m['genres']) else '',
            'directors': str(m['directors']) if pd.notna(m['directors']) else '',
            'consensus': str(m['critics_consensus']) if pd.notna(m['critics_consensus']) else '',
            'release_date': str(m['original_release_date']) if pd.notna(m['original_release_date']) else '',
            'runtime': str(m['runtime']) if pd.notna(m['runtime']) else 'N/A',
            'sentiment': {
                'positive': pos_count,
                'negative': neg_count,
                'positive_pct': round(pos_count / (pos_count + neg_count) * 100, 1) if (pos_count + neg_count) > 0 else 0,
            },
            'keywords': [{'text': w, 'count': c} for w, c in kw_freq],
        })

    return jsonify({'a': results[0], 'b': results[1]})


# --- API: Movie List (for Dashboard drill-down) ---

@app.route('/api/movies/list')
def api_movies_list():
    """Filter movies by status, year, genre, rating range. Used by Dashboard drill-down."""
    status = request.args.get('status', '').strip()  # Fresh / Rotten / Certified-Fresh
    year = request.args.get('year', '').strip()
    genre = request.args.get('genre', '').strip()
    rating_min = request.args.get('rating_min', '').strip()
    rating_max = request.args.get('rating_max', '').strip()
    sort = request.args.get('sort', 'rating').strip()  # rating | year | title
    limit = int(request.args.get('limit', 20))

    movies_df = load_movies()
    df = movies_df.copy()

    if status:
        if status == 'Fresh':
            df = df[df['tomatometer_status'].isin(['Fresh', 'Certified-Fresh'])]
        elif status in ('Rotten', 'Certified-Fresh'):
            df = df[df['tomatometer_status'] == status]
    if year:
        df['release_year'] = df['original_release_date'].astype(str).str[:4]
        df = df[df['release_year'] == year]
    if genre:
        df = df[df['genres'].astype(str).str.contains(genre, case=False, na=False)]
    if rating_min:
        df = df[df['tomatometer_rating'] >= int(rating_min)]
    if rating_max:
        df = df[df['tomatometer_rating'] <= int(rating_max)]

    # Sort
    if sort == 'rating':
        df = df.sort_values('tomatometer_rating', ascending=False)
    elif sort == 'year':
        df = df.sort_values('original_release_date', ascending=False)
    else:
        df = df.sort_values('movie_title')

    total = len(df)
    df = df.head(limit)

    return jsonify({
        'total': int(total),
        'movies': [{
            'id': str(r['rotten_tomatoes_link']),
            'title': str(r['movie_title']),
            'rating': int(r['tomatometer_rating']) if pd.notna(r['tomatometer_rating']) else 0,
            'status': str(r['tomatometer_status']) if pd.notna(r['tomatometer_status']) else '',
            'genres': str(r['genres']) if pd.notna(r['genres']) else '',
            'year': str(r['original_release_date'])[:4] if pd.notna(r['original_release_date']) else '',
            'consensus': str(r['critics_consensus'])[:150] if pd.notna(r['critics_consensus']) else '',
        } for _, r in df.iterrows()]
    })


# --- API: Dashboard ---

@app.route('/api/dashboard/overview')
def api_dashboard_overview():
    movies_df = load_movies()
    reviews_df = load_and_sample_reviews()

    total_movies = len(movies_df) if movies_df is not None else 0
    total_reviews = 1130017  # known total from dataset
    fresh_count = int((movies_df['tomatometer_status'] == 'Fresh').sum() +
                      (movies_df['tomatometer_status'] == 'Certified-Fresh').sum())

    return jsonify({
        'total_movies': int(total_movies),
        'total_reviews': total_reviews,
        'avg_tomatometer': round(float(movies_df['tomatometer_rating'].dropna().mean()), 1),
        'fresh_pct': round(fresh_count / len(movies_df) * 100, 1) if len(movies_df) > 0 else 0
    })


@app.route('/api/dashboard/sentiment-distribution')
def api_dashboard_sentiment_distribution():
    movies_df = load_movies()
    fresh = int((movies_df['tomatometer_status'] == 'Fresh').sum() +
                (movies_df['tomatometer_status'] == 'Certified-Fresh').sum())
    rotten = int((movies_df['tomatometer_status'] == 'Rotten').sum())

    return jsonify({
        'labels': ['Fresh', 'Rotten'],
        'data': [fresh, rotten]
    })


@app.route('/api/dashboard/rating-distribution')
def api_dashboard_rating_distribution():
    movies_df = load_movies()
    ratings = movies_df['tomatometer_rating'].dropna()
    bins = [0, 20, 40, 60, 80, 100]
    labels = ['0-20', '20-40', '40-60', '60-80', '80-100']
    hist = [int(((ratings >= b) & (ratings < b + 20)).sum()) for b in bins[:-1]]
    hist[-1] += int((ratings == 100).sum())

    return jsonify({'labels': labels, 'data': hist})


@app.route('/api/dashboard/genre-popularity')
def api_dashboard_genre_popularity():
    movies_df = load_movies()
    genre_counter = Counter()
    for genres_str in movies_df['genres'].dropna():
        for g in genres_str.split(','):
            g = g.strip().strip('"')
            if g:
                genre_counter[g] += 1

    top = genre_counter.most_common(10)
    return jsonify({
        'labels': [t[0] for t in top],
        'data': [t[1] for t in top]
    })


@app.route('/api/dashboard/yearly-trend')
def api_dashboard_yearly_trend():
    movies_df = load_movies()
    movies_df = movies_df.copy()
    movies_df['year'] = movies_df['original_release_date'].str[:4]
    yearly = movies_df.groupby('year').agg(
        count=('rotten_tomatoes_link', 'count'),
        avg_rating=('tomatometer_rating', 'mean')
    ).reset_index()
    yearly = yearly[(yearly['year'] >= '2000') & (yearly['year'] <= '2025')].sort_values('year')

    return jsonify({
        'years': yearly['year'].tolist(),
        'counts': yearly['count'].tolist(),
        'ratings': [round(float(r), 1) for r in yearly['avg_rating'].tolist()]
    })


@app.route('/api/dashboard/wordcloud')
def api_dashboard_wordcloud():
    movies_df = load_movies()
    text = ' '.join(movies_df['critics_consensus'].dropna().tolist())
    words = clean_text(text).split()
    word_freq = Counter(words).most_common(80)

    return jsonify({
        'words': [{'text': w, 'size': c} for w, c in word_freq]
    })


@app.route('/api/generate-review', methods=['POST'])
def api_generate_review():
    data = request.get_json()
    movie_name = data.get('movie_name', '').strip()
    short_feeling = data.get('short_feeling', '').strip()
    style = data.get('style', 'hype')
    sentiment_ratio = data.get('sentiment_ratio', None)
    use_api = data.get('use_api', False)
    max_words = data.get('max_words', 200)

    if not movie_name or not short_feeling:
        return jsonify({'error': 'movie_name and short_feeling are required'}), 400

    api_key = os.environ.get('DEEPSEEK_API_KEY') if use_api else None

    if use_api and api_key:
        result = llm_generate(movie_name, short_feeling, style,
                              sentiment_ratio=sentiment_ratio, max_words=max_words,
                              api_key=api_key)
    else:
        result = template_generate(movie_name, short_feeling, style,
                                   sentiment_ratio=sentiment_ratio)

    return jsonify(result)


if __name__ == '__main__':
    print("Starting TomatoMind server...")
    print("Loading models (this may take a minute on first run)...")
    get_sentiment_model()
    get_recommender()
    print("Server ready at http://127.0.0.1:5001")
    app.run(debug=True, host='127.0.0.1', port=5001)
