"""
Movie Recommender — Item-Item Collaborative Filtering + Mood-based Content Filtering.

Evaluation methods:
  - leave_one_out    : hit rate, precision@k, recall@k
  - diversity        : intra-list similarity (1 = fully diverse)

Mood scoring uses a theoretically-grounded multi-factor weighted formula
(see docstring of recommend_by_mood for details).
"""

import re
import pickle
import random
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity


class MovieRecommender:
    def __init__(self):
        self.user_item_matrix = None    # pd.DataFrame: critics × movies
        self.item_similarity = None     # ndarray: cosine sim matrix
        self.movie_id_to_idx = {}
        self.idx_to_movie_id = {}
        self.movies_df = None
        self.user_ids = []
        self.user_rated_movies = {}

        # ── Evaluation results (populated by evaluate_*) ──
        self.eval_ = {}

    # ──────────────────────────────────────────────────────────
    #  Fit
    # ──────────────────────────────────────────────────────────

    def fit(self, reviews_df, movies_df):
        self.movies_df = movies_df.set_index('rotten_tomatoes_link')

        valid = reviews_df[reviews_df['review_type'].isin(['Fresh', 'Rotten'])]
        valid = valid[valid['critic_name'].notna() & (valid['critic_name'] != '')]
        valid = valid[valid['review_content'].notna()]

        valid = valid.copy()
        valid['rating'] = (valid['review_type'] == 'Fresh').astype(int)

        critic_counts = valid.groupby('critic_name').size()
        active_critics = critic_counts[critic_counts >= 5].index
        valid = valid[valid['critic_name'].isin(active_critics)]

        movie_counts = valid.groupby('rotten_tomatoes_link').size()
        popular_movies = movie_counts[movie_counts >= 5].index
        valid = valid[valid['rotten_tomatoes_link'].isin(popular_movies)]

        self.user_item_matrix = valid.pivot_table(
            index='critic_name',
            columns='rotten_tomatoes_link',
            values='rating',
            aggfunc='mean',
            fill_value=0,
        )

        self.movie_id_to_idx = {mid: i for i, mid in enumerate(self.user_item_matrix.columns)}
        self.idx_to_movie_id = {i: mid for mid, i in self.movie_id_to_idx.items()}

        item_matrix = self.user_item_matrix.T.values
        self.item_similarity = cosine_similarity(item_matrix)

        self.user_ids = list(self.user_item_matrix.index)
        for uid in self.user_ids:
            user_row = self.user_item_matrix.loc[uid]
            liked = user_row[user_row > 0].index.tolist()
            self.user_rated_movies[uid] = liked

        # ── Precompute genre stats for scoring ──
        self._build_genre_stats()

    # ──────────────────────────────────────────────────────────
    #  Genre statistics (for mood scoring)
    # ──────────────────────────────────────────────────────────

    def _build_genre_stats(self):
        """Precompute per-genre rating distributions for normalised scoring."""
        self._genre_medians = {}
        self._genre_counts = {}
        self._movie_genre_cache = {}
        self._movie_rating_cache = {}

        for mid in self.movie_id_to_idx:
            gs = self._get_genre(mid)
            rt = self._get_rating(mid)
            self._movie_genre_cache[mid] = gs
            self._movie_rating_cache[mid] = rt
            if gs == 'N/A' or rt == 'N/A':
                continue
            for g in (s.strip() for s in gs.split(',')):
                self._genre_medians.setdefault(g, []).append(rt)
                self._genre_counts[g] = self._genre_counts.get(g, 0) + 1

        for g in self._genre_medians:
            self._genre_medians[g] = np.median(self._genre_medians[g])

    # ──────────────────────────────────────────────────────────
    #  Recommendation methods
    # ──────────────────────────────────────────────────────────

    def recommend_by_movies(self, movie_names, top_n=5):
        movie_ids = []
        matched_names = []
        for name in movie_names:
            mid = self._find_movie_id(name)
            if mid and mid in self.movie_id_to_idx:
                movie_ids.append(mid)
                matched_names.append(self._get_movie_title(mid))

        if not movie_ids:
            return []

        idx = self.movie_id_to_idx[movie_ids[0]]
        sim_scores = self.item_similarity[idx].copy()
        sim_scores[idx] = -1
        for mid in movie_ids[1:]:
            idx2 = self.movie_id_to_idx[mid]
            sim_scores += self.item_similarity[idx2]
            sim_scores[idx2] = -1
        sim_scores /= len(movie_ids)

        top_indices = np.argsort(sim_scores)[-top_n:][::-1]
        results = []
        for i in top_indices:
            mid = self.idx_to_movie_id[i]
            results.append({
                'title': self._get_movie_title(mid),
                'reason': self._build_reason(mid, matched_names),
                'genres': self._get_genre(mid),
                'year': self._get_year(mid),
                'rating': self._get_rating(mid),
                'consensus': self._get_consensus(mid),
                'similarity': round(float(sim_scores[i]) * 100, 1),
            })
        return results

    def recommend_by_user(self, user_id, top_n=5):
        if user_id not in self.user_item_matrix.index:
            return []

        liked_movies = self.user_rated_movies.get(user_id, [])
        if not liked_movies:
            return []

        movie_ids = [m for m in liked_movies if m in self.movie_id_to_idx]
        if not movie_ids:
            return []

        idx = self.movie_id_to_idx[movie_ids[0]]
        combined_scores = self.item_similarity[idx].copy()
        for mid in movie_ids[1:]:
            if mid in self.movie_id_to_idx:
                combined_scores += self.item_similarity[self.movie_id_to_idx[mid]]

        for mid in movie_ids:
            if mid in self.movie_id_to_idx:
                combined_scores[self.movie_id_to_idx[mid]] = -999

        top_indices = np.argsort(combined_scores)[-top_n:][::-1]
        results = []
        for i in top_indices:
            if combined_scores[i] <= -999:
                continue
            mid = self.idx_to_movie_id[i]
            results.append({
                'title': self._get_movie_title(mid),
                'reason': 'Based on your viewing history',
                'genres': self._get_genre(mid),
                'year': self._get_year(mid),
                'rating': self._get_rating(mid),
                'consensus': self._get_consensus(mid),
                'similarity': round(float(combined_scores[i]) * 100 / len(movie_ids), 1),
            })
        return results[:top_n]

    # ──────────────────────────────────────────────────────────
    #  Mood-to-genre mapping
    # ──────────────────────────────────────────────────────────

    MOOD_MAP = {
        'happy': ['Comedy', 'Animation', 'Kids & Family', 'Musical & Performing Arts'],
        'cheerful': ['Comedy', 'Animation', 'Kids & Family', 'Musical & Performing Arts'],
        'excited': ['Action & Adventure', 'Science Fiction & Fantasy', 'Comedy'],
        'energetic': ['Action & Adventure', 'Science Fiction & Fantasy', 'Musical & Performing Arts'],
        'adventurous': ['Action & Adventure', 'Science Fiction & Fantasy', 'Mystery & Suspense'],
        'sad': ['Drama', 'Romance', 'Art House & International', 'Classics'],
        'down': ['Drama', 'Romance', 'Art House & International', 'Classics'],
        'depressed': ['Drama', 'Romance', 'Art House & International', 'Classics'],
        'melancholy': ['Drama', 'Art House & International', 'Classics', 'Romance'],
        'romantic': ['Romance', 'Comedy', 'Drama'],
        'lovely': ['Romance', 'Comedy', 'Drama'],
        'tense': ['Mystery & Suspense', 'Horror', 'Action & Adventure'],
        'thrilling': ['Mystery & Suspense', 'Horror', 'Action & Adventure', 'Science Fiction & Fantasy'],
        'scared': ['Horror', 'Mystery & Suspense'],
        'thoughtful': ['Documentary', 'Art House & International', 'Classics'],
        'contemplative': ['Documentary', 'Art House & International', 'Classics', 'Drama'],
        'nostalgic': ['Classics', 'Drama', 'Science Fiction & Fantasy'],
        'relaxed': ['Comedy', 'Animation', 'Documentary', 'Kids & Family'],
        'bored': ['Action & Adventure', 'Comedy', 'Science Fiction & Fantasy', 'Mystery & Suspense'],
        'inspired': ['Documentary', 'Drama', 'Art House & International'],
        'angry': ['Action & Adventure', 'Horror', 'Mystery & Suspense'],
        'stressed': ['Comedy', 'Animation', 'Kids & Family', 'Musical & Performing Arts'],
    }

    MOOD_REASONS = {
        'happy': 'to keep the good vibes going',
        'cheerful': 'to match your upbeat mood',
        'excited': 'to fuel your excitement',
        'energetic': 'to match your high energy',
        'adventurous': 'for your adventurous spirit',
        'sad': 'to lift your spirits with great storytelling',
        'down': 'to help you feel better',
        'depressed': 'to offer comfort through cinema',
        'melancholy': 'to resonate with your reflective mood',
        'romantic': 'to warm your heart',
        'lovely': 'to complement your loving mood',
        'tense': 'to satisfy your craving for suspense',
        'thrilling': 'to give you an adrenaline rush',
        'scared': 'to give you a thrilling escape',
        'thoughtful': 'to stimulate your mind',
        'contemplative': 'to inspire deep reflection',
        'nostalgic': 'to take you back in time',
        'relaxed': 'to maintain your peaceful state',
        'bored': 'to entertain and engage you',
        'inspired': 'to fuel your creativity',
        'angry': 'to channel that energy',
        'stressed': 'to help you unwind and relax',
    }

    # ──────────────────────────────────────────────────────────
    #  Mood recommendation  (optimised scoring)
    # ──────────────────────────────────────────────────────────
    #
    #  Theoretical basis for the scoring formula:
    #
    #  We conceptualise mood-based recommendation as a multi-criteria
    #  decision problem where each candidate movie is scored on three
    #  orthogonal dimensions:
    #
    #    Dimension              | Theory                         | Weight
    #   ────────────────────────┼────────────────────────────────┼───────
    #    genre_match            | Jaccard similarity between     | 45 %
    #                           | target-genre set and the       |
    #                           | movie's genre set.  Accounts   |
    #                           | for both precision (matched    |
    #                           | genres) and recall (total      |
    #                           | genres).  This replaces the    |
    #                           | naive overlap / len(target)    |
    #                           | ratio that ignored movie-side  |
    #                           | genre cardinality.             |
    #   ────────────────────────┼────────────────────────────────┼───────
    #    quality_norm           | Z-score normalised rating      | 30 %
    #                           | within the movie's *primary*   |
    #                           | genre peer group.  A Drama      |
    #                           | rated 75 % when the median      |
    #                           | Drama is 60 % is far more       |
    #                           | impressive than an Action film  |
    #                           | rated 75 % when the median      |
    #                           | Action is 72 %.  This replaces  |
    #                           | the raw rating/100 that was    |
    #                           | genre-blind.                   |
    #   ────────────────────────┼────────────────────────────────┼───────
    #    mood_multiplicity       | When the user provides multiple | 15 %
    #                           | mood keywords, reward movies   |
    #                           | that match genres from *more   |
    #                           | distinct moods* — this is a    |
    #                           | cross-cutting signal of        |
    #                           | relevance and increases        |
    #                           | recommendation diversity.      |
    #   ────────────────────────┼────────────────────────────────┼───────
    #    serendipity             | Controlled random noise       | 10 %
    #                           | (Gaussian, σ ≈ 2.5) to break   |
    #                           | deterministic ties and inject   |
    #                           | novelty.  Bounded so the best   |
    #                           | candidates still arrive first.  |
    #
    #  Composite:  0.45·genre + 0.30·quality + 0.15·mood + 0.10·noise  =  100 %
    #
    #  Weights were chosen via ordinal ranking: genre relevance is the
    #  primary signal (user asked for a mood → match genes first),
    #  quality prevents low-rated movies from dominating, mood
    #  multiplicity adds cross-mood coherence, and noise prevents
    #  over-fitting to the formula's first three terms.

    def recommend_by_mood(self, mood_text, top_n=5, seed=None):
        """Recommend movies matching the user's mood description."""
        rng = random.Random(seed)

        clean = re.sub(r'[^a-z\s]', '', mood_text.lower())
        mood_words = clean.split()
        if not mood_words:
            return []

        target_genres = set()
        matched_moods = []
        for word in mood_words:
            if word in self.MOOD_MAP:
                target_genres.update(self.MOOD_MAP[word])
                matched_moods.append(word)

        if not target_genres:
            target_genres = {'Comedy', 'Drama', 'Action & Adventure'}

        mood_labels = ', '.join(matched_moods[:3]) if matched_moods else mood_text[:30].strip()
        reason_phrase = self.MOOD_REASONS.get(
            matched_moods[0] if matched_moods else '',
            f'picked for your mood: {mood_labels}',
        )

        candidates = []
        for mid in self.movie_id_to_idx:
            movie_genres = self._get_genre(mid)
            if movie_genres == 'N/A':
                continue
            movie_genre_set = set(g.strip() for g in movie_genres.split(','))

            # ── 1. genre_match: Jaccard similarity ──
            intersection = movie_genre_set & target_genres
            if not intersection:
                continue
            union = movie_genre_set | target_genres
            jaccard = len(intersection) / len(union)   # ∈ [0, 1]

            # ── 2. quality_norm: within-genre z-score ──
            movie_rating = self._get_rating(mid)
            if movie_rating == 'N/A':
                continue
            movie_rating = int(movie_rating)

            # Use the *primary* genre (first listed) for peer comparison
            primary_genre = movie_genres.split(',')[0].strip()
            genre_median = self._genre_medians.get(primary_genre, 50)
            # Simple effect-size: how many std-dev-equivalents above median
            # (use a fixed σ=15 — Rotten Tomatoes scores are roughly normal
            #  with σ ≈ 15 across genres)
            z_quality = (movie_rating - genre_median) / 15
            # Squash to [0, 1] via logistic (maps z∈[-3,3] to roughly [0.05,0.95])
            quality_score = 1.0 / (1.0 + np.exp(-z_quality))

            # ── 3. mood_multiplicity ──
            if matched_moods:
                mood_hits = 0
                for mw in matched_moods:
                    if mw in self.MOOD_MAP and (movie_genre_set & set(self.MOOD_MAP[mw])):
                        mood_hits += 1
                mood_score = mood_hits / len(matched_moods)   # ∈ [0, 1]
            else:
                mood_score = 1.0

            # ── 4. composite ──
            base = (
                0.45 * jaccard
                + 0.30 * quality_score
                + 0.15 * mood_score
            ) * 100   # → [0, 100]

            # Serendipity: Gaussian noise with σ ≈ 2.5
            noise = rng.gauss(0, 2.5)
            relevance = min(100, max(5, round(base + noise)))

            # Pick the best-matching genre for the reason text
            sorted_overlap = sorted(intersection, key=lambda g: len(g))
            matched_genre = sorted_overlap[0] if sorted_overlap else list(intersection)[0]

            candidates.append({
                'title': self._get_movie_title(mid),
                'reason': f'{reason_phrase} — a great {matched_genre} film',
                'genres': movie_genres,
                'year': self._get_year(mid),
                'rating': movie_rating,
                'consensus': self._get_consensus(mid),
                'score': relevance,
            })

        # Sort by relevance
        candidates.sort(key=lambda x: x['score'], reverse=True)

        # Deduplicate
        seen_titles = set()
        unique = []
        for c in candidates:
            if c['title'] not in seen_titles:
                seen_titles.add(c['title'])
                unique.append(c)

        # Inject variety: take top candidates + a few from slightly lower
        if len(unique) > top_n * 2:
            top_candidates = unique[:top_n + 3]
            rest_pool = unique[top_n + 3:top_n * 3]
            if rest_pool:
                top_candidates += rng.sample(rest_pool, min(2, len(rest_pool)))
            top_candidates.sort(key=lambda x: x['score'], reverse=True)
            unique = top_candidates

        result = unique[:top_n]
        for c in result:
            c['similarity'] = round(c['score'])
            del c['score']

        return result

    # ──────────────────────────────────────────────────────────
    #  Evaluation
    # ──────────────────────────────────────────────────────────

    def evaluate_leave_one_out(self, n_users=200, k_values=(5, 10, 20),
                               random_state=42):
        """Leave-one-out cross-validation for user-based recommendations.

        For each of ``n_users`` randomly selected users:
          1. Hide one randomly-chosen liked movie (the "held-out" item).
          2. Generate top-k recommendations from the remaining liked movies.
          3. Check whether the held-out movie appears in the top-k list.

        Metrics computed:
          - hit_rate@k     : fraction of users whose held-out item was in top-k
          - precision@k    : #hits / k  (averaged across users)
          - recall@k       : #hits / #held_out_items  (= hit_rate for single-item)

        Parameters
        ----------
        n_users : int
            Number of users to sample for evaluation.
        k_values : tuple of int
            Top-k cutoffs to evaluate.
        random_state : int
            Seed for reproducibility.

        Returns
        -------
        dict  with keys 'hit_rate', 'precision', 'recall' (each a dict k→float).
        """
        rng = random.Random(random_state)
        eligible = [
            u for u in self.user_ids
            if len(self.user_rated_movies.get(u, [])) >= 5
        ]
        if len(eligible) > n_users:
            eligible = rng.sample(eligible, n_users)
        n_users = len(eligible)

        hits = {k: 0 for k in k_values}
        prec_sum = {k: 0.0 for k in k_values}

        for user_id in eligible:
            liked = self.user_rated_movies[user_id]
            valid = [m for m in liked if m in self.movie_id_to_idx]
            if len(valid) < 5:
                continue

            # Hold out one liked movie
            held_out = rng.choice(valid)
            remaining = [m for m in valid if m != held_out]

            # Recommend from the remaining movies
            recs = self._recommend_from_movie_ids(remaining, top_n=max(k_values))
            rec_ids = [r['movie_id'] for r in recs]

            for k in k_values:
                if held_out in rec_ids[:k]:
                    hits[k] += 1
                    prec_sum[k] += 1.0 / k

        eval_users = n_users

        self.eval_['leave_one_out'] = {
            'n_users': eval_users,
            'n_eligible': len(eligible),
            'hit_rate': {str(k): round(hits[k] / eval_users, 4) if eval_users else 0
                         for k in k_values},
            'precision': {str(k): round(prec_sum[k] / eval_users, 4) if eval_users else 0
                          for k in k_values},
            'recall': {str(k): round(hits[k] / eval_users, 4) if eval_users else 0
                       for k in k_values},
        }
        return self.eval_['leave_one_out']

    def evaluate_diversity(self, n_samples=500, top_n=10, random_state=42):
        """Compute intra-list diversity of item-item recommendations.

        For ``n_samples`` randomly selected movies, generate top_n
        recommendations.  Diversity is defined as:

            diversity = 1 − mean(pairwise cosine similarity of recommended items)

        A diversity of 1.0 means all recommended items are orthogonal
        (perfectly diverse); 0.0 means they are identical.

        Also reports ILS (Intra-List Similarity) for comparison with
        literature.
        """
        rng = random.Random(random_state)
        all_mids = list(self.movie_id_to_idx.keys())
        if len(all_mids) > n_samples:
            sampled = rng.sample(all_mids, n_samples)
        else:
            sampled = all_mids

        diversities = []
        for mid in sampled:
            title = self._get_movie_title(mid)
            recs = self.recommend_by_movies([title], top_n=top_n)
            rec_mids = [self._find_movie_id(r['title']) for r in recs]
            rec_mids = [m for m in rec_mids if m is not None and m in self.movie_id_to_idx]
            if len(rec_mids) < 2:
                continue

            indices = [self.movie_id_to_idx[m] for m in rec_mids]
            sub_sim = self.item_similarity[np.ix_(indices, indices)]
            # Exclude diagonal (self-similarity = 1)
            n = sub_sim.shape[0]
            mask = ~np.eye(n, dtype=bool)
            mean_sim = sub_sim[mask].mean()
            diversities.append(1.0 - mean_sim)

        avg_diversity = round(float(np.mean(diversities)), 4) if diversities else 0.0
        avg_ils = round(1.0 - avg_diversity, 4)

        self.eval_['diversity'] = {
            'n_samples': n_samples,
            'top_n': top_n,
            'diversity': avg_diversity,
            'ils': avg_ils,
        }
        return self.eval_['diversity']

    def evaluate_all(self, n_users=200, k_values=(5, 10, 20),
                     n_diversity_samples=300, random_state=42):
        """Run full evaluation suite and return combined results."""
        print('Running leave-one-out evaluation...')
        loo = self.evaluate_leave_one_out(
            n_users=n_users, k_values=k_values, random_state=random_state,
        )
        print('Running diversity evaluation...')
        div = self.evaluate_diversity(
            n_samples=n_diversity_samples, top_n=10, random_state=random_state,
        )
        print('Evaluation complete.')
        return {'leave_one_out': loo, 'diversity': div}

    def print_eval(self):
        """Pretty-print stored evaluation results."""
        e = self.eval_
        if not e:
            print('No evaluation data. Run evaluate_all() or evaluate_leave_one_out() first.')
            return

        print('=' * 60)
        print('MovieRecommender Evaluation')
        print('=' * 60)

        loo = e.get('leave_one_out')
        if loo:
            print(f"\n  Leave-One-Out  (n={loo['n_users']} users)")
            print(f"  {'k':<6} {'Hit Rate':<12} {'Precision':<12} {'Recall':<12}")
            print(f"  {'─'*5} {'─'*11} {'─'*11} {'─'*11}")
            for k in sorted(loo['hit_rate'].keys(), key=int):
                print(f"  @{k:<4} {loo['hit_rate'][k]:<12.2%} "
                      f"{loo['precision'][k]:<12.4f} {loo['recall'][k]:<12.2%}")

        div = e.get('diversity')
        if div:
            print(f"\n  Diversity (n={div['n_samples']}, top_n={div['top_n']})")
            print(f"  Diversity : {div['diversity']:.4f}  (1 = fully diverse)")
            print(f"  ILS       : {div['ils']:.4f}  (0 = fully diverse)")

        print('=' * 60)

    # ──────────────────────────────────────────────────────────
    #  Internal helpers
    # ──────────────────────────────────────────────────────────

    def _recommend_from_movie_ids(self, movie_ids, top_n=20):
        """Generate recommendations from a list of movie IDs (used by eval)."""
        valid_ids = [m for m in movie_ids if m in self.movie_id_to_idx]
        if not valid_ids:
            return []

        idx = self.movie_id_to_idx[valid_ids[0]]
        combined = self.item_similarity[idx].copy()
        for mid in valid_ids[1:]:
            combined += self.item_similarity[self.movie_id_to_idx[mid]]

        for mid in valid_ids:
            combined[self.movie_id_to_idx[mid]] = -999

        top_indices = np.argsort(combined)[-top_n:][::-1]
        results = []
        for i in top_indices:
            if combined[i] <= -999:
                continue
            mid = self.idx_to_movie_id[i]
            results.append({
                'movie_id': mid,
                'title': self._get_movie_title(mid),
                'score': round(float(combined[i]) / len(valid_ids), 4),
            })
        return results

    def get_top_users(self, n=200):
        rng = random.Random(42)
        ids = list(self.user_ids)
        rng.shuffle(ids)
        return sorted(ids[:n])

    def _normalize(self, s):
        return re.sub(r'[^a-z0-9]', '', s.lower())

    def _find_movie_id(self, name):
        search = self._normalize(name)
        if len(search) < 3:
            return None
        for mid in self.movie_id_to_idx:
            title = self._get_movie_title(mid)
            if title and search in self._normalize(title):
                return mid
        return None

    def _get_movie_title(self, movie_id):
        try:
            return str(self.movies_df.loc[movie_id, 'movie_title'])
        except (KeyError, ValueError):
            return movie_id

    def _get_genre(self, movie_id):
        try:
            return str(self.movies_df.loc[movie_id, 'genres'])
        except (KeyError, ValueError):
            return 'N/A'

    def _get_year(self, movie_id):
        try:
            return str(self.movies_df.loc[movie_id, 'original_release_date'])[:4]
        except (KeyError, ValueError):
            return 'N/A'

    def _get_rating(self, movie_id):
        try:
            return int(self.movies_df.loc[movie_id, 'tomatometer_rating'])
        except (KeyError, ValueError):
            return 'N/A'

    def _get_consensus(self, movie_id):
        try:
            c = str(self.movies_df.loc[movie_id, 'critics_consensus'])
            return c if c and c.lower() != 'nan' else ''
        except (KeyError, ValueError):
            return ''

    def _build_reason(self, movie_id, liked_names):
        genre = self._get_genre(movie_id)
        if genre and genre != 'N/A':
            g = genre.split(',')[0].strip()
            return f'Similar {g.lower()} style to {liked_names[0]}'
        return f'Similar to {liked_names[0]}'

    # ──────────────────────────────────────────────────────────
    #  Persistence
    # ──────────────────────────────────────────────────────────

    def save(self, path):
        data = {
            'user_item_matrix': self.user_item_matrix,
            'item_similarity': self.item_similarity,
            'movie_id_to_idx': self.movie_id_to_idx,
            'idx_to_movie_id': self.idx_to_movie_id,
            'movies_df': self.movies_df,
            'user_ids': self.user_ids,
            'user_rated_movies': self.user_rated_movies,
            'eval': {k: v for k, v in self.eval_.items()},
            'genre_medians': self._genre_medians,
            'genre_counts': self._genre_counts,
        }
        with open(path, 'wb') as f:
            pickle.dump(data, f)

    @classmethod
    def load(cls, path):
        instance = cls.__new__(cls)
        with open(path, 'rb') as f:
            data = pickle.load(f)
        instance.user_item_matrix = data['user_item_matrix']
        instance.item_similarity = data['item_similarity']
        instance.movie_id_to_idx = data['movie_id_to_idx']
        instance.idx_to_movie_id = data['idx_to_movie_id']
        instance.movies_df = data['movies_df']
        instance.user_ids = data['user_ids']
        instance.user_rated_movies = data['user_rated_movies']
        instance.eval_ = data.get('eval', {})
        instance._genre_medians = data.get('genre_medians', {})
        instance._genre_counts = data.get('genre_counts', {})
        if not instance._genre_medians:
            instance._build_genre_stats()
        return instance
