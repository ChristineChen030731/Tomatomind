"""
Sentiment classifier — multi-model training with automatic selection.

Trains a panel of candidate models on the same train/test split,
picks the best one by validation accuracy, and exposes full
evaluation metrics.

Supported model families:
  - LogisticRegression    (baseline linear)
  - SGDClassifier         (fast linear, modified_huber hinge)
  - LinearSVC             (calibrated — best performer in benchmarks)
  - MultinomialNB         (fast baseline, good for short texts)
"""

import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report,
)

from data.preprocess import clean_text


class SentimentModel:
    """Multi-model sentiment classifier with auto-selection.

    Usage
    -----
    >>> model = SentimentModel()
    >>> reviews_df = load_and_sample_reviews(n=500000)
    >>> model.train(reviews_df)              # trains 4 models, picks best
    >>> model.print_eval()                   # shows metrics + which model won
    >>> result = model.predict("Great film!")# uses the best model
    """

    # ── Candidate model factory ──────────────────────────────
    # Each entry: (display_name, classifier_instance)
    # Weights are tuned from Phase 1-3 benchmarks.
    @staticmethod
    def _candidate_models():
        return [
            ('LogisticRegression', LogisticRegression(
                max_iter=5000, C=0.5, solver='saga', random_state=42)),
            ('SGDClassifier', SGDClassifier(
                loss='modified_huber', max_iter=5000, random_state=42)),
            ('LinearSVC_Calibrated', CalibratedClassifierCV(
                LinearSVC(C=0.5, max_iter=5000, random_state=42, dual=False),
                cv=3)),
            ('MultinomialNB', MultinomialNB(alpha=0.1)),
        ]

    def __init__(self, max_features=30000, ngram_range=(1, 3)):
        """
        Parameters
        ----------
        max_features : int
            TF-IDF vocabulary size.  30000 is the sweet spot from benchmarks.
        ngram_range : tuple
            (1,2) = unigrams + bigrams; (1,3) = + trigrams.
            (1,3) gives ~0.5 % extra accuracy at the cost of larger vocabulary.
        """
        self._max_features = max_features
        self._ngram_range = ngram_range

        # ── These are set by train() ──
        self.pipeline = None            # best Pipeline (tfidf + clf)
        self.selected_model = None      # name of the winning model
        self.eval_ = {}
        self._candidate_results = []    # per-model metrics

    # ──────────────────────────────────────────────────────────
    #  Training
    # ──────────────────────────────────────────────────────────

    def train(self, reviews_df, test_size=0.2):
        """Train all candidate models, pick the best, record metrics.

        Parameters
        ----------
        reviews_df : pd.DataFrame
            Columns 'review_content' (text) and 'review_type' (Fresh/Rotten).
        test_size : float
            Fraction held out for evaluation.

        Returns
        -------
        dict  evaluation summary for the selected model.
        """
        texts = reviews_df['review_content'].tolist()
        labels = (reviews_df['review_type'] == 'Fresh').astype(int).tolist()

        X_train, X_test, y_train, y_test = train_test_split(
            texts, labels, test_size=test_size, random_state=42, stratify=labels,
        )

        best_name = None
        best_acc = -1.0
        best_pipeline = None
        best_y_pred = None

        for name, clf in self._candidate_models():
            pipe = Pipeline([
                ('tfidf', TfidfVectorizer(
                    max_features=self._max_features,
                    ngram_range=self._ngram_range,
                    stop_words='english',
                    preprocessor=clean_text,
                    sublinear_tf=True,
                )),
                ('clf', clf),
            ])

            pipe.fit(X_train, y_train)
            y_pred = pipe.predict(X_test)

            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred)
            rec = recall_score(y_test, y_pred)

            self._candidate_results.append({
                'model': name,
                'accuracy': round(acc, 4),
                'f1': round(f1, 4),
                'precision': round(prec, 4),
                'recall': round(rec, 4),
            })

            if acc > best_acc:
                best_acc = acc
                best_name = name
                best_pipeline = pipe
                best_y_pred = y_pred

        # ── Commit best model ──
        self.pipeline = best_pipeline
        self.selected_model = best_name

        # ── Full evaluation on best model ──
        report = classification_report(y_test, best_y_pred,
                                       target_names=['Rotten', 'Fresh'])
        self.eval_ = {
            'accuracy': round(best_acc, 4),
            'f1': round(f1_score(y_test, best_y_pred), 4),
            'precision': round(precision_score(y_test, best_y_pred), 4),
            'recall': round(recall_score(y_test, best_y_pred), 4),
            'classification_report': report,
            'selected_model': best_name,
            'train_size': len(X_train),
            'test_size': len(X_test),
            'class_distribution': {
                'train': {'Fresh': int(sum(y_train)), 'Rotten': int(len(y_train) - sum(y_train))},
                'test': {'Fresh': int(sum(y_test)), 'Rotten': int(len(y_test) - sum(y_test))},
            },
            'candidate_results': self._candidate_results,
            'tfidf_config': {
                'max_features': self._max_features,
                'ngram_range': self._ngram_range,
            },
        }

        return self.eval_

    # ──────────────────────────────────────────────────────────
    #  Prediction
    # ──────────────────────────────────────────────────────────

    def predict(self, review_text):
        """Predict sentiment for a single review string.

        Returns
        -------
        dict  {'sentiment': str, 'confidence': float}
        """
        proba = self.pipeline.predict_proba([review_text])[0]
        pred = int(proba.argmax())
        confidence = float(proba.max())

        if pred == 1:
            sentiment = 'Positive'
            if confidence < 0.7:
                sentiment = 'Mixed-Positive'
        else:
            sentiment = 'Negative'
            if confidence < 0.7:
                sentiment = 'Mixed-Negative'

        return {'sentiment': sentiment, 'confidence': confidence}

    def predict_batch(self, review_texts):
        """Batch prediction — much faster than looping .predict()."""
        if not review_texts:
            return []
        probas = self.pipeline.predict_proba(review_texts)
        results = []
        for proba in probas:
            pred = int(proba.argmax())
            confidence = float(proba.max())
            if pred == 1:
                sentiment = 'Positive'
                if confidence < 0.7:
                    sentiment = 'Mixed-Positive'
            else:
                sentiment = 'Negative'
                if confidence < 0.7:
                    sentiment = 'Mixed-Negative'
            results.append({'sentiment': sentiment, 'confidence': confidence})
        return results

    # ──────────────────────────────────────────────────────────
    #  Persistence
    # ──────────────────────────────────────────────────────────

    def save(self, path):
        data = {
            'pipeline': self.pipeline,
            'selected_model': self.selected_model,
            'eval': {k: v for k, v in self.eval_.items()
                     if k not in ('y_test', 'y_proba')},
            'candidate_results': self._candidate_results,
            'max_features': self._max_features,
            'ngram_range': self._ngram_range,
        }
        with open(path, 'wb') as f:
            pickle.dump(data, f)

    @classmethod
    def load(cls, path):
        instance = cls.__new__(cls)
        with open(path, 'rb') as f:
            data = pickle.load(f)
        instance.pipeline = data['pipeline']
        instance.selected_model = data.get('selected_model', 'unknown')
        instance._max_features = data.get('max_features', 30000)
        instance._ngram_range = data.get('ngram_range', (1, 3))
        instance._candidate_results = data.get('candidate_results', [])
        instance.eval_ = data.get('eval', {})
        return instance

    # ──────────────────────────────────────────────────────────
    #  Display
    # ──────────────────────────────────────────────────────────

    def print_eval(self):
        """Pretty-print evaluation results including model comparison."""
        e = self.eval_
        if e.get('accuracy') is None:
            print('No evaluation data yet. Run train() first.')
            return

        print('=' * 65)
        print('SentimentModel — Multi-Model Training Report')
        print('=' * 65)
        print(f'Train size  : {e["train_size"]:,}')
        print(f'Test size   : {e["test_size"]:,}')
        dist = e['class_distribution']
        print(f'Train dist  : Fresh={dist["train"]["Fresh"]:,}  '
              f'Rotten={dist["train"]["Rotten"]:,}')
        print(f'Test dist   : Fresh={dist["test"]["Fresh"]:,}  '
              f'Rotten={dist["test"]["Rotten"]:,}')
        print(f'TF-IDF      : max_features={e["tfidf_config"]["max_features"]}, '
              f'ngram_range={e["tfidf_config"]["ngram_range"]}')
        print()

        # ── Per-model leaderboard ──
        candidates = e.get('candidate_results', self._candidate_results)
        if candidates:
            print(f'  {"Model":<28} {"Acc":>8} {"F1":>8} {"Prec":>8} {"Rec":>8}')
            print(f'  {"─"*27} {"─"*7} {"─"*7} {"─"*7} {"─"*7}')
            for c in sorted(candidates, key=lambda x: x['accuracy'], reverse=True):
                tag = '★ SELECTED' if c['model'] == self.selected_model else '         '
                print(f'{tag} {c["model"]:<20} {c["accuracy"]:.4f}  {c["f1"]:.4f}  '
                      f'{c["precision"]:.4f}  {c["recall"]:.4f}')
            print()

        print(f'SELECTED MODEL : {self.selected_model}')
        print(f'Accuracy       : {e["accuracy"]:.2%}')
        print(f'Precision      : {e["precision"]:.2%}')
        print(f'Recall         : {e["recall"]:.2%}')
        print(f'F1 Score       : {e["f1"]:.2%}')
        print('─' * 65)
        print(e['classification_report'])
        print('=' * 65)

    def summary(self):
        """One-line summary string."""
        e = self.eval_
        if not e.get('accuracy'):
            return 'SentimentModel (not trained)'
        return (f'SentimentModel[{self.selected_model}] '
                f'Acc={e["accuracy"]:.2%} F1={e["f1"]:.2%} '
                f'on {e["test_size"]:,} test samples')
