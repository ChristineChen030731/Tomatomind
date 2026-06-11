"""
Transformer-based sentiment classifier — fine-tunes DistilBERT / RoBERTa.

Key design choice — pre-tokenization:
  Tokenization happens ONCE before training (not in Dataset.__getitem__).
  On M1 with 80k–400k texts this saves hours of repeated tokenization and
  keeps the GPU fed.  Tokenized tensors are stored in a simple TensorDataset.

Benchmark targets:
  - TF-IDF + LinearSVC @ 500k  →  80.77 % accuracy
  - DistilBERT @ 100k           →  expected 84–87 %
  - DistilBERT @ 500k           →  expected 86–90 %
  - RoBERTa-base @ 500k         →  expected 87–92 %

Hardware: Apple M1 with MPS acceleration.
"""

import os
import sys
import pickle
import time
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import AdamW
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)

from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report,
)


# ──────────────────────────────────────────────────────────────
#  Pre-tokenization helper
# ──────────────────────────────────────────────────────────────

def _pretokenize(texts, tokenizer, max_length, desc='Tokenizing'):
    """Tokenize a list of texts and return stacked tensors.

    Runs a single pass through all texts, so it can take 1-2 minutes
    for 100k texts on CPU — but saves 20-30 minutes of repeated
    tokenization during training.
    """
    input_ids_list = []
    attention_mask_list = []
    n = len(texts)

    # Use batched tokenization for speed (hundreds of texts per call)
    batch_size = 4096
    for i in range(0, n, batch_size):
        batch_texts = [str(t)[:4000] for t in texts[i:i + batch_size]]
        enc = tokenizer(
            batch_texts,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )
        input_ids_list.append(enc['input_ids'])
        attention_mask_list.append(enc['attention_mask'])

        pct = min(100, (i + batch_size) / n * 100)
        print(f'  {desc}: {pct:.0f}%  ({min(i + batch_size, n):,}/{n:,})',
              end='\r', flush=True)

    print(f'  {desc}: 100%  ({n:,}/{n:,})           ', flush=True)

    return (
        torch.cat(input_ids_list, dim=0),
        torch.cat(attention_mask_list, dim=0),
    )


# ──────────────────────────────────────────────────────────────
#  Model registry
# ──────────────────────────────────────────────────────────────

MODEL_OPTIONS = {
    'distilbert': {
        'name': 'DistilBERT base (uncased)',
        'hf_name': 'distilbert/distilbert-base-uncased',
        'description': '66M params — best speed / accuracy tradeoff',
    },
    'roberta': {
        'name': 'RoBERTa base',
        'hf_name': 'FacebookAI/roberta-base',
        'description': '125M params — higher accuracy, slower training',
    },
}


# ──────────────────────────────────────────────────────────────
#  Transformer Sentiment Classifier
# ──────────────────────────────────────────────────────────────

class TransformerSentimentModel:
    """Fine-tuned transformer for movie review sentiment analysis."""

    def __init__(self, model_name='distilbert'):
        if model_name not in MODEL_OPTIONS:
            raise ValueError(
                f'Unknown model "{model_name}". Choose from {list(MODEL_OPTIONS)}')
        self.model_name = model_name
        self.config = MODEL_OPTIONS[model_name]
        self.hf_name = self.config['hf_name']

        self.tokenizer = None
        self.model = None
        self.device = None
        self.eval_ = {}

    # ── device selection ─────────────────────────────────────

    def _get_device(self):
        if torch.backends.mps.is_available():
            return torch.device('mps')
        elif torch.cuda.is_available():
            return torch.device('cuda')
        return torch.device('cpu')

    # ── training ─────────────────────────────────────────────
    # Hyperparameters from the fine-tuning literature
    # (Devlin et al. 2019, Liu et al. 2019, Dodge et al. 2020):
    #   lr=2e-5      standard for BERT-family models
    #   batch_size=16 fits M1 unified memory comfortably
    #   epochs=3     sweet spot — 2 underfits, 4+ overfits on this scale
    #   warmup=10%   avoids catastrophic forgetting in early steps
    #   max_length=256  covers 95 % of Rotten Tomatoes reviews

    def train(
        self,
        reviews_df,
        test_size=0.2,
        batch_size=16,
        epochs=3,
        learning_rate=2e-5,
        max_length=256,
        random_state=42,
    ):
        """Fine-tune the transformer on review data.

        Pre-tokenizes all texts before training to avoid the
        classic __getitem__-tokenization bottleneck.
        """
        self.device = self._get_device()
        print(f'\n{"="*65}', flush=True)
        print(f'  Transformer Fine-tuning: {self.config["name"]}')
        print(f'  Device: {self.device}  |  {self.config["description"]}')
        print(f'{"="*65}', flush=True)

        # ── Train / test split ──
        from sklearn.model_selection import train_test_split
        texts = reviews_df['review_content'].tolist()
        labels = (reviews_df['review_type'] == 'Fresh').astype(int).tolist()

        X_train, X_test, y_train, y_test = train_test_split(
            texts, labels, test_size=test_size,
            random_state=random_state, stratify=labels,
        )
        print(f'  Train: {len(X_train):,}   Test: {len(X_test):,}')
        print(f'  Fresh ratio: {sum(y_train)/len(y_train):.1%}', flush=True)

        # ── Load tokenizer & model ──
        print(f'  Loading {self.hf_name} ...', flush=True)
        self.tokenizer = AutoTokenizer.from_pretrained(self.hf_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.hf_name, num_labels=2,
        )
        self.model.to(self.device)

        # ── PRE-TOKENIZE (the critical optimisation) ──
        t_tok = time.time()
        print(f'  Pre-tokenizing {len(X_train):,} training texts ...', flush=True)
        train_ids, train_mask = _pretokenize(X_train, self.tokenizer, max_length,
                                              desc='  Tokenizing train')

        print(f'  Pre-tokenizing {len(X_test):,} test texts ...', flush=True)
        test_ids, test_mask = _pretokenize(X_test, self.tokenizer, max_length,
                                            desc='  Tokenizing test ')
        print(f'  Tokenization done in {time.time() - t_tok:.0f}s', flush=True)

        # ── PyTorch datasets (tensors only — no __getitem__ tokenization) ──
        train_labels = torch.tensor(y_train, dtype=torch.long)
        test_labels = torch.tensor(y_test, dtype=torch.long)

        train_dataset = TensorDataset(train_ids, train_mask, train_labels)
        test_dataset = TensorDataset(test_ids, test_mask, test_labels)

        train_loader = DataLoader(train_dataset, batch_size=batch_size,
                                  shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size * 2,
                                 shuffle=False)

        # ── Optimizer & scheduler ──
        optimizer = AdamW(self.model.parameters(), lr=learning_rate)
        total_steps = len(train_loader) * epochs
        warmup_steps = int(total_steps * 0.1)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )
        loss_fn = nn.CrossEntropyLoss()

        # ── Training loop ──
        best_acc = 0.0
        for epoch in range(1, epochs + 1):
            t0 = time.time()
            self.model.train()
            train_loss = 0.0
            n_batches = len(train_loader)
            report_every = max(1, n_batches // 4)

            for batch_idx, (input_ids, attention_mask, lbls) in enumerate(train_loader):
                input_ids = input_ids.to(self.device)
                attention_mask = attention_mask.to(self.device)
                lbls = lbls.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(input_ids, attention_mask=attention_mask)
                loss = loss_fn(outputs.logits, lbls)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

                train_loss += loss.item()
                if (batch_idx + 1) % report_every == 0:
                    pct = (batch_idx + 1) / n_batches * 100
                    print(f'    Epoch {epoch} [{pct:.0f}%]  '
                          f'loss={loss.item():.4f}  '
                          f'lr={scheduler.get_last_lr()[0]:.2e}',
                          end='\r', flush=True)
                elif batch_idx == n_batches - 1:
                    print(f'    Epoch {epoch} [100%]  '
                          f'loss={loss.item():.4f}  '
                          f'lr={scheduler.get_last_lr()[0]:.2e}',
                          flush=True)

            # ── Evaluation after each epoch ──
            self.model.eval()
            all_preds, all_labels = [], []
            eval_loss = 0.0
            with torch.no_grad():
                for input_ids, attention_mask, lbls in test_loader:
                    input_ids = input_ids.to(self.device)
                    attention_mask = attention_mask.to(self.device)
                    lbls = lbls.to(self.device)
                    outputs = self.model(input_ids, attention_mask=attention_mask)
                    eval_loss += loss_fn(outputs.logits, lbls).item()
                    preds = torch.argmax(outputs.logits, dim=1)
                    all_preds.extend(preds.cpu().tolist())
                    all_labels.extend(lbls.cpu().tolist())

            avg_train_loss = train_loss / n_batches
            avg_eval_loss = eval_loss / len(test_loader)
            acc = accuracy_score(all_labels, all_preds)
            f1 = f1_score(all_labels, all_preds)
            epoch_time = time.time() - t0

            if acc > best_acc:
                best_acc = acc
                self.eval_['best_epoch'] = epoch

            print(f'    Epoch {epoch} done  |  '
                  f'train_loss={avg_train_loss:.4f}  eval_loss={avg_eval_loss:.4f}  '
                  f'acc={acc:.4f}  f1={f1:.4f}  [{epoch_time:.0f}s]', flush=True)

        # ── Final evaluation ──
        self.model.eval()
        all_preds, all_probas = [], []
        with torch.no_grad():
            for input_ids, attention_mask, _ in test_loader:
                input_ids = input_ids.to(self.device)
                attention_mask = attention_mask.to(self.device)
                outputs = self.model(input_ids, attention_mask=attention_mask)
                probas = torch.softmax(outputs.logits, dim=1)
                preds = torch.argmax(outputs.logits, dim=1)
                all_preds.extend(preds.cpu().tolist())
                all_probas.extend(probas.cpu().tolist())

        acc = accuracy_score(y_test, all_preds)
        f1 = f1_score(y_test, all_preds)
        prec = precision_score(y_test, all_preds)
        rec = recall_score(y_test, all_preds)
        report = classification_report(
            y_test, all_preds, target_names=['Rotten', 'Fresh'])

        self.eval_ = {
            'model': self.config['name'],
            'hf_name': self.hf_name,
            'accuracy': round(acc, 4),
            'f1': round(f1, 4),
            'precision': round(prec, 4),
            'recall': round(rec, 4),
            'classification_report': report,
            'train_size': len(X_train),
            'test_size': len(X_test),
            'class_distribution': {
                'train': {'Fresh': int(sum(y_train)),
                          'Rotten': int(len(y_train) - sum(y_train))},
                'test': {'Fresh': int(sum(y_test)),
                         'Rotten': int(len(y_test) - sum(y_test))},
            },
            'hyperparams': {
                'batch_size': batch_size,
                'epochs': epochs,
                'learning_rate': learning_rate,
                'max_length': max_length,
            },
            'y_test': y_test,
            'y_proba': [p[1] for p in all_probas],
        }

        return self.eval_

    # ── inference ────────────────────────────────────────────

    def predict(self, review_text):
        """Predict sentiment for a single review string."""
        if self.tokenizer is None or self.model is None:
            raise RuntimeError(
                'Model not trained or loaded. Call train() or load() first.')

        self.model.eval()
        encoding = self.tokenizer(
            str(review_text)[:2000],
            max_length=256,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)

        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask=attention_mask)
            proba = torch.softmax(outputs.logits, dim=1)[0]
            pred = int(torch.argmax(proba))
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

    # ── persistence ───────────────────────────────────────────

    def save(self, save_dir):
        """Save tokenizer, model, and evaluation metadata."""
        os.makedirs(save_dir, exist_ok=True)
        self.tokenizer.save_pretrained(save_dir)
        self.model.save_pretrained(save_dir)
        meta = {k: v for k, v in self.eval_.items()
                if k not in ('y_test', 'y_proba')}
        meta['model_name'] = self.model_name
        meta['hf_name'] = self.hf_name
        with open(os.path.join(save_dir, 'eval_meta.pkl'), 'wb') as f:
            pickle.dump(meta, f)
        print(f'Saved transformer model to {save_dir}', flush=True)

    @classmethod
    def load(cls, save_dir):
        """Load a previously saved transformer model."""
        instance = cls.__new__(cls)
        with open(os.path.join(save_dir, 'eval_meta.pkl'), 'rb') as f:
            meta = pickle.load(f)
        instance.model_name = meta.get('model_name', 'distilbert')
        instance.hf_name = meta.get(
            'hf_name', MODEL_OPTIONS[instance.model_name]['hf_name'])
        instance.config = MODEL_OPTIONS.get(
            instance.model_name, MODEL_OPTIONS['distilbert'])
        instance.device = instance._get_device()

        instance.tokenizer = AutoTokenizer.from_pretrained(save_dir)
        instance.model = AutoModelForSequenceClassification.from_pretrained(
            save_dir)
        instance.model.to(instance.device)
        instance.model.eval()
        instance.eval_ = {k: v for k, v in meta.items()
                          if k not in ('model_name', 'hf_name')}
        print(f'Loaded transformer from {save_dir}: {instance.summary()}',
              flush=True)
        return instance

    # ── display ────────────────────────────────────────────────

    def print_eval(self):
        e = self.eval_
        if not e.get('accuracy'):
            print('No evaluation data.')
            return
        print('\n' + '=' * 65)
        print(f'Transformer Model Evaluation — {e.get("model", "?")}')
        print('=' * 65)
        print(f'Train size : {e["train_size"]:,}')
        print(f'Test size  : {e["test_size"]:,}')
        dist = e['class_distribution']
        print(f'Train dist : Fresh={dist["train"]["Fresh"]:,}  '
              f'Rotten={dist["train"]["Rotten"]:,}')
        print(f'Test dist  : Fresh={dist["test"]["Fresh"]:,}  '
              f'Rotten={dist["test"]["Rotten"]:,}')
        hp = e.get('hyperparams', {})
        print(f'Hyperparams: epochs={hp.get("epochs")}  '
              f'bs={hp.get("batch_size")}  lr={hp.get("learning_rate")}  '
              f'max_len={hp.get("max_length")}')
        print(f'Accuracy   : {e["accuracy"]:.2%}')
        print(f'Precision  : {e["precision"]:.2%}')
        print(f'Recall     : {e["recall"]:.2%}')
        print(f'F1 Score   : {e["f1"]:.2%}')
        print('─' * 65)
        print(e['classification_report'])
        print('=' * 65, flush=True)

    def summary(self):
        e = self.eval_
        if not e.get('accuracy'):
            return 'TransformerSentimentModel (not trained)'
        return (f'Transformer[{e.get("model", "?")}] '
                f'Acc={e["accuracy"]:.2%} F1={e["f1"]:.2%} '
                f'on {e["test_size"]:,} test samples')


# ──────────────────────────────────────────────────────────────
#  Quick-launch entry point
# ──────────────────────────────────────────────────────────────

def train_and_compare(model_type='distilbert', n_samples=500_000, epochs=3):
    """Train a transformer model and print comparison vs TF-IDF baseline."""
    from data.loader import load_and_sample_reviews

    print(f'\nLoading {n_samples:,} reviews ...', flush=True)
    df = load_and_sample_reviews(n=n_samples, random_state=42, force=True)
    print(f'  Loaded: {len(df):,} reviews', flush=True)

    model = TransformerSentimentModel(model_name=model_type)
    t0 = time.time()
    model.train(reviews_df=df, test_size=0.2, epochs=epochs)
    elapsed = time.time() - t0

    model.print_eval()
    print(f'\nTotal training time: {elapsed:.0f}s ({elapsed/60:.1f} min)')

    # Compare with TF-IDF baseline
    tfidf_baseline = {
        100_000: 0.7902,
        300_000: 0.8007,
        500_000: 0.8077,
    }
    base = tfidf_baseline.get(n_samples)
    if base is not None:
        delta = model.eval_['accuracy'] - base
        print(f'\n{"="*65}')
        print(f'  COMPARISON @ {n_samples//1000}k samples')
        print(f'{"="*65}')
        print(f'  TF-IDF + LinearSVC               : {base:.2%}')
        print(f'  {model.eval_["model"]:<35} : {model.eval_["accuracy"]:.2%}')
        print(f'  {"Δ":<35} : {delta:+.2%}')
        if delta > 0:
            print(f'  ✓ Transformer wins by {delta*100:.1f} percentage points')

    return model
