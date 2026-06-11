"""
Transformer sentiment classifier — BERT-based fine-tuning for RT reviews.

Matches the Colab notebook's TransformerSentimentModel API exactly.
Adds save() / load() for importing Colab-trained checkpoints into Flask.

Colab results (DistilBERT, 100k, 3 epochs, batch=16):
    Epoch 1  |  train_loss=0.3907  eval_loss=0.3183  acc=0.8592  f1=0.8895
    Epoch 2  |  train_loss=0.2541  eval_loss=0.3533  acc=0.8643  f1=0.8957  ← best
    Epoch 3  |  train_loss=0.1716  eval_loss=0.4478  acc=0.8665  f1=0.8949  (overfit)

Usage (local Flask app):
    model = TransformerSentimentModel.load('model/saved/transformer')
    result = model.predict("Great film!")
"""

import os
import re
import pickle
import time
import numpy as np

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
from sklearn.model_selection import train_test_split

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
#  Pre-tokenization helper  (same as Colab)
# ──────────────────────────────────────────────────────────────

def _pretokenize(texts, tokenizer, max_length, desc='Tokenizing'):
    input_ids_list, attention_mask_list = [], []
    n = len(texts)
    batch_size = 4096
    for i in range(0, n, batch_size):
        batch_texts = [str(t)[:4000] for t in texts[i:i + batch_size]]
        enc = tokenizer(
            batch_texts, max_length=max_length, padding='max_length',
            truncation=True, return_tensors='pt',
        )
        input_ids_list.append(enc['input_ids'])
        attention_mask_list.append(enc['attention_mask'])
        pct = min(100, (i + batch_size) / n * 100)
        print(f'  {desc}: {pct:.0f}%', end='\r', flush=True)
    print(f'  {desc}: 100%  ({n:,}/{n:,})', flush=True)
    return torch.cat(input_ids_list, dim=0), torch.cat(attention_mask_list, dim=0)


# ──────────────────────────────────────────────────────────────
#  TransformerSentimentModel  (matches Colab Cell 8)
# ──────────────────────────────────────────────────────────────

class TransformerSentimentModel:

    def __init__(self, model_name='distilbert'):
        self.model_name = model_name
        self.config = MODEL_OPTIONS[model_name]
        self.hf_name = self.config['hf_name']
        self.tokenizer = None
        self.model = None
        self.device = None
        self.eval_ = {}

    def _get_device(self):
        if torch.backends.mps.is_available():
            return torch.device('mps')
        elif torch.cuda.is_available():
            return torch.device('cuda')
        return torch.device('cpu')

    # ── Training  (same as Colab Cell 8) ─────────────────────

    def train(self, reviews_df, test_size=0.2, batch_size=16, epochs=3,
              learning_rate=2e-5, max_length=256, random_state=42):

        self.device = self._get_device()
        print(f'\n{"="*60}')
        print(f'  Transformer: {self.config["name"]}  |  Device: {self.device}')
        print(f'{"="*60}', flush=True)

        texts = reviews_df['review_content'].tolist()
        labels = (reviews_df['review_type'] == 'Fresh').astype(int).tolist()
        X_train, X_test, y_train, y_test = train_test_split(
            texts, labels, test_size=test_size, random_state=random_state,
            stratify=labels,
        )
        print(f'  Train={len(X_train):,}  Test={len(X_test):,}  '
              f'Fresh={sum(y_train)/len(y_train):.1%}', flush=True)

        print(f'  Loading {self.hf_name} ...', flush=True)
        self.tokenizer = AutoTokenizer.from_pretrained(self.hf_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.hf_name, num_labels=2,
        )
        self.model.to(self.device)

        # Pre-tokenize
        t_tok = time.time()
        train_ids, train_mask = _pretokenize(
            X_train, self.tokenizer, max_length, 'Train token')
        test_ids, test_mask = _pretokenize(
            X_test, self.tokenizer, max_length, 'Test token ')
        print(f'  Tokenization: {time.time()-t_tok:.0f}s', flush=True)

        train_ds = TensorDataset(
            train_ids, train_mask, torch.tensor(y_train, dtype=torch.long))
        test_ds = TensorDataset(
            test_ids, test_mask, torch.tensor(y_test, dtype=torch.long))
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=batch_size * 2, shuffle=False)

        optimizer = AdamW(self.model.parameters(), lr=learning_rate)
        total_steps = len(train_loader) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(total_steps * 0.1),
            num_training_steps=total_steps,
        )
        loss_fn = nn.CrossEntropyLoss()

        # Training loop
        best_acc = 0.0
        best_state_dict = None

        for epoch in range(1, epochs + 1):
            t0 = time.time()
            self.model.train()
            train_loss = 0.0
            n_batches = len(train_loader)

            for batch_idx, (input_ids, attn, lbls) in enumerate(train_loader):
                input_ids = input_ids.to(self.device)
                attn = attn.to(self.device)
                lbls = lbls.to(self.device)

                optimizer.zero_grad()
                loss = loss_fn(
                    self.model(input_ids, attention_mask=attn).logits, lbls)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                train_loss += loss.item()

                if (batch_idx + 1) % max(1, n_batches // 4) == 0:
                    print(f'    Epoch {epoch} [{(batch_idx+1)/n_batches*100:.0f}%]  '
                          f'loss={loss.item():.4f}', end='\r', flush=True)

            # ── Eval after each epoch ──
            self.model.eval()
            all_preds, all_labels, eval_loss = [], [], 0.0
            with torch.no_grad():
                for input_ids, attn, lbls in test_loader:
                    input_ids = input_ids.to(self.device)
                    attn = attn.to(self.device)
                    lbls = lbls.to(self.device)
                    outputs = self.model(input_ids, attention_mask=attn)
                    eval_loss += loss_fn(outputs.logits, lbls).item()
                    all_preds.extend(
                        torch.argmax(outputs.logits, dim=1).cpu().tolist())
                    all_labels.extend(lbls.cpu().tolist())

            acc = accuracy_score(all_labels, all_preds)
            f1 = f1_score(all_labels, all_preds)

            if acc > best_acc:
                best_acc = acc
                self.eval_['best_epoch'] = epoch
                # Snapshot the best model weights
                best_state_dict = {
                    k: v.cpu().clone() for k, v in self.model.state_dict().items()
                }

            print(f'    Epoch {epoch} done  |  '
                  f'train_loss={train_loss/n_batches:.4f}  '
                  f'eval_loss={eval_loss/len(test_loader):.4f}  '
                  f'acc={acc:.4f}  f1={f1:.4f}  [{time.time()-t0:.0f}s]',
                  flush=True)

        # ── Restore best checkpoint ──
        if best_state_dict is not None:
            self.model.load_state_dict(best_state_dict)
            print(f'  Restored best model from epoch {self.eval_.get("best_epoch", "?")}')

        # ── Final evaluation ──
        self.model.eval()
        all_preds, all_probas = [], []
        with torch.no_grad():
            for input_ids, attn, _ in test_loader:
                input_ids = input_ids.to(self.device)
                attn = attn.to(self.device)
                outputs = self.model(input_ids, attention_mask=attn)
                all_probas.extend(
                    torch.softmax(outputs.logits, dim=1).cpu().tolist())
                all_preds.extend(
                    torch.argmax(outputs.logits, dim=1).cpu().tolist())

        self.eval_.update({
            'model': self.config['name'],
            'hf_name': self.hf_name,
            'accuracy': round(accuracy_score(y_test, all_preds), 4),
            'f1': round(f1_score(y_test, all_preds), 4),
            'precision': round(precision_score(y_test, all_preds), 4),
            'recall': round(recall_score(y_test, all_preds), 4),
            'classification_report': classification_report(
                y_test, all_preds, target_names=['Rotten', 'Fresh']),
            'train_size': len(X_train),
            'test_size': len(X_test),
            'class_distribution': {
                'train': {
                    'Fresh': int(sum(y_train)),
                    'Rotten': int(len(y_train) - sum(y_train)),
                },
                'test': {
                    'Fresh': int(sum(y_test)),
                    'Rotten': int(len(y_test) - sum(y_test)),
                },
            },
            'hyperparams': {
                'batch_size': batch_size,
                'epochs': epochs,
                'learning_rate': learning_rate,
                'max_length': max_length,
            },
        })
        return self.eval_

    # ── Prediction ───────────────────────────────────────────

    def predict(self, review_text):
        """Single-review prediction."""
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
            proba = torch.softmax(
                self.model(input_ids, attention_mask=attention_mask).logits,
                dim=1,
            )[0]
        pred = int(torch.argmax(proba))
        confidence = float(proba.max())
        if pred == 1:
            sentiment = 'Positive' if confidence >= 0.7 else 'Mixed-Positive'
        else:
            sentiment = 'Negative' if confidence >= 0.7 else 'Mixed-Negative'
        return {'sentiment': sentiment, 'confidence': confidence}

    def predict_batch(self, review_texts, batch_size=64):
        """Batch prediction — 10-50× faster than calling predict() in a loop.

        Parameters
        ----------
        review_texts : list of str
        batch_size : int
            GPU batch size (64 is safe for most GPUs; reduce if OOM).

        Returns
        -------
        list of dict  [{'sentiment': str, 'confidence': float}, ...]
        """
        if not review_texts:
            return []

        self.model.eval()
        results = []

        for i in range(0, len(review_texts), batch_size):
            batch = [str(t)[:2000] for t in review_texts[i:i + batch_size]]
            enc = self.tokenizer(
                batch, max_length=256, padding='max_length',
                truncation=True, return_tensors='pt',
            )
            input_ids = enc['input_ids'].to(self.device)
            attn_mask = enc['attention_mask'].to(self.device)

            with torch.no_grad():
                probas = torch.softmax(
                    self.model(input_ids, attention_mask=attn_mask).logits,
                    dim=1,
                ).cpu().numpy()

            for proba in probas:
                pred = int(proba.argmax())
                confidence = float(proba.max())
                if pred == 1:
                    sentiment = 'Positive' if confidence >= 0.7 else 'Mixed-Positive'
                else:
                    sentiment = 'Negative' if confidence >= 0.7 else 'Mixed-Negative'
                results.append({'sentiment': sentiment, 'confidence': confidence})

        return results

    # ──────────────────────────────────────────────────────────
    #  Save / Load  (for Colab → local transfer)
    # ──────────────────────────────────────────────────────────

    def save(self, path_dir):
        """Save tokenizer + model weights + eval metadata to a directory.

        Produces:
            {path_dir}/tokenizer/    — HuggingFace tokenizer
            {path_dir}/model/        — HuggingFace model weights
            {path_dir}/eval.pkl      — evaluation + config dict
        """
        os.makedirs(path_dir, exist_ok=True)
        tokenizer_path = os.path.join(path_dir, 'tokenizer')
        model_path = os.path.join(path_dir, 'model')
        eval_path = os.path.join(path_dir, 'eval.pkl')

        self.tokenizer.save_pretrained(tokenizer_path)
        self.model.save_pretrained(model_path)

        with open(eval_path, 'wb') as f:
            pickle.dump({
                'eval': self.eval_,
                'model_name': self.model_name,
                'hf_name': self.hf_name,
                'config_name': self.config['name'],
            }, f)

        print(f'Transformer model saved to {path_dir}/')
        print(f'  tokenizer/  — tokenizer files')
        print(f'  model/      — model weights')
        print(f'  eval.pkl    — evaluation metadata')

    @classmethod
    def load(cls, path_dir):
        """Load a saved transformer model from a directory.

        Usage:
            model = TransformerSentimentModel.load('model/saved/transformer')
        """
        import warnings
        warnings.filterwarnings('ignore')

        tokenizer_path = os.path.join(path_dir, 'tokenizer')
        model_path = os.path.join(path_dir, 'model')
        eval_path = os.path.join(path_dir, 'eval.pkl')

        # ── Load metadata ──
        with open(eval_path, 'rb') as f:
            meta = pickle.load(f)

        instance = cls.__new__(cls)
        instance.model_name = meta.get('model_name', 'distilbert')
        instance.config = MODEL_OPTIONS.get(
            instance.model_name, MODEL_OPTIONS['distilbert'])
        instance.hf_name = meta.get('hf_name', instance.config['hf_name'])
        instance.eval_ = meta.get('eval', {})
        instance.device = instance._get_device()

        # ── Load tokenizer & model weights ──
        instance.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        instance.model = AutoModelForSequenceClassification.from_pretrained(
            model_path)
        instance.model.to(instance.device)
        instance.model.eval()

        print(f'Loaded transformer model from {path_dir}/')
        print(f'  {instance.summary()}')
        return instance

    # ── Display ──────────────────────────────────────────────

    def print_eval(self):
        e = self.eval_
        if not e.get('accuracy'):
            print('No evaluation data. Run train() first.')
            return
        print(f'\n{"="*60}')
        print(f'Transformer: {e["model"]}  |  '
              f'Acc={e["accuracy"]:.2%}  F1={e["f1"]:.2%}')
        print(f'Train={e["train_size"]:,}  Test={e["test_size"]:,}')
        hp = e.get('hyperparams', {})
        print(f'epochs={hp.get("epochs")}  bs={hp.get("batch_size")}  '
              f'lr={hp.get("learning_rate")}')
        print(f'{"─"*60}\n{e["classification_report"]}{"="*60}')

    def summary(self):
        e = self.eval_
        if not e.get('accuracy'):
            return (f'Transformer[{e.get("model", self.model_name)}] '
                    f'(loaded — eval data missing, but model is ready)')
        return (f'Transformer[{e["model"]}] '
                f'Acc={e["accuracy"]:.2%} F1={e["f1"]:.2%} '
                f'on {e["test_size"]:,} test samples')
