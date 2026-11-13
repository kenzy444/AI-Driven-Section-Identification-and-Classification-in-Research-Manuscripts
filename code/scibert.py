#!/usr/bin/env python3
"""SciBERT-only multi-label baseline for page-level section classification.

Fine-tunes ``allenai/scibert_scivocab_uncased`` with a linear multi-label head.
No handcrafted rules are used: the page text is the only input.

Protocol
--------
    fit on train -> early stop on val loss -> tune per-label thresholds on val
    -> score test once -> report metrics + article-level bootstrap CIs

Usage
-----
    python code/scibert_baseline.py
    python code/scibert_baseline.py --epochs 3 --batch-size 8
    python code/scibert_baseline.py --max-length 256 --batch-size 16
    python code/scibert_baseline.py --limit-train 32 --limit-val 8 --limit-test 8 \
        --epochs 1 --max-length 64 --bootstrap 0      # smoke test

Requires: torch, transformers, scikit-learn, numpy.
SciBERT is limited to 512 tokens; longer pages are truncated.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# Import torch before scikit-learn.  On Windows, loading sklearn first can make
# torch's c10.dll fail to initialise (WinError 1114).
try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, Dataset
    from transformers import (
        AutoModel,
        AutoTokenizer,
        DataCollatorWithPadding,
        get_linear_schedule_with_warmup,
    )
except ImportError as exc:
    raise SystemExit(
        "SciBERT dependencies are missing. Install them with:\n"
        "  pip install torch transformers scikit-learn numpy"
    ) from exc

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    hamming_loss,
    precision_recall_curve,
    precision_recall_fscore_support,
)

LABELS = [
    "title",
    "abstract",
    "keywords",
    "introduction",
    "background",
    "objectives",
    "analysis",
    "study_design",
    "materials",
    "results",
    "method",
    "experiments",
    "evaluation",
    "discussion",
    "conclusion",
    "references",
    "declarations",
    "supplementary",
]
LABEL_INDEX = {name: i for i, name in enumerate(LABELS)}

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent if (HERE.parent / "data").exists() else HERE
DEFAULT_SPLITS = ROOT / "data" / "splits"
DEFAULT_OUT = ROOT / "results" / "scibert_baseline"


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------
def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if not rows:
        raise SystemExit(f"no rows found in {path}")
    return rows


def label_matrix(rows: list[dict]) -> np.ndarray:
    y = np.zeros((len(rows), len(LABELS)), dtype=np.int8)
    for i, row in enumerate(rows):
        stored = row.get("y")
        if isinstance(stored, list) and len(stored) == len(LABELS):
            y[i] = np.asarray(stored, dtype=np.int8)
            continue
        for name in row.get("labels_covered", []):
            j = LABEL_INDEX.get(name)
            if j is not None:
                y[i, j] = 1
    return y


def page_texts(rows: list[dict]) -> list[str]:
    return [row.get("text") or "" for row in rows]


# ---------------------------------------------------------------------------
# thresholds / metrics
# ---------------------------------------------------------------------------
def tune_thresholds(
    y_true: np.ndarray, proba: np.ndarray, min_support: int
) -> tuple[np.ndarray, list[dict]]:
    """Choose a per-label F1 threshold; skip labels with too few val positives."""
    thresholds = np.full(y_true.shape[1], 0.5, dtype=np.float64)
    stats: list[dict] = []

    for j, name in enumerate(LABELS):
        support = int(y_true[:, j].sum())
        if support < min_support:
            stats.append(
                {
                    "label": name,
                    "val_support": support,
                    "tuned": False,
                    "threshold": 0.5,
                }
            )
            continue

        precision, recall, candidates = precision_recall_curve(
            y_true[:, j], proba[:, j]
        )
        denom = precision[:-1] + recall[:-1]
        f1 = np.divide(
            2.0 * precision[:-1] * recall[:-1],
            denom,
            out=np.zeros(len(candidates)),
            where=denom > 0,
        )
        if len(candidates) == 0 or f1.max() <= 0:
            best_t, best_f1 = 0.5, 0.0
        else:
            k = int(np.argmax(f1))
            best_t = float(np.clip(candidates[k], 1e-3, 1.0))
            best_f1 = float(f1[k])

        thresholds[j] = best_t
        stats.append(
            {
                "label": name,
                "val_support": support,
                "tuned": True,
                "threshold": best_t,
                "val_f1": best_f1,
            }
        )
    return thresholds, stats


def predict(proba: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    return (proba >= thresholds[None, :]).astype(np.int8)


def apply_fallback(y_pred: np.ndarray, proba: np.ndarray) -> int:
    empty = np.flatnonzero(y_pred.sum(axis=1) == 0)
    if len(empty):
        y_pred[empty, proba[empty].argmax(axis=1)] = 1
    return int(len(empty))


def _prf(y_true: np.ndarray, y_pred: np.ndarray, average: str) -> dict:
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average=average, zero_division=0
    )
    return {"precision": float(p), "recall": float(r), "f1": float(f1)}


def evaluate(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    proba: np.ndarray,
    thresholds: np.ndarray,
    rows: list[dict],
    fallback_pages: int,
) -> dict:
    n_labels = len(LABELS)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=np.arange(n_labels),
        average=None,
        zero_division=0,
    )

    per_label = []
    aps: list[float] = []
    for j, name in enumerate(LABELS):
        tp = int(np.count_nonzero((y_true[:, j] == 1) & (y_pred[:, j] == 1)))
        fp = int(np.count_nonzero((y_true[:, j] == 0) & (y_pred[:, j] == 1)))
        fn = int(np.count_nonzero((y_true[:, j] == 1) & (y_pred[:, j] == 0)))
        tn = int(len(y_true) - tp - fp - fn)
        ap = (
            float(average_precision_score(y_true[:, j], proba[:, j]))
            if y_true[:, j].sum() > 0
            else None
        )
        if ap is not None:
            aps.append(ap)
        per_label.append(
            {
                "label": name,
                "precision": float(precision[j]),
                "recall": float(recall[j]),
                "f1": float(f1[j]),
                "accuracy": float(np.mean(y_pred[:, j] == y_true[:, j])),
                "support": int(support[j]),
                "predicted_support": int(y_pred[:, j].sum()),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
                "average_precision": ap,
                "max_score": float(proba[:, j].max()),
                "threshold": float(thresholds[j]),
            }
        )

    top_idx = proba.argmax(axis=1)
    top1_presence = float(np.mean(y_true[np.arange(len(y_true)), top_idx]))
    single = y_true.sum(axis=1) == 1
    top1_single = (
        float(np.mean(top_idx[single] == y_true[single].argmax(axis=1)))
        if single.any()
        else None
    )

    fields = np.asarray([row.get("field", "") for row in rows], dtype=object)
    per_field = {}
    for field in sorted({f for f in fields if f}):
        mask = fields == field
        per_field[field] = {
            "n_pages": int(mask.sum()),
            "macro_f1": float(
                f1_score(y_true[mask], y_pred[mask], average="macro", zero_division=0)
            ),
        }

    manual = np.asarray([bool(row.get("is_section_start")) for row in rows], dtype=bool)
    manual_macro = (
        float(f1_score(y_true[manual], y_pred[manual], average="macro", zero_division=0))
        if manual.any()
        else None
    )

    return {
        "n_pages": int(len(y_true)),
        "subset_accuracy": float(accuracy_score(y_true, y_pred)),
        "hamming_loss": float(hamming_loss(y_true, y_pred)),
        "micro": _prf(y_true, y_pred, "micro"),
        "macro": _prf(y_true, y_pred, "macro"),
        "weighted": _prf(y_true, y_pred, "weighted"),
        "samples_f1": float(f1_score(y_true, y_pred, average="samples", zero_division=0)),
        "macro_average_precision": float(np.mean(aps)) if aps else None,
        "top1_presence_accuracy": top1_presence,
        "top1_single_label_accuracy": top1_single,
        "n_single_label_pages": int(single.sum()),
        "mean_labels_predicted": float(y_pred.sum(axis=1).mean()),
        "pages_with_no_prediction": int(np.count_nonzero(y_pred.sum(axis=1) == 0)),
        "fallback_pages": fallback_pages,
        "manual_only_macro_f1": manual_macro,
        "n_manual_pages": int(manual.sum()),
        "per_field": per_field,
        "per_label": per_label,
    }


def bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    article_ids: np.ndarray,
    n_boot: int = 1000,
    seed: int = 42,
) -> dict | None:
    if n_boot <= 0:
        return None

    unique, inverse = np.unique(article_ids, return_inverse=True)
    groups = [np.flatnonzero(inverse == i) for i in range(len(unique))]
    rng = np.random.default_rng(seed)

    macro, micro, subset = [], [], []
    for _ in range(n_boot):
        picked = rng.integers(0, len(unique), size=len(unique))
        idx = np.concatenate([groups[i] for i in picked])
        yt, yp = y_true[idx], y_pred[idx]
        macro.append(f1_score(yt, yp, average="macro", zero_division=0))
        micro.append(f1_score(yt, yp, average="micro", zero_division=0))
        subset.append(accuracy_score(yt, yp))

    def interval(values: list[float]) -> dict:
        return {
            "lower": float(np.percentile(values, 2.5)),
            "upper": float(np.percentile(values, 97.5)),
        }

    return {
        "n_boot": int(n_boot),
        "n_articles": int(len(unique)),
        "macro_f1": interval(macro),
        "micro_f1": interval(micro),
        "subset_accuracy": interval(subset),
    }


# ---------------------------------------------------------------------------
# model / training
# ---------------------------------------------------------------------------
class PageDataset(Dataset):
    def __init__(self, encodings, labels: np.ndarray):
        self.encodings = encodings
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> dict:
        item = {
            key: torch.tensor(values[index], dtype=torch.long)
            for key, values in self.encodings.items()
        }
        item["labels"] = torch.tensor(self.labels[index], dtype=torch.float32)
        return item


class SciBertClassifier(nn.Module):
    def __init__(self, model_name: str, num_labels: int, dropout: float = 0.1):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.encoder.config.hidden_size, num_labels)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True,
        )
        cls = outputs.last_hidden_state[:, 0]
        return self.classifier(self.dropout(cls))


def _autocast(enabled: bool):
    if not enabled:
        return nullcontext()
    try:
        return torch.amp.autocast("cuda", dtype=torch.float16)
    except AttributeError:  # torch < 2.1
        return torch.cuda.amp.autocast(dtype=torch.float16)


def _grad_scaler(enabled: bool):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):  # torch < 2.1
        return torch.cuda.amp.GradScaler(enabled=enabled)


def make_loader(
    dataset: PageDataset,
    tokenizer,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=DataCollatorWithPadding(tokenizer=tokenizer),
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    criterion: nn.Module,
    device: torch.device,
    amp: bool,
    log_every: int,
) -> float:
    model.train()
    scaler = _grad_scaler(amp)
    total_loss = 0.0
    seen = 0

    for step, batch in enumerate(loader, start=1):
        labels = batch.pop("labels").to(device)
        inputs = {key: value.to(device) for key, value in batch.items()}

        optimizer.zero_grad(set_to_none=True)
        with _autocast(amp):
            logits = model(**inputs)
            loss = criterion(logits, labels)

        if amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        total_loss += float(loss.item()) * len(labels)
        seen += len(labels)
        if log_every and step % log_every == 0:
            print(
                f"    step {step}/{len(loader)}  "
                f"loss {total_loss / seen:.4f}",
                flush=True,
            )
    return total_loss / max(seen, 1)


@torch.no_grad()
def run_eval(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    amp: bool,
) -> tuple[float, np.ndarray]:
    model.eval()
    total_loss = 0.0
    seen = 0
    probabilities: list[np.ndarray] = []

    for batch in loader:
        labels = batch.pop("labels").to(device)
        inputs = {key: value.to(device) for key, value in batch.items()}
        with _autocast(amp):
            logits = model(**inputs)
            loss = criterion(logits, labels)
        total_loss += float(loss.item()) * len(labels)
        seen += len(labels)
        probabilities.append(torch.sigmoid(logits.float()).cpu().numpy())

    if not probabilities:
        return 0.0, np.zeros((0, len(LABELS)), dtype=np.float32)
    return total_loss / max(seen, 1), np.vstack(probabilities)


# ---------------------------------------------------------------------------
# outputs
# ---------------------------------------------------------------------------
def write_outputs(
    out_dir: Path,
    report: dict,
    test_rows: list[dict],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    proba: np.ndarray,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    with (out_dir / "metrics.json").open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)

    columns = [
        "label",
        "precision",
        "recall",
        "f1",
        "accuracy",
        "support",
        "predicted_support",
        "tp",
        "fp",
        "fn",
        "tn",
        "average_precision",
        "max_score",
        "threshold",
    ]
    with (out_dir / "per_label.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        writer.writeheader()
        for row in report["per_label"]:
            writer.writerow({key: row[key] for key in columns})

    with (out_dir / "predictions_test.jsonl").open("w", encoding="utf-8") as fh:
        for row, gold, pred, probs in zip(test_rows, y_true, y_pred, proba):
            record = {
                "sample_id": row.get("sample_id"),
                "manuscript_id": row.get("manuscript_id"),
                "field": row.get("field"),
                "page": row.get("page"),
                "gold": [LABELS[j] for j in np.flatnonzero(gold)],
                "predicted": [LABELS[j] for j in np.flatnonzero(pred)],
                "probabilities": {
                    name: round(float(probs[j]), 6) for j, name in enumerate(LABELS)
                },
            }
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def print_report(
    metrics: dict,
    report: dict,
    threshold_stats: list[dict],
    bootstrap: dict | None,
) -> None:
    def block(title: str, values: dict) -> None:
        print(
            f"  {title:<12} P {values['precision']:.4f}  "
            f"R {values['recall']:.4f}  F1 {values['f1']:.4f}"
        )

    print("\nSciBERT-only multi-label classifier | test split")
    print(
        f"  pages {metrics['n_pages']} | best epoch {report['best_epoch']}"
        f" | max_length {report['max_length']} | device {report['device']}"
    )
    skipped = [s["label"] for s in threshold_stats if not s["tuned"]]
    if skipped:
        print(f"  fixed at 0.5 (few val positives): {', '.join(skipped)}")

    print("\nOverall")
    print(f"  subset accuracy      {metrics['subset_accuracy']:.4f}")
    print(f"  hamming loss         {metrics['hamming_loss']:.4f}")
    block("micro", metrics["micro"])
    block("macro", metrics["macro"])
    block("weighted", metrics["weighted"])
    print(f"  samples F1           {metrics['samples_f1']:.4f}")
    if metrics["macro_average_precision"] is not None:
        print(f"  macro AP             {metrics['macro_average_precision']:.4f}")
    print(f"  top-1 presence acc   {metrics['top1_presence_accuracy']:.4f}")
    if metrics["top1_single_label_accuracy"] is not None:
        print(
            f"  top-1 single-label   {metrics['top1_single_label_accuracy']:.4f}"
            f"  ({metrics['n_single_label_pages']} pages)"
        )
    print(
        f"  labels/page          {metrics['mean_labels_predicted']:.2f} predicted, "
        f"{metrics['fallback_pages']} pages filled by fallback"
    )
    if bootstrap:
        macro_ci = bootstrap["macro_f1"]
        micro_ci = bootstrap["micro_f1"]
        subset_ci = bootstrap["subset_accuracy"]
        print(
            f"  95% CI (articles)    macro [{macro_ci['lower']:.4f}, "
            f"{macro_ci['upper']:.4f}]  micro [{micro_ci['lower']:.4f}, "
            f"{micro_ci['upper']:.4f}]  subset [{subset_ci['lower']:.4f}, "
            f"{subset_ci['upper']:.4f}]"
        )

    print("\nPer label")
    print(
        f"  {'label':<15} {'prec':>6} {'rec':>6} {'f1':>6} {'supp':>6} "
        f"{'pred':>6} {'AP':>6} {'thr':>6}"
    )
    for row in metrics["per_label"]:
        ap = (
            f"{row['average_precision']:6.3f}"
            if row["average_precision"] is not None
            else "   n/a"
        )
        print(
            f"  {row['label']:<15} {row['precision']:6.3f} {row['recall']:6.3f} "
            f"{row['f1']:6.3f} {row['support']:6d} {row['predicted_support']:6d} "
            f"{ap} {row['threshold']:6.3f}"
        )

    if metrics["manual_only_macro_f1"] is not None:
        print(
            f"\nManual pages only  macro F1 {metrics['manual_only_macro_f1']:.4f}"
            f"  ({metrics['n_manual_pages']} pages)"
        )

    print("\nPer field (macro F1)")
    for field, values in metrics["per_field"].items():
        print(f"  {field:<8} {values['macro_f1']:.4f}  ({values['n_pages']} pages)")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SciBERT-only multi-label baseline for page sections."
    )
    parser.add_argument("--model-name", default="allenai/scibert_scivocab_uncased")
    parser.add_argument("--splits-dir", type=Path, default=DEFAULT_SPLITS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=2,
                        help="early-stopping patience in epochs")
    parser.add_argument("--min-tune-support", type=int, default=10,
                        help="minimum val positives required to tune a threshold")
    parser.add_argument("--bootstrap", type=int, default=1000,
                        help="article-level bootstrap resamples; 0 disables")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto",
                        help="auto, cuda, cuda:0 or cpu")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--no-amp", action="store_true",
                        help="disable mixed precision on CUDA")
    parser.add_argument("--pos-weight", action="store_true",
                        help="use train-derived positive weights in BCE loss")
    parser.add_argument("--no-fallback", action="store_true",
                        help="do not assign argmax to pages with no prediction")
    parser.add_argument("--limit-train", type=int, default=0,
                        help="debug: use only the first N train pages")
    parser.add_argument("--limit-val", type=int, default=0,
                        help="debug: use only the first N val pages")
    parser.add_argument("--limit-test", type=int, default=0,
                        help="debug: use only the first N test pages")
    parser.add_argument("--save-model", action="store_true",
                        help="save the encoder, tokenizer and classification head")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def choose_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def positive_weights(y_train: np.ndarray) -> np.ndarray:
    pos = y_train.sum(axis=0).astype(np.float64)
    neg = len(y_train) - pos
    weights = np.ones(len(LABELS), dtype=np.float64)
    mask = pos > 0
    weights[mask] = neg[mask] / pos[mask]
    return np.clip(weights, 1.0, 50.0).astype(np.float32)


def main() -> None:
    args = parse_args()
    if args.max_length > 512:
        raise SystemExit("SciBERT supports max_length <= 512")
    if args.max_length < 8:
        raise SystemExit("max_length is too small")

    t_start = time.perf_counter()
    set_seed(args.seed)
    device = choose_device(args.device)
    if device.type == "cpu":
        print("WARNING: running SciBERT on CPU is slow; use a GPU when possible.")

    splits_dir = args.splits_dir
    print(f"loading splits from {splits_dir}")
    train_rows = load_jsonl(splits_dir / "train.jsonl")
    val_rows = load_jsonl(splits_dir / "val.jsonl")
    test_rows = load_jsonl(splits_dir / "test.jsonl")

    if args.limit_train > 0:
        train_rows = train_rows[: args.limit_train]
    if args.limit_val > 0:
        val_rows = val_rows[: args.limit_val]
    if args.limit_test > 0:
        test_rows = test_rows[: args.limit_test]

    y_train = label_matrix(train_rows)
    y_val = label_matrix(val_rows)
    y_test = label_matrix(test_rows)
    print(
        f"  train {len(train_rows)} pages | val {len(val_rows)} | "
        f"test {len(test_rows)}"
    )

    print(f"loading tokenizer and model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = SciBertClassifier(args.model_name, len(LABELS), args.dropout).to(device)

    def encode(rows: list[dict]):
        return tokenizer(
            page_texts(rows),
            truncation=True,
            max_length=args.max_length,
            padding=False,
        )

    print("tokenizing pages ...")
    enc_train, enc_val, enc_test = encode(train_rows), encode(val_rows), encode(test_rows)

    train_loader = make_loader(
        PageDataset(enc_train, y_train),
        tokenizer,
        args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = make_loader(
        PageDataset(enc_val, y_val),
        tokenizer,
        args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    test_loader = make_loader(
        PageDataset(enc_test, y_test),
        tokenizer,
        args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    pos_weight = None
    if args.pos_weight:
        pos_weight = torch.tensor(positive_weights(y_train), dtype=torch.float32, device=device)
        print(f"  positive weights enabled: {pos_weight.detach().cpu().numpy().round(2)}")

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    total_steps = max(1, len(train_loader) * args.epochs)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * args.warmup_ratio),
        num_training_steps=total_steps,
    )
    amp = (not args.no_amp) and device.type == "cuda"
    print(
        f"training: {args.epochs} epochs, batch {args.batch_size}, "
        f"lr {args.lr}, amp={amp}"
    )

    history: list[dict] = []
    best_loss = float("inf")
    best_state = None
    best_epoch = 0
    bad_epochs = 0

    for epoch in range(1, args.epochs + 1):
        t_epoch = time.perf_counter()
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            criterion,
            device,
            amp,
            args.log_every,
        )
        val_loss, val_proba = run_eval(model, val_loader, criterion, device, amp)
        val_pred = (val_proba >= 0.5).astype(np.int8)
        val_macro = float(
            f1_score(y_val, val_pred, average="macro", zero_division=0)
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_macro_f1_at_0.5": val_macro,
                "seconds": time.perf_counter() - t_epoch,
            }
        )
        print(
            f"  epoch {epoch}/{args.epochs}  train_loss {train_loss:.4f}  "
            f"val_loss {val_loss:.4f}  val_macro@0.5 {val_macro:.4f}  "
            f"({history[-1]['seconds']:.1f}s)",
            flush=True,
        )

        if val_loss < best_loss - 1e-5:
            best_loss = val_loss
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print(f"  early stopping at epoch {epoch}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)
    print(f"  best epoch: {best_epoch} (val loss {best_loss:.4f})")

    print("predicting validation and test ...")
    _, val_proba = run_eval(model, val_loader, criterion, device, amp)
    _, test_proba = run_eval(model, test_loader, criterion, device, amp)

    thresholds, threshold_stats = tune_thresholds(
        y_val, val_proba, min_support=args.min_tune_support
    )
    y_pred = predict(test_proba, thresholds)
    fallback_pages = 0 if args.no_fallback else apply_fallback(y_pred, test_proba)

    metrics = evaluate(
        y_test, y_pred, test_proba, thresholds, test_rows, fallback_pages
    )
    bootstrap = bootstrap_ci(
        y_test,
        y_pred,
        np.asarray([row.get("manuscript_id", "") for row in test_rows], dtype=object),
        n_boot=args.bootstrap,
        seed=args.seed,
    )

    report = {
        "model": args.model_name,
        "split": "test",
        "labels": LABELS,
        "device": str(device),
        "max_length": int(args.max_length),
        "batch_size": int(args.batch_size),
        "epochs_requested": int(args.epochs),
        "epochs_run": len(history),
        "best_epoch": int(best_epoch),
        "best_val_loss": float(best_loss),
        "pos_weight": bool(args.pos_weight),
        "thresholds": {name: float(thresholds[j]) for j, name in enumerate(LABELS)},
        "threshold_stats": threshold_stats,
        "training_history": history,
        "bootstrap_95ci": bootstrap,
        **metrics,
    }

    write_outputs(args.out_dir, report, test_rows, y_test, y_pred, test_proba)

    if args.save_model:
        model.encoder.save_pretrained(args.out_dir / "encoder")
        tokenizer.save_pretrained(args.out_dir / "tokenizer")
        torch.save(
            {
                "classifier": model.classifier.state_dict(),
                "thresholds": thresholds,
                "labels": LABELS,
            },
            args.out_dir / "head.pt",
        )
        print(f"model written to {args.out_dir}")

    print_report(metrics, report, threshold_stats, bootstrap)
    print(f"\noutputs written to {args.out_dir}")
    print(f"total {time.perf_counter() - t_start:.1f}s")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
