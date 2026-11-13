#!/usr/bin/env python3
"""TF-IDF + structural cues + one-vs-rest Random Forest baseline.

Pipeline
--------
    page text  -> word TF-IDF (1-2 grams)
    page meta  -> 22 structural features + 17 heading-cue flags
               -> one binary Random Forest per label
               -> per-label thresholds (validation-tuned, min-support aware)
               -> labels for the page

The vectorizer and rules are fit only on the train split.  Thresholds are
tuned on val; test is scored once.  Rare labels with too few validation
positives are not tuned, because a threshold fitted on 3-10 positives is noise.

Usage
-----
    python code/tfidf_rf_baseline.py
    python code/tfidf_rf_baseline.py --no-rules      # text-only ablation
    python code/tfidf_rf_baseline.py --no-tune       # fixed 0.5 thresholds
    python code/tfidf_rf_baseline.py --min-tune-support 20

Outputs (results/tfidf_rf_baseline/)
------------------------------------
    metrics.json                  overall, per-label, per-field, CIs, threshold summary
    per_label.csv                 precision/recall/F1/accuracy/AP/TP/FP/FN/threshold
    predictions_test.jsonl        gold labels, predicted labels, probabilities per page
    pr_curves.csv, pr_curves.png  per-label precision-recall curves
    threshold_sensitivity_val.csv global threshold-offset trade-off on validation
    label_cooccurrence.csv        gold-vs-predicted label co-occurrence counts
    model.joblib                  only with --save-model
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    hamming_loss,
    precision_recall_curve,
    precision_recall_fscore_support,
)
from sklearn.multiclass import OneVsRestClassifier

try:
    from page_features import feature_names as rule_feature_names
    from page_features import page_feature_matrix
except ImportError:  # allows `python -m code.tfidf_rf_baseline`
    from .page_features import feature_names as rule_feature_names
    from .page_features import page_feature_matrix

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
DEFAULT_OUT = ROOT / "results" / "tfidf_rf_baseline"


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
    """Return an (n_pages, n_labels) 0/1 matrix from `y` or `labels_covered`."""
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
# features
# ---------------------------------------------------------------------------
def build_features(
    train_rows: list[dict],
    test_rows: list[dict],
    use_rules: bool,
    max_features: int,
    min_df: int,
) -> tuple:
    vectorizer = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        ngram_range=(1, 2),
        min_df=min_df,
        max_features=max_features if max_features > 0 else None,
        sublinear_tf=True,
        dtype=np.float32,
    )
    X_train_text = vectorizer.fit_transform(page_texts(train_rows))
    X_test_text = vectorizer.transform(page_texts(test_rows))
    n_text = X_train_text.shape[1]

    if not use_rules:
        return vectorizer, X_train_text.tocsc(), X_test_text.tocsc(), n_text, 0

    rules_train = csr_matrix(page_feature_matrix(train_rows))
    rules_test = csr_matrix(page_feature_matrix(test_rows))
    X_train = hstack([X_train_text, rules_train], format="csc")
    X_test = hstack([X_test_text, rules_test], format="csc")
    return vectorizer, X_train, X_test, n_text, rules_train.shape[1]


# ---------------------------------------------------------------------------
# thresholds
# ---------------------------------------------------------------------------
def threshold_candidates(scores: np.ndarray) -> np.ndarray:
    """Fixed grid plus label-specific score quantiles.

    Quantiles let a label whose probabilities live below 0.05 still be tuned,
    while the 0.001 floor avoids the pathological "predict everything positive"
    threshold of 0.
    """
    quantiles = np.quantile(scores, np.linspace(0.50, 0.999, 150))
    fixed = np.arange(0.01, 0.99, 0.02)
    candidates = np.concatenate([fixed, quantiles])
    return np.unique(np.clip(candidates, 0.001, 1.0)).astype(np.float64)


def tune_thresholds(
    y_true: np.ndarray,
    proba: np.ndarray,
    min_support: int,
) -> tuple[np.ndarray, list[dict]]:
    """Per-label threshold maximising F1 on validation.

    Labels with fewer than `min_support` positives are left at 0.5; tuning on
    3-10 examples is not meaningful.
    """
    thresholds = np.full(y_true.shape[1], 0.5, dtype=np.float64)
    stats: list[dict] = []

    for j, name in enumerate(LABELS):
        positive = y_true[:, j].astype(bool)
        support = int(positive.sum())
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

        scores = proba[:, j]
        best_f1, best_t = -1.0, 0.5
        for t in threshold_candidates(scores):
            predicted = scores >= t
            tp = int(np.count_nonzero(predicted & positive))
            fp = int(np.count_nonzero(predicted & ~positive))
            fn = int(np.count_nonzero(~predicted & positive))
            denom = 2 * tp + fp + fn
            f1 = (2 * tp / denom) if denom else 0.0
            if f1 > best_f1:
                best_f1, best_t = f1, float(t)

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
    """Give every page at least one label: use argmax where nothing passed."""
    empty = np.flatnonzero(y_pred.sum(axis=1) == 0)
    if len(empty):
        y_pred[empty, proba[empty].argmax(axis=1)] = 1
    return int(len(empty))


def threshold_sensitivity(
    y_true: np.ndarray,
    proba: np.ndarray,
    thresholds: np.ndarray,
    step: float = 0.02,
    limit: float = 0.40,
) -> dict:
    """Evaluate one global offset added to every label's threshold.

    This is the cardinality/precision trade-off: positive offsets remove
    predictions, negative offsets add them.  It is fitted on validation only.
    """
    gold_cardinality = float(y_true.sum(axis=1).mean())
    rows: list[dict] = []
    for delta in np.round(np.arange(-limit, limit + 1e-9, step), 4):
        shifted = np.clip(thresholds + delta, 0.001, 1.0)
        pred = (proba >= shifted[None, :]).astype(np.int8)
        p, r, f1, _ = precision_recall_fscore_support(
            y_true, pred, average="macro", zero_division=0
        )
        rows.append(
            {
                "delta": float(delta),
                "labels_per_page": float(pred.sum(axis=1).mean()),
                "cardinality_gap": abs(float(pred.sum(axis=1).mean()) - gold_cardinality),
                "macro_f1": float(f1),
                "macro_precision": float(p),
                "macro_recall": float(r),
                "micro_f1": float(
                    f1_score(y_true, pred, average="micro", zero_division=0)
                ),
                "subset_accuracy": float(accuracy_score(y_true, pred)),
            }
        )

    best_macro = max(
        rows,
        key=lambda row: (round(row["macro_f1"], 6), -abs(row["delta"])),
    )
    best_cardinality = min(
        rows,
        key=lambda row: (round(row["cardinality_gap"], 4), abs(row["delta"])),
    )
    return {
        "gold_labels_per_page": gold_cardinality,
        "best_macro_offset": best_macro,
        "best_cardinality_offset": best_cardinality,
        "rows": rows,
    }


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------
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
    """95% percentile intervals from resampling whole articles."""
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


def write_threshold_csv(out_dir: Path, sensitivity: dict | None) -> None:
    if not sensitivity:
        return
    rows = sensitivity["rows"]
    with (out_dir / "threshold_sensitivity_val.csv").open(
        "w", encoding="utf-8", newline=""
    ) as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_pr_curves(out_dir: Path, y_true: np.ndarray, proba: np.ndarray) -> None:
    rows = []
    curves = []
    for j, name in enumerate(LABELS):
        if y_true[:, j].sum() == 0:
            continue
        precision, recall, thresholds = precision_recall_curve(
            y_true[:, j], proba[:, j]
        )
        ap = float(average_precision_score(y_true[:, j], proba[:, j]))
        curves.append((name, precision, recall, ap))
        for t, p, r in zip(thresholds, precision[:-1], recall[:-1]):
            rows.append(
                {
                    "label": name,
                    "threshold": float(t),
                    "precision": float(p),
                    "recall": float(r),
                }
            )

    with (out_dir / "pr_curves.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=["label", "threshold", "precision", "recall"]
        )
        writer.writeheader()
        writer.writerows(rows)

    if not curves:
        return
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    ncols = 6
    nrows = (len(curves) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(18, 3.0 * nrows), squeeze=False)
    for ax, (name, precision, recall, ap) in zip(axes.flat, curves):
        ax.plot(recall, precision, linewidth=1.2)
        ax.set_title(f"{name}  AP={ap:.3f}", fontsize=9)
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.grid(alpha=0.2)
        ax.set_xlabel("recall", fontsize=7)
        ax.set_ylabel("precision", fontsize=7)
    for ax in axes.flat[len(curves):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_dir / "pr_curves.png", dpi=150)
    plt.close(fig)


def write_cooccurrence(
    out_dir: Path, y_true: np.ndarray, y_pred: np.ndarray
) -> None:
    matrix = y_true.T.astype(np.int32) @ y_pred.astype(np.int32)
    with (out_dir / "label_cooccurrence.csv").open(
        "w", encoding="utf-8", newline=""
    ) as fh:
        writer = csv.writer(fh)
        writer.writerow(["gold \\ predicted"] + LABELS)
        for name, values in zip(LABELS, matrix):
            writer.writerow([name] + values.tolist())


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------
def print_report(
    metrics: dict,
    timings: dict,
    n_text_features: int,
    n_rule_features: int,
    threshold_stats: list[dict],
    sensitivity: dict | None,
    bootstrap: dict | None,
    tuned: bool,
    min_support: int,
) -> None:
    def block(title: str, values: dict) -> None:
        print(
            f"  {title:<12} P {values['precision']:.4f}  "
            f"R {values['recall']:.4f}  F1 {values['f1']:.4f}"
        )

    print("\nTF-IDF + structural cues + one-vs-rest Random Forest | test split")
    print(
        f"  pages {metrics['n_pages']} | text {n_text_features:,} + "
        f"rules {n_rule_features} = {n_text_features + n_rule_features:,} features"
    )
    print(
        f"  fit {timings['fit']:.1f}s | predict {timings['predict']:.1f}s | "
        f"thresholds: {'tuned on val' if tuned else 'fixed at 0.5'}"
    )
    if tuned:
        skipped = [s["label"] for s in threshold_stats if not s["tuned"]]
        print(f"  min tuning support: {min_support} val positives")
        if skipped:
            print(f"  fixed at 0.5 (too few val positives): {', '.join(skipped)}")

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

    if sensitivity:
        zero = next(
            (row for row in sensitivity["rows"] if abs(row["delta"]) < 1e-9), None
        )
        print("\nThreshold sensitivity (validation, global offset)")
        print(
            f"  gold labels/page {sensitivity['gold_labels_per_page']:.2f}"
            + (
                f" | base {zero['labels_per_page']:.2f}, macro {zero['macro_f1']:.4f}"
                if zero
                else ""
            )
        )
        best = sensitivity["best_macro_offset"]
        print(
            f"  best macro offset {best['delta']:+.2f}: macro {best['macro_f1']:.4f}, "
            f"labels/page {best['labels_per_page']:.2f}"
        )
        card = sensitivity["best_cardinality_offset"]
        print(
            f"  cardinality offset {card['delta']:+.2f}: "
            f"labels/page {card['labels_per_page']:.2f}, macro {card['macro_f1']:.4f}"
        )


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="TF-IDF + structural cues + one-vs-rest Random Forest baseline."
    )
    parser.add_argument("--splits-dir", type=Path, default=DEFAULT_SPLITS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--n-estimators", type=int, default=300,
                        help="trees per label (default: 300)")
    parser.add_argument("--max-features", type=int, default=150_000,
                        help="TF-IDF vocabulary cap; <=0 means unlimited")
    parser.add_argument("--min-df", type=int, default=3,
                        help="minimum document frequency for a term")
    parser.add_argument("--n-jobs", type=int, default=-1,
                        help="cores used by each Random Forest")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-rules", action="store_true",
                        help="disable structural/heading features (text-only ablation)")
    parser.add_argument("--no-tune", action="store_true",
                        help="use 0.5 for every label instead of tuning on val")
    parser.add_argument("--min-tune-support", type=int, default=10,
                        help="minimum val positives required to tune a label threshold")
    parser.add_argument("--bootstrap", type=int, default=1000,
                        help="article-level bootstrap resamples; 0 disables")
    parser.add_argument("--no-fallback", action="store_true",
                        help="do not assign argmax to pages with no prediction")
    parser.add_argument("--save-model", action="store_true",
                        help="also write vectorizer/forest/thresholds to model.joblib")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    t_start = time.perf_counter()

    train_path = args.splits_dir / "train.jsonl"
    test_path = args.splits_dir / "test.jsonl"
    val_path = args.splits_dir / "val.jsonl"

    print(f"loading splits from {args.splits_dir}")
    train_rows = load_jsonl(train_path)
    test_rows = load_jsonl(test_path)

    tune = not args.no_tune
    val_rows: list[dict] = []
    if tune:
        if val_path.exists():
            val_rows = load_jsonl(val_path)
        else:
            print(f"warning: {val_path} not found, falling back to fixed 0.5 thresholds")
            tune = False

    y_train = label_matrix(train_rows)
    y_test = label_matrix(test_rows)
    print(
        f"  train {len(train_rows)} pages | "
        f"val {len(val_rows) if tune else 0} | test {len(test_rows)}"
    )

    print("building features (word TF-IDF + structural cues) ...")
    vectorizer, X_train, X_test, n_text_features, n_rule_features = build_features(
        train_rows,
        test_rows,
        use_rules=not args.no_rules,
        max_features=args.max_features,
        min_df=args.min_df,
    )
    print(
        f"  {n_text_features:,} text features + {n_rule_features} rule features, "
        f"{X_train.nnz:,} non-zeros in train"
    )

    print(
        f"fitting Random Forest: {args.n_estimators} trees x {len(LABELS)} labels "
        f"(one-vs-rest) ..."
    )
    forest = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_features="sqrt",
        min_samples_leaf=1,
        n_jobs=args.n_jobs,
        random_state=args.seed,
    )
    classifier = OneVsRestClassifier(forest, n_jobs=1)

    t_fit = time.perf_counter()
    classifier.fit(X_train, y_train)
    fit_seconds = time.perf_counter() - t_fit
    print(f"  fit done in {fit_seconds:.1f}s")

    t_predict = time.perf_counter()
    proba_test = classifier.predict_proba(X_test)

    thresholds = np.full(len(LABELS), 0.5, dtype=np.float64)
    threshold_stats: list[dict] = []
    sensitivity = None
    if tune:
        y_val = label_matrix(val_rows)
        X_val_text = vectorizer.transform(page_texts(val_rows))
        if args.no_rules:
            X_val = X_val_text.tocsc()
        else:
            X_val = hstack(
                [X_val_text, csr_matrix(page_feature_matrix(val_rows))], format="csc"
            )
        proba_val = classifier.predict_proba(X_val)
        thresholds, threshold_stats = tune_thresholds(
            y_val, proba_val, min_support=args.min_tune_support
        )
        sensitivity = threshold_sensitivity(y_val, proba_val, thresholds)
        print(
            f"  thresholds tuned on {len(val_rows)} val pages "
            f"(min support {args.min_tune_support})"
        )
    predict_seconds = time.perf_counter() - t_predict

    y_pred = predict(proba_test, thresholds)
    fallback_pages = 0 if args.no_fallback else apply_fallback(y_pred, proba_test)
    metrics = evaluate(
        y_test, y_pred, proba_test, thresholds, test_rows, fallback_pages
    )
    bootstrap = bootstrap_ci(
        y_test,
        y_pred,
        np.asarray([row.get("manuscript_id", "") for row in test_rows], dtype=object),
        n_boot=args.bootstrap,
        seed=args.seed,
    )

    sensitivity_summary = None
    if sensitivity:
        sensitivity_summary = {
            "gold_labels_per_page": sensitivity["gold_labels_per_page"],
            "best_macro_offset": sensitivity["best_macro_offset"],
            "best_cardinality_offset": sensitivity["best_cardinality_offset"],
        }

    report = {
        "model": "tfidf_rules_ovr_random_forest",
        "split": "test",
        "labels": LABELS,
        "n_features_text": int(n_text_features),
        "n_features_rules": int(n_rule_features),
        "n_features_total": int(n_text_features + n_rule_features),
        "rule_features": rule_feature_names() if not args.no_rules else [],
        "thresholds": {name: float(thresholds[j]) for j, name in enumerate(LABELS)},
        "threshold_stats": threshold_stats,
        "threshold_analysis": sensitivity_summary,
        "bootstrap_95ci": bootstrap,
        **metrics,
    }

    write_outputs(args.out_dir, report, test_rows, y_test, y_pred, proba_test)
    write_threshold_csv(args.out_dir, sensitivity)
    write_pr_curves(args.out_dir, y_test, proba_test)
    write_cooccurrence(args.out_dir, y_test, y_pred)

    if args.save_model:
        import joblib

        model_path = args.out_dir / "model.joblib"
        joblib.dump(
            {
                "vectorizer": vectorizer,
                "classifier": classifier,
                "thresholds": thresholds,
                "labels": LABELS,
                "rule_features": rule_feature_names() if not args.no_rules else [],
            },
            model_path,
        )
        print(f"model written to {model_path}")

    print_report(
        metrics,
        {"fit": fit_seconds, "predict": predict_seconds},
        n_text_features,
        n_rule_features,
        threshold_stats,
        sensitivity,
        bootstrap,
        tuned=tune,
        min_support=args.min_tune_support,
    )
    print(f"\noutputs written to {args.out_dir}")
    print(f"total {time.perf_counter() - t_start:.1f}s")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
