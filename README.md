# AI-Driven Section Identification and Classification in Research Manuscripts

Page-level multi-label classification of the structural sections of scientific
manuscripts. Each input is a single page of a single PDF; the output is the set
of canonical sections present on that page. The repository contains the
annotated dataset, the article-level train/val/test splits, and three baselines
evaluated on the same held-out test set:

- **TF-IDF + structural cues + one-vs-rest Random Forest**
- **SciBERT** (`allenai/scibert_scivocab_uncased`) fine-tuned with a multi-label head
- **Qwen2.5-7B-Instruct, 4-bit**, run locally 

The 18-label taxonomy, in canonical order:

```text
title  abstract  keywords  introduction  background  objectives  analysis
study_design  materials  results  method  experiments  evaluation
discussion  conclusion  references  declarations  supplementary
```

A page can carry more than one label (for example `title`, `abstract` and
`introduction` on page 1), so the task is multi-label rather than multi-class.

---

## Repository layout

```text
code/
  llm.py                    Qwen2.5-7B baseline (page level by default, article ablation)
  scibert.py                SciBERT multi-label baseline
  tfidf_rf_baseline.py      TF-IDF + structural cues + Random Forest baseline
  page_features.py          22 layout features + 17 heading-cue flags
  __init__.py

data/
  manuscripts.json          raw corpus: 500 manuscripts with page texts and metadata (22 MB)
  v1/
    pages.jsonl             page-level dataset: 6,201 samples x 18 labels
    metadata.jsonl          one row per manuscript
    build_report.json       provenance, label counts, every dropped page
    audit_sample.jsonl      audit sample
    README.md               schema and label derivation rule
  splits/
    train.jsonl  val.jsonl  test.jsonl          article-level splits
    metadata_*.jsonl  splits.json  split_report.json

docs/
  stats.md                  corpus, label support, field distribution, co-occurrence
  audit.md                  structural and label-integrity checks

notebooks/
  01_dataset_overview.ipynb exploratory plots for data/v1/pages.jsonl
                            (imports the data-build module `dataset.py`, which is not
                            part of this snapshot)

results/
  tfidf_rf_baseline/        metrics, per-label table, predictions, PR curves
  scibert_results/          metrics, per-label table, predictions
  llm_results/              article-level LLM run (ablation)
  llm_pages/                created by the page-level LLM run

archive/                    earlier collection scripts and raw JSON (git-ignored)
requirements.txt            pinned dependencies for the scikit-learn baselines
```

---

## Dataset

The corpus was collected from **arXiv** across six domains:

| Field | Manuscripts |
|---|---:|
| Computer Science (`cs`) | 150 |
| Quantitative Biology (`q-bio`) | 98 |
| Physics (`physics`) | 80 |
| Mathematics (`math`) | 68 |
| Statistics (`stat`) | 55 |
| Economics (`econ`) | 49 |
| **Total** | **500** |

Publication types in the source metadata: 298 journal papers, 197 preprints and
5 conference papers. Each PDF was converted to plain text page by page; reading
order and article membership are preserved.

### Corpus statistics

| Quantity | Value |
|---|---|
| Manuscripts | 500 |
| Source pages | 6,386 |
| Usable page samples | 6,201 |
| Labels | 18 |

Four labels are low-support and their per-label F1 is noisy: `evaluation` (87
covered pages), `analysis` (79), `objectives` (47) and `materials` (44). The
full support table is in [`docs/stats.md`](docs/stats.md).

### Splits

Splits are made **by article**, so pages from one manuscript never appear in two
splits (seed 42, 5 folds):

| Split | Articles | Pages | Manual pages |
|---|---:|---:|---:|
| Train | 352 | 4,289 | 1,554 |
| Validation | 74 | 924 | 324 |
| Test | 74 | 988 | 337 |


## Annotation protocol

Annotation is manual and page-level. For each page the annotator identified
every section heading present and mapped it to exactly one canonical label by
the **function** of the section, not its wording (for example "Methods",
"Methodology" and "Proposed Approach" all map to `method`).


Python 3.10–3.11 is recommended (the code uses `X | None` type unions, and the
pinned scikit-learn build matches CPython 3.11).

```bash
pip install -r requirements.txt
```

`requirements.txt` pins what the scikit-learn baselines need. The neural
baselines require extra packages:

```bash
# SciBERT
pip install torch transformers

# page-level / article-level LLM (4-bit, local)
pip install torch transformers accelerate bitsandbytes
```

The LLM baseline is free and local: it loads `Qwen/Qwen2.5-7B-Instruct` in 4-bit
NF4 (about 5 GB of weights), so a free Colab/Kaggle T4 is enough. No API key is
used. On Windows, `scibert.py` imports torch before scikit-learn to avoid a
`c10.dll` initialisation error.

---

## Reproducing the baselines

Run from the repository root. All three read the same splits and write to
separate result directories.

### 1. TF-IDF + structural cues + Random Forest

```bash
python code/tfidf_rf_baseline.py
```

Defaults: 300 trees per label, 150k TF-IDF vocabulary cap, `min_df=3`, 39
structural features (22 layout + 17 heading cues), per-label thresholds tuned on
validation. Outputs go to `results/tfidf_rf_baseline/`.

Useful flags: `--no-rules` (text-only ablation), `--no-tune` (fixed 0.5
thresholds), `--min-tune-support 20`, `--bootstrap 0`, `--save-model`.

### 2. SciBERT

```bash
python code/scibert.py --epochs 3 --batch-size 16
```

The recorded run used `max_length=512`, batch size 16, 3 epochs (best epoch 2)
and per-label threshold tuning on validation. Outputs go to
`results/scibert_results/`.

Useful flags: `--max-length`, `--batch-size`, `--lr`, `--pos-weight`,
`--no-amp`, `--device cpu`, and `--limit-train/--limit-val/--limit-test` for a
smoke test.

### 3. Qwen2.5-7B-Instruct (page level)

Full run on a GPU:

```bash
python code/llm.py --unit page --data data/splits/test.jsonl \
    --out results/llm_pages --batch-size 4
```





### Metric definitions

- **Subset accuracy**: fraction of pages whose predicted label set exactly
  equals the gold set. The strictest measure.
- **Hamming loss**: fraction of wrong cells over all page × label pairs.
- **Micro / macro / weighted / samples P/R/F1**: standard multi-label averages;
  macro treats all 18 labels equally, samples averages per page first.
- **Bootstrap CI**: 95% interval from 1,000 resamples of articles with
  replacement.

---

## Known limitations

- **Class imbalance.** `evaluation`, `analysis`, `objectives` and `materials`
  have fewer than 100 positive pages each; per-label F1 for these is unstable
  and should be reported with intervals.
- **Inherited labels.** 64.3% of samples inherit a label rather than carry a
  manual one. Manual-only macro F1 is reported alongside the full number so the
  two effects can be separated.
- **Input length differs across models.** SciBERT is capped at 512 wordpieces
  and truncates longer pages; the LLM sees the full page (up to its prompt
  budget). This is a modelling difference, not a data difference.
- **LLM page-level results are pending.** The code, evaluation and mock
  plumbing are complete; the GPU run has not been executed yet, so no page-level
  Qwen numbers are claimed here.
- **Low-support labels dominate the error.** They are the main source of missed
  labels for all three approaches.

---

