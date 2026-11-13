#!/usr/bin/env python3
"""
llm.py - Manuscript section identification with Qwen2.5-7B-Instruct (4-bit).

Two evaluation units
  page     (default) ONE PAGE per prompt -> predict every section present on that page.
           This is the dataset's native task and is directly comparable with the
           TF-IDF and SciBERT baselines (same pages, same labels, same metrics).
  article  (ablation) the whole article is passed in one prompt (or page-aligned
           chunks if it does not fit) -> predict the article's distinct section set.

Pipeline (both units)
  1. INPUT        page mode: `text` + optional page position (off by default);
                  article mode: pages grouped by `manuscript_id`, ordered and joined.
  2. EXTRACTION   FIXED TAXONOMY: the LLM is given strictly the exact list of allowed
                  labels (no descriptions, synonyms or variations).
  3. VALIDATION   Every returned label must belong strictly to the target label set;
                  anything else is rejected (not mapped, not repaired). Page mode adds
                  one repair retry; article mode does not.
     EVALUATION   page mode: per-page set comparison (subset accuracy, Hamming loss,
                  micro/macro/weighted/samples P/R/F1, per-label, per-field,
                  labels/page, bootstrap CI clustered by article).
                  article mode: distinct predicted labels == distinct ground-truth set.

Ground truth
  page mode   : `labels_covered` (falls back to `labels` if absent).
  article mode: union of the `labels` field over all pages of the article.

Usage
  pip install torch transformers accelerate bitsandbytes

  # page level (default), full held-out test split
  python llm.py --unit page --data data/splits/test.jsonl --out results/llm_pages

  # smoke test without a GPU
  python llm.py --unit page --data data/splits/test.jsonl --out /tmp/llm_pages \
      --backend mock --limit 20

  # article-level ablation (previous behaviour)
  python llm.py --unit article --data data/splits/test.jsonl --out results/llm_articles

  # re-run = resume; predictions already in <out>/predictions.jsonl are skipped
  python llm.py --unit page --data data/splits/test.jsonl --out results/llm_pages

Already-quantised checkpoints also work (AWQ / GPTQ), e.g.
  --model Qwen/Qwen2.5-7B-Instruct-AWQ        (needs `pip install autoawq`)
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# Canonical taxonomy, in the dataset's order (data/v1/README.md).
LABELS: List[str] = [
    "title", "abstract", "keywords", "introduction", "background", "objectives",
    "analysis", "study_design", "materials", "results", "method", "experiments",
    "evaluation", "discussion", "conclusion", "references", "declarations",
    "supplementary",
]

# --------------------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------------------

def _read_jsonl(path: str) -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_pages(path: str) -> List[dict]:
    """One row = one page = one sample. Target is `labels_covered`."""
    pages = []
    for r in _read_jsonl(path):
        page = r.get("page")
        sample_id = r.get("sample_id") or f"{r['manuscript_id']}#p{page}"
        gt = r.get("labels_covered")
        if gt is None:
            gt = r.get("labels", [])
        pages.append({
            "sample_id": sample_id,
            "manuscript_id": r["manuscript_id"],
            "field": r.get("field"),
            "article_type": r.get("article_type"),
            "page": page,
            "n_pages": r.get("n_pages"),
            "text": r["text"],
            "gt": sorted(set(gt)),
            "is_section_start": bool(r.get("is_section_start", False)),
            "label_source": r.get("label_source") or ("manual" if r.get("is_section_start") else "inherited"),
        })
    pages.sort(key=lambda p: (p["manuscript_id"], p["page"] if p["page"] is not None else 0))
    return pages


def load_articles(path: str) -> List[dict]:
    """Group page-level rows into articles (ordered by page) with distinct GT labels."""
    by_ms: Dict[str, List[dict]] = defaultdict(list)
    for r in _read_jsonl(path):
        by_ms[r["manuscript_id"]].append(r)
    articles = []
    for mid, rows in by_ms.items():
        rows.sort(key=lambda r: r["page"])
        articles.append({
            "manuscript_id": mid,
            "field": rows[0].get("field"),
            "article_type": rows[0].get("article_type"),
            "n_pages": rows[0].get("n_pages"),
            "pages": [r["text"] for r in rows],
            "text": "\n\n".join(r["text"] for r in rows),
            # distinct ground-truth section labels for the article
            "gt": sorted({l for r in rows for l in (r.get("labels") or r.get("labels_covered") or [])}),
        })
    articles.sort(key=lambda a: a["manuscript_id"])
    return articles


def all_labels(articles: Sequence[dict]) -> List[str]:
    """All labels actually present in the data, in canonical order."""
    seen = {l for a in articles for l in a["gt"]}
    unknown = sorted(seen - set(LABELS))
    if unknown:
        print(f"[warn] labels outside the canonical taxonomy: {unknown}", file=sys.stderr)
    ordered = [l for l in LABELS if l in seen]
    return ordered + unknown


# --------------------------------------------------------------------------------------
# Prompting
# --------------------------------------------------------------------------------------
SYSTEM_PROMPT = (
    "You are an expert at analysing the structure of scientific manuscripts. "
    "You always answer with a single valid JSON object and nothing else."
)


def build_user_prompt(article_text: str, labels: Sequence[str], partial: bool = False) -> str:
    label_block = "\n".join(f'- "{l}"' for l in labels)
    scope = ("a CONTIGUOUS PART of a manuscript" if partial else "the full text of a manuscript")
    return f"""Below is {scope}, extracted from a PDF (the extraction is noisy: spaces may be
missing, columns may be interleaved, page headers/footers may appear in the text).

ALLOWED LABELS (fixed taxonomy - this is the complete list):
{label_block}

TASK
Read the entire text and extract ONLY the sections whose heading maps directly to one of the
allowed labels above. For each extracted section output the heading as found in the text and
exactly one allowed label.
- The label must be copied exactly as written in the allowed list.
- Never output a label that is not in the allowed list. Do not create new labels, variations or synonyms.
- If a section does not map directly to an allowed label, skip it.
- Do not list sections that are not present in the text.

OUTPUT FORMAT (JSON only, no commentary):
{{"sections": [{{"heading": "<heading as found in the text>", "label": "<allowed label>"}}, ...]}}

MANUSCRIPT TEXT
\"\"\"
{article_text}
\"\"\"

JSON:"""


def build_page_prompt(page_text: str, labels: Sequence[str],
                      position: Optional[str] = None,
                      repair: Optional[dict] = None) -> str:
    """One page per prompt. Accepts multiple labels, mid-section and boundary pages.

    `position` is only passed when --page-position-context is enabled; by default the
    model sees the page text alone, matching the text-only input of the page baselines.
    """
    label_block = "\n".join(f'- "{l}"' for l in labels)
    position_block = f"\nPAGE POSITION\n{position}\n" if position else ""
    repair_block = ""
    if repair:
        repair_block = f"""
CORRECTION
Your previous answer was rejected.
Previous answer: {repair.get('previous', '')}
Problem: {repair.get('problem', '')}
Answer again, using only labels from the allowed list.
"""
    return f"""Below is ONE PAGE from a scientific manuscript, extracted from a PDF (the
extraction is noisy: spaces may be missing, columns may be interleaved, page headers and
footers may appear in the text).
{position_block}
ALLOWED LABELS (fixed taxonomy - this is the complete list):
{label_block}

TASK
Identify every section that is present on this page.
- A page may contain the start of a section, the middle or continuation of a section, or a
  boundary between two or more sections.
- Include a section that continues from a previous page even if its heading is not repeated
  here; judge from the content.
- Most pages belong to exactly one section. Output more than one label only when several
  sections genuinely appear on the page (for example a title page with title, abstract and
  introduction, or a page where one section ends and another starts).
- If a section starts on this page, output it even if most of its text is on the next page.
- Copy each label exactly as written in the allowed list.
- Never output a label that is not in the allowed list. Do not create new labels, variations
  or synonyms.
- Every page belongs to at least one section, so never answer with "none" or an empty list.
{repair_block}
OUTPUT FORMAT (JSON only, no commentary):
{{"labels": ["<allowed label>", "<allowed label>", ...]}}

PAGE TEXT
\"\"\"
{page_text}
\"\"\"

JSON:"""

# --------------------------------------------------------------------------------------
# Output parsing / normalisation
# --------------------------------------------------------------------------------------

def validate_label(raw: str, labels: Set[str]) -> Optional[str]:
    """Strict validation: accept only an exact member of the target label set
    (surrounding whitespace / letter case ignored). No synonym mapping, no repair."""
    k = str(raw).strip().lower()
    return k if k in labels else None


def parse_model_output(text: str) -> List[Tuple[str, str]]:
    """Return [(heading, raw_label)] from the model output; tolerant to junk around JSON."""
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    dec = json.JSONDecoder()
    for m in re.finditer(r"[{\[]", text):
        try:
            obj, _ = dec.raw_decode(text[m.start():])
        except json.JSONDecodeError:
            continue
        items = obj.get("sections", obj.get("labels")) if isinstance(obj, dict) else obj
        if isinstance(items, list):
            out = []
            for it in items:
                if isinstance(it, dict) and "label" in it:
                    out.append((str(it.get("heading", "")), str(it["label"])))
                elif isinstance(it, str):
                    out.append(("", it))
            return out
    # last resort: truncated JSON -> regex out the labels
    return [("", m.group(1)) for m in re.finditer(r'"label"\s*:\s*"([^"]+)"', text)]


def parse_page_output(text: str) -> Tuple[List[str], bool]:
    """Return (raw_labels, parsed_cleanly) for a page-level answer."""
    cleaned = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    dec = json.JSONDecoder()
    for m in re.finditer(r"[{\[]", cleaned):
        try:
            obj, _ = dec.raw_decode(cleaned[m.start():])
        except json.JSONDecodeError:
            continue
        items = obj.get("labels", obj.get("sections")) if isinstance(obj, dict) else obj
        if isinstance(items, list):
            out: List[str] = []
            for it in items:
                if isinstance(it, dict) and "label" in it:
                    out.append(str(it["label"]))
                elif isinstance(it, str):
                    out.append(it)
            return out, True
    found = re.findall(r'"label"\s*:\s*"([^"]+)"', cleaned)
    if found:
        return found, False
    # tolerate a bare JSON list of strings that also looks like a label dump
    return [], False

# --------------------------------------------------------------------------------------
# Backends
# --------------------------------------------------------------------------------------

class HFQwenBackend:
    """Qwen2.5-7B-Instruct, 4-bit NF4 via bitsandbytes (or a pre-quantised AWQ/GPTQ repo)."""

    def __init__(self, model_id: str, max_new_tokens: int = 1024, no_4bit: bool = False,
                 attn: str = "sdpa", prefill_chunk: int = 1024):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        self.torch = torch
        self.max_new_tokens = max_new_tokens
        self.prefill_chunk = prefill_chunk
        self.tok = AutoTokenizer.from_pretrained(model_id)
        kwargs = dict(device_map="auto", attn_implementation=attn)
        prequant = any(t in model_id.lower() for t in ("awq", "gptq", "gguf", "bnb"))
        if not prequant and not no_4bit:
            # bf16 only on Ampere+ (compute capability >= 8); T4 (7.5) must use fp16
            bf16_ok = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
            dtype = torch.bfloat16 if bf16_ok else torch.float16
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype)
        else:
            kwargs["torch_dtype"] = "auto"
        self.model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs).eval()
        self.ctx = getattr(self.model.config, "max_position_embeddings", 32768)

    def _chat(self, user: str) -> str:
        return self.tok.apply_chat_template(
            [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}],
            tokenize=False, add_generation_prompt=True)

    def count_tokens(self, text: str) -> int:
        return len(self.tok(text, add_special_tokens=False)["input_ids"])

    def generate(self, user_prompt: str) -> str:
        """Chunked prefill: the long prompt is pushed through the model in slices of
        `prefill_chunk` tokens into a KV cache (the lm_head is skipped), so peak memory is
        bounded by chunk x context instead of context^2 (this is what OOM'd on a T4).
        Generation then continues from the cache, so the model still sees the FULL prompt.
        Used for article mode and as the sequential fallback for page batches."""
        torch = self.torch
        from transformers import DynamicCache

        inputs = self.tok(self._chat(user_prompt), return_tensors="pt").to(self.model.device)
        ids, n = inputs["input_ids"], inputs["input_ids"].shape[1]
        cache = DynamicCache()
        with torch.inference_mode():
            pos = 0
            while n - 1 - pos > 0:  # leave the last token for generate()
                end = min(pos + self.prefill_chunk, n - 1)
                self.model.model(input_ids=ids[:, pos:end], past_key_values=cache, use_cache=True)
                pos = end
            out = self.model.generate(input_ids=ids, attention_mask=torch.ones_like(ids),
                                      past_key_values=cache, max_new_tokens=self.max_new_tokens,
                                      do_sample=False, temperature=None, top_p=None, top_k=None,
                                      pad_token_id=self.tok.eos_token_id)
        res = self.tok.decode(out[0, n:], skip_special_tokens=True)
        del cache, out
        torch.cuda.empty_cache()
        return res

    def generate_batch(self, prompts: Sequence[str]) -> List[str]:
        """Batched generation with left padding. `logits_to_keep=1` keeps only the last
        position's logits, which avoids a batch x seq x vocab tensor (several GB on a T4).
        Falls back to the sequential chunked-prefill path if batching is unsupported or
        runs out of memory."""
        torch = self.torch
        if len(prompts) <= 1:
            return [self.generate(prompts[0])] if prompts else []

        chats = [self._chat(p) for p in prompts]
        tok = self.tok
        old_side = tok.padding_side
        tok.padding_side = "left"
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        try:
            enc = tok(chats, return_tensors="pt", padding=True, truncation=False).to(self.model.device)
            n_prompt = enc["input_ids"].shape[1]
            try:
                with torch.inference_mode():
                    out = self.model.generate(
                        **enc, max_new_tokens=self.max_new_tokens,
                        do_sample=False, temperature=None, top_p=None, top_k=None,
                        pad_token_id=tok.pad_token_id, logits_to_keep=1)
            except TypeError:  # older transformers without logits_to_keep
                with torch.inference_mode():
                    out = self.model.generate(
                        **enc, max_new_tokens=self.max_new_tokens,
                        do_sample=False, temperature=None, top_p=None, top_k=None,
                        pad_token_id=tok.pad_token_id)
            texts = tok.batch_decode(out[:, n_prompt:], skip_special_tokens=True)
            del out
            torch.cuda.empty_cache()
            return texts
        except (RuntimeError, ValueError) as exc:
            print(f"[batch] sequential fallback ({type(exc).__name__}: {exc})", file=sys.stderr)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return [self.generate(p) for p in prompts]
        finally:
            tok.padding_side = old_side


class MockBackend:
    """Regex heading finder - NO LLM. Only for testing the pipeline/evaluation without a GPU."""
    ctx = 10 ** 9
    PATTERNS = [
        ("abstract", r"\babstract\b"), ("keywords", r"\b(key\s?words|index terms)\b"),
        ("introduction", r"(^|\n)\s*(\d+\.?|[IVX]+\.?)?\s*introduction"),
        ("background", r"related\s?work|background|literature review"),
        ("method", r"(^|\n)\s*(\d+\.?|[IVX]+\.?)?\s*(methods?|methodology|approach)\b"),
        ("experiments", r"(^|\n)\s*(\d+\.?)?\s*experiments?\b"),
        ("results", r"(^|\n)\s*(\d+\.?)?\s*results"), ("discussion", r"(^|\n)\s*(\d+\.?)?\s*discussion"),
        ("conclusion", r"(^|\n)\s*(\d+\.?|[IVX]+\.?)?\s*conclusions?"),
        ("references", r"(^|\n)\s*references\s*(\n|$)"),
        ("declarations", r"acknowledge?ments?|conflicts? of interest|author contributions|funding"),
        ("supplementary", r"(^|\n)\s*appendix|supplementary"),
    ]

    def count_tokens(self, text: str) -> int:
        return len(text) // 3

    @staticmethod
    def _body(prompt: str) -> str:
        parts = prompt.split('"""')
        return parts[1] if len(parts) >= 3 else prompt

    def generate(self, user_prompt: str) -> str:
        body = self._body(user_prompt)
        found = [{"heading": l, "label": l} for l, p in self.PATTERNS
                 if re.search(p, body, flags=re.I)]
        found.insert(0, {"heading": "title", "label": "title"})
        return json.dumps({"sections": found})

    def generate_batch(self, prompts: Sequence[str]) -> List[str]:
        return [self.generate(p) for p in prompts]

# --------------------------------------------------------------------------------------
# Inference
# --------------------------------------------------------------------------------------

def make_chunks(pages: List[str], budget_tokens: int, count) -> List[str]:
    """Greedy page-aligned chunks, each <= budget_tokens (only used for very long articles)."""
    chunks, cur, cur_n = [], [], 0
    for p in pages:
        n = count(p)
        if cur and cur_n + n > budget_tokens:
            chunks.append("\n\n".join(cur)); cur, cur_n = [], 0
        cur.append(p); cur_n += n
    if cur:
        chunks.append("\n\n".join(cur))
    return chunks


def predict_article(backend, art: dict, labels: List[str], max_input_tokens: int,
                    long_doc: str) -> dict:
    """Pass the FULL article to the LLM. If it exceeds the context window, fall back to
    page-aligned chunks (long_doc='chunk') and union the results, or truncate."""
    labelset = set(labels)
    overhead = backend.count_tokens(build_user_prompt("", labels)) + 200
    budget = max_input_tokens - overhead
    n_tok = backend.count_tokens(art["text"])

    mode, texts = "full", [art["text"]]
    if n_tok > budget:
        if long_doc == "chunk":
            mode, texts = "chunked", make_chunks(art["pages"], budget, backend.count_tokens)
        else:  # truncate: keep head and tail (title/abstract + references/appendix)
            mode = "truncated"
            half = len(art["text"]) * budget // n_tok // 2
            texts = [art["text"][:half] + "\n...\n" + art["text"][-half:]]

    raw_outputs, items = [], []
    for t in texts:
        out = backend.generate(build_user_prompt(t, labels, partial=mode == "chunked"))
        raw_outputs.append(out)
        items.extend(parse_model_output(out))

    pred, invalid = set(), []
    for heading, raw in items:
        lab = validate_label(raw, labelset)
        (pred.add(lab) if lab else invalid.append(raw))
    return {"manuscript_id": art["manuscript_id"], "pred": sorted(pred), "gt": art["gt"],
            "mode": mode, "n_input_tokens": n_tok, "n_chunks": len(texts),
            "sections": [{"heading": h, "label": r} for h, r in items],
            "invalid_labels": invalid, "raw_output": raw_outputs}


def _generate_many(backend, prompts: Sequence[str]) -> List[str]:
    batch_fn = getattr(backend, "generate_batch", None)
    if batch_fn is not None and len(prompts) > 1:
        return list(batch_fn(prompts))
    return [backend.generate(p) for p in prompts]


def clip_page_text(backend, text: str, labels: Sequence[str],
                   max_input_tokens: int, position: Optional[str]) -> Tuple[str, bool, int]:
    """Char-level clip so the full prompt fits the context budget. Returns
    (text, truncated, estimated_tokens_of_original)."""
    n_tok = backend.count_tokens(text)
    overhead = backend.count_tokens(build_page_prompt("", labels, position)) + 32
    budget = max_input_tokens - overhead
    if n_tok <= budget or not text:
        return text, False, n_tok
    ratio = n_tok / max(1, len(text))
    max_chars = max(200, int(budget / max(ratio, 1e-6) * 0.95))
    return text[:max_chars] + "\n[...truncated...]", True, n_tok


def predict_pages(backend, pages: Sequence[dict], labels: List[str],
                  max_input_tokens: int, batch_size: int = 4,
                  position_context: bool = False, repair: bool = True) -> List[dict]:
    """Page-by-page inference with batching and a single repair retry per failed page."""
    labelset = set(labels)
    results: List[dict] = []

    for start in range(0, len(pages), max(1, batch_size)):
        batch = list(pages[start:start + max(1, batch_size)])
        prompts, prepared = [], []
        for page in batch:
            position = None
            if position_context and page.get("page") and page.get("n_pages"):
                position = f"Page {page['page']} of {page['n_pages']}."
            text, truncated, n_tok = clip_page_text(backend, page["text"], labels,
                                                    max_input_tokens, position)
            prompts.append(build_page_prompt(text, labels, position))
            prepared.append((page, text, position, truncated, n_tok))

        outputs = _generate_many(backend, prompts)

        for (page, text, position, truncated, n_tok), out in zip(prepared, outputs):
            raw_labels, parsed_ok = parse_page_output(out)
            pred, invalid = set(), []
            for raw in raw_labels:
                lab = validate_label(raw, labelset)
                (pred.add(lab) if lab else invalid.append(raw))

            repaired = False
            if repair and (not parsed_ok or invalid or not pred):
                if not parsed_ok:
                    problem = "the answer was not valid JSON in the required format"
                elif invalid:
                    problem = "these labels are not in the allowed list: " + ", ".join(map(str, invalid))
                else:
                    problem = "the answer was empty; every page has at least one section"
                repair_prompt = build_page_prompt(
                    text, labels, position,
                    repair={"previous": out.strip()[:800], "problem": problem})
                out2 = backend.generate(repair_prompt)
                raw2, parsed2 = parse_page_output(out2)
                repaired = True
                pred, invalid = set(), []
                for raw in raw2:
                    lab = validate_label(raw, labelset)
                    (pred.add(lab) if lab else invalid.append(raw))
                out, parsed_ok = out2, parsed2

            results.append({
                "sample_id": page["sample_id"],
                "manuscript_id": page["manuscript_id"],
                "field": page.get("field"),
                "page": page.get("page"),
                "n_pages": page.get("n_pages"),
                "pred": sorted(pred),
                "gt": page["gt"],
                "invalid_labels": invalid,
                "parse_error": not parsed_ok,
                "repair_used": repaired,
                "truncated": truncated,
                "n_input_tokens": n_tok,
                "raw_output": out,
            })
    return results

# --------------------------------------------------------------------------------------
# Evaluation (no sklearn dependency)
# --------------------------------------------------------------------------------------

def _prf(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def _set_metrics(P: List[Set[str]], G: List[Set[str]], labels: Sequence[str]) -> dict:
    """Multi-label set metrics over already-binarised sample sets."""
    n = len(P)
    per = {}
    for l in labels:
        tp = sum(l in p and l in g for p, g in zip(P, G))
        fp = sum(l in p and l not in g for p, g in zip(P, G))
        fn = sum(l not in p and l in g for p, g in zip(P, G))
        tn = n - tp - fp - fn
        pr, rc, f1 = _prf(tp, fp, fn)
        per[l] = {"precision": pr, "recall": rc, "f1": f1, "accuracy": (tp + tn) / n if n else 0.0,
                  "support": tp + fn, "predicted": tp + fp, "tp": tp, "fp": fp, "fn": fn, "tn": tn}

    TP = sum(v["tp"] for v in per.values()); FP = sum(v["fp"] for v in per.values())
    FN = sum(v["fn"] for v in per.values()); TN = sum(v["tn"] for v in per.values())
    mp, mr, mf = _prf(TP, FP, FN)
    macro = {k: sum(v[k] for v in per.values()) / len(per) for k in ("precision", "recall", "f1")}
    sup = sum(v["support"] for v in per.values())
    weighted = {k: sum(v[k] * v["support"] for v in per.values()) / sup if sup else 0.0
                for k in ("precision", "recall", "f1")}
    sp, sr, sf, jac = [], [], [], []
    for p, g in zip(P, G):
        a, b, c = _prf(len(p & g), len(p - g), len(g - p))
        sp.append(a); sr.append(b); sf.append(c)
        jac.append(len(p & g) / len(p | g) if p | g else 1.0)
    n_lab = len(labels) * n
    return {
        "per_label": per,
        "micro": {"precision": mp, "recall": mr, "f1": mf},
        "macro": macro,
        "weighted": weighted,
        "samples": {"precision": sum(sp) / n if n else 0.0,
                    "recall": sum(sr) / n if n else 0.0,
                    "f1": sum(sf) / n if n else 0.0},
        "jaccard_samples": sum(jac) / n if n else 0.0,
        "hamming_accuracy_micro": (TP + TN) / n_lab if n_lab else 0.0,
        "hamming_loss": (FP + FN) / n_lab if n_lab else 0.0,
    }


def evaluate(preds: List[dict], labels: List[str]) -> dict:
    """Article-level evaluation (distinct predicted set vs distinct gold set)."""
    n = len(preds)
    P = [set(x["pred"]) for x in preds]
    G = [set(x["gt"]) for x in preds]

    exact = sum(p == g for p, g in zip(P, G)) / n  # LLM_extracted == distinct(gt)

    common = _set_metrics(P, G, labels)
    per = common["per_label"]

    by_field = defaultdict(list)
    for x, p, g in zip(preds, P, G):
        by_field[x.get("field", "?")].append(p == g)

    fn_c, fp_c = Counter(), Counter()
    for p, g in zip(P, G):
        fn_c.update(g - p); fp_c.update(p - g)

    return {
        "n_articles": n,
        "exact_match_accuracy": exact,
        "exact_match_count": sum(p == g for p, g in zip(P, G)),
        "hamming_accuracy_micro": common["hamming_accuracy_micro"],
        "accuracy_macro_per_label": sum(v["accuracy"] for v in per.values()) / len(per),
        "micro": common["micro"],
        "macro": common["macro"],
        "weighted": common["weighted"],
        "samples": common["samples"],
        "jaccard_samples": common["jaccard_samples"],
        "per_label": per,
        "exact_match_by_field": {k: sum(v) / len(v) for k, v in sorted(by_field.items())},
        "most_missed_labels": fn_c.most_common(), "most_spurious_labels": fp_c.most_common(),
    }


def _percentile(values: Sequence[float], q: float) -> float:
    s = sorted(values)
    if not s:
        return 0.0
    k = (len(s) - 1) * q / 100.0
    lo = int(k); hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def bootstrap_page_ci(preds: List[dict], labels: List[str],
                      n_boot: int = 1000, seed: int = 42) -> Optional[dict]:
    """95% CI for page metrics, resampling ARTICLES (not pages) with replacement so that
    correlated pages from the same manuscript stay together."""
    if n_boot <= 0 or not preds:
        return None
    by_article: Dict[str, List[dict]] = defaultdict(list)
    for p in preds:
        by_article[p["manuscript_id"]].append(p)
    article_ids = sorted(by_article)
    rng = random.Random(seed)

    subset, hamming, micro, macro = [], [], [], []
    for _ in range(n_boot):
        picked = [rng.choice(article_ids) for _ in article_ids]
        sample = [p for a in picked for p in by_article[a]]
        P = [set(x["pred"]) for x in sample]
        G = [set(x["gt"]) for x in sample]
        m = _set_metrics(P, G, labels)
        subset.append(sum(p == g for p, g in zip(P, G)) / len(P))
        hamming.append(m["hamming_loss"])
        micro.append(m["micro"]["f1"])
        macro.append(m["macro"]["f1"])

    def interval(values: Sequence[float]) -> dict:
        return {"lower": _percentile(values, 2.5), "upper": _percentile(values, 97.5)}

    return {
        "n_boot": int(n_boot),
        "n_articles": len(article_ids),
        "subset_accuracy": interval(subset),
        "hamming_loss": interval(hamming),
        "micro_f1": interval(micro),
        "macro_f1": interval(macro),
    }


def evaluate_pages(preds: List[dict], labels: List[str]) -> dict:
    """Page-level multi-label evaluation (one sample = one page)."""
    n = len(preds)
    P = [set(x["pred"]) for x in preds]
    G = [set(x["gt"]) for x in preds]
    common = _set_metrics(P, G, labels)

    fn_c, fp_c = Counter(), Counter()
    for p, g in zip(P, G):
        fn_c.update(g - p); fp_c.update(p - g)

    # per field
    by_field = defaultdict(lambda: {"P": [], "G": []})
    for x, p, g in zip(preds, P, G):
        by_field[x.get("field") or "?"]["P"].append(p)
        by_field[x.get("field") or "?"]["G"].append(g)
    per_field = {}
    for f, d in sorted(by_field.items()):
        m = _set_metrics(d["P"], d["G"], labels)
        per_field[f] = {
            "n_pages": len(d["P"]),
            "subset_accuracy": sum(p == g for p, g in zip(d["P"], d["G"])) / len(d["P"]),
            "micro_f1": m["micro"]["f1"],
            "macro_f1": m["macro"]["f1"],
        }

    # manual vs inherited
    per_source = {}
    src = defaultdict(lambda: {"P": [], "G": []})
    for x, p, g in zip(preds, P, G):
        key = x.get("label_source") or ("manual" if x.get("is_section_start") else "inherited")
        src[key]["P"].append(p); src[key]["G"].append(g)
    for key, d in sorted(src.items()):
        if not d["P"]:
            continue
        m = _set_metrics(d["P"], d["G"], labels)
        per_source[key] = {
            "n_pages": len(d["P"]),
            "subset_accuracy": sum(p == g for p, g in zip(d["P"], d["G"])) / len(d["P"]),
            "micro_f1": m["micro"]["f1"],
            "macro_f1": m["macro"]["f1"],
        }

    return {
        "n_pages": n,
        "n_articles": len({x["manuscript_id"] for x in preds}),
        "subset_accuracy": sum(p == g for p, g in zip(P, G)) / n if n else 0.0,
        "subset_correct": sum(p == g for p, g in zip(P, G)),
        "hamming_loss": common["hamming_loss"],
        "hamming_accuracy_micro": common["hamming_accuracy_micro"],
        "micro": common["micro"],
        "macro": common["macro"],
        "weighted": common["weighted"],
        "samples": common["samples"],
        "samples_f1": common["samples"]["f1"],
        "jaccard_samples": common["jaccard_samples"],
        "mean_labels_predicted": sum(len(p) for p in P) / n if n else 0.0,
        "mean_labels_gold": sum(len(g) for g in G) / n if n else 0.0,
        "pages_with_no_prediction": sum(not p for p in P),
        "parse_failures": sum(bool(x.get("parse_error")) for x in preds),
        "n_invalid_labels": sum(len(x.get("invalid_labels", [])) for x in preds),
        "n_pages_with_invalid": sum(bool(x.get("invalid_labels")) for x in preds),
        "n_repairs": sum(bool(x.get("repair_used")) for x in preds),
        "n_truncated": sum(bool(x.get("truncated")) for x in preds),
        "invalid_label_counts": Counter(
            l for x in preds for l in x.get("invalid_labels", [])).most_common(),
        "per_label": common["per_label"],
        "per_field": per_field,
        "per_source": per_source,
        "most_missed_labels": fn_c.most_common(),
        "most_spurious_labels": fp_c.most_common(),
    }


def format_report(m: dict, labels: List[str]) -> str:
    L = []
    L.append("=" * 78)
    L.append("SECTION IDENTIFICATION - EVALUATION REPORT")
    L.append("=" * 78)
    L.append(f"Articles evaluated : {m['n_articles']}")
    L.append(f"Exact-match accuracy (LLM set == distinct GT set): "
             f"{m['exact_match_accuracy']:.4f}  ({m['exact_match_count']}/{m['n_articles']})")
    L.append(f"Hamming accuracy (micro, all article x label cells): {m['hamming_accuracy_micro']:.4f}")
    L.append(f"Accuracy (macro, mean of per-label accuracy)       : {m['accuracy_macro_per_label']:.4f}")
    L.append(f"Jaccard (samples)                                  : {m['jaccard_samples']:.4f}")
    L.append(f"Invalid (out-of-taxonomy) labels rejected          : {m.get('n_invalid_labels', 0)} "
             f"in {m.get('n_articles_with_invalid', 0)} articles")
    L.append("")
    L.append(f"{'averaging':<12}{'precision':>11}{'recall':>11}{'f1':>11}")
    for k in ("micro", "macro", "weighted", "samples"):
        v = m[k]; L.append(f"{k:<12}{v['precision']:>11.4f}{v['recall']:>11.4f}{v['f1']:>11.4f}")
    L.append("")
    L.append(f"{'label':<15}{'prec':>8}{'recall':>8}{'f1':>8}{'acc':>8}{'support':>9}{'pred':>7}"
             f"{'TP':>5}{'FP':>5}{'FN':>5}")
    for l in labels:
        v = m["per_label"][l]
        L.append(f"{l:<15}{v['precision']:>8.3f}{v['recall']:>8.3f}{v['f1']:>8.3f}{v['accuracy']:>8.3f}"
                 f"{v['support']:>9}{v['predicted']:>7}{v['tp']:>5}{v['fp']:>5}{v['fn']:>5}")
    L.append("")
    L.append("Exact-match by field: " + ", ".join(f"{k}={v:.3f}" for k, v in m["exact_match_by_field"].items()))
    L.append("Most missed labels  : " + ", ".join(f"{k}({v})" for k, v in m["most_missed_labels"][:8]))
    L.append("Most spurious labels: " + ", ".join(f"{k}({v})" for k, v in m["most_spurious_labels"][:8]))
    return "\n".join(L)


def format_page_report(m: dict, labels: List[str]) -> str:
    L = []
    L.append("=" * 78)
    L.append("PAGE-LEVEL SECTION IDENTIFICATION - EVALUATION REPORT")
    L.append("=" * 78)
    L.append(f"Pages evaluated    : {m['n_pages']}  ({m['n_articles']} articles)")
    L.append(f"Subset accuracy (predicted set == gold set) : {m['subset_accuracy']:.4f}  "
             f"({m['subset_correct']}/{m['n_pages']})")
    L.append(f"Hamming loss (micro, page x label cells)    : {m['hamming_loss']:.4f}")
    L.append(f"Hamming accuracy (micro)                    : {m['hamming_accuracy_micro']:.4f}")
    L.append(f"Labels per page    : predicted {m['mean_labels_predicted']:.2f} | "
             f"gold {m['mean_labels_gold']:.2f}")
    L.append(f"Pages with no prediction: {m['pages_with_no_prediction']} | "
             f"parse failures: {m['parse_failures']} | repairs: {m['n_repairs']} | "
             f"truncated: {m['n_truncated']}")
    L.append(f"Invalid (out-of-taxonomy) labels rejected: {m['n_invalid_labels']} "
             f"in {m['n_pages_with_invalid']} pages")
    ci = m.get("bootstrap_95ci")
    if ci:
        L.append("")
        L.append(f"95% CI (article-clustered bootstrap, n={ci['n_boot']}, "
                 f"{ci['n_articles']} articles):")
        L.append(f"  subset accuracy [{ci['subset_accuracy']['lower']:.4f}, {ci['subset_accuracy']['upper']:.4f}]  "
                 f"hamming loss [{ci['hamming_loss']['lower']:.4f}, {ci['hamming_loss']['upper']:.4f}]")
        L.append(f"  micro F1        [{ci['micro_f1']['lower']:.4f}, {ci['micro_f1']['upper']:.4f}]  "
                 f"macro F1   [{ci['macro_f1']['lower']:.4f}, {ci['macro_f1']['upper']:.4f}]")
    L.append("")
    L.append(f"{'averaging':<12}{'precision':>11}{'recall':>11}{'f1':>11}")
    for k in ("micro", "macro", "weighted", "samples"):
        v = m[k]; L.append(f"{k:<12}{v['precision']:>11.4f}{v['recall']:>11.4f}{v['f1']:>11.4f}")
    L.append("")
    L.append(f"{'label':<15}{'prec':>8}{'recall':>8}{'f1':>8}{'acc':>8}{'support':>9}{'pred':>7}"
             f"{'TP':>5}{'FP':>5}{'FN':>5}")
    for l in labels:
        v = m["per_label"][l]
        L.append(f"{l:<15}{v['precision']:>8.3f}{v['recall']:>8.3f}{v['f1']:>8.3f}{v['accuracy']:>8.3f}"
                 f"{v['support']:>9}{v['predicted']:>7}{v['tp']:>5}{v['fp']:>5}{v['fn']:>5}")
    L.append("")
    L.append("Per field (subset accuracy / micro F1 / macro F1):")
    for f, v in m["per_field"].items():
        L.append(f"  {f:<8} n={v['n_pages']:<4} subset={v['subset_accuracy']:.4f}  "
                 f"micro={v['micro_f1']:.4f}  macro={v['macro_f1']:.4f}")
    L.append("Per page source (subset accuracy / micro F1 / macro F1):")
    for s, v in m["per_source"].items():
        L.append(f"  {s:<10} n={v['n_pages']:<4} subset={v['subset_accuracy']:.4f}  "
                 f"micro={v['micro_f1']:.4f}  macro={v['macro_f1']:.4f}")
    L.append("")
    L.append("Most missed labels  : " + ", ".join(f"{k}({v})" for k, v in m["most_missed_labels"][:8]))
    L.append("Most spurious labels: " + ", ".join(f"{k}({v})" for k, v in m["most_spurious_labels"][:8]))
    if m["invalid_label_counts"]:
        L.append("Rejected labels     : " + ", ".join(
            f"{k}({v})" for k, v in m["invalid_label_counts"][:8]))
    return "\n".join(L)

# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------

DEFAULT_OUT = {"page": "results/llm_pages", "article": "results/llm_articles"}
DEFAULT_MAX_NEW = {"page": 160, "article": 1024}
DEFAULT_MAX_INPUT = {"page": 3072, "article": 30000}


def run_articles(args) -> int:
    articles = load_articles(args.data)
    labels = all_labels(articles)
    if args.limit:
        articles = articles[: args.limit]
    print(f"[data] {len(articles)} articles | {len(labels)} target labels: {labels}")

    pred_path = Path(args.out) / "predictions.jsonl"
    done: Dict[str, dict] = {}
    if pred_path.exists() and not args.no_resume:
        done = {d["manuscript_id"]: d for d in _read_jsonl(str(pred_path))}
        print(f"[resume] {len(done)} cached predictions")
    todo = [a for a in articles if a["manuscript_id"] not in done]

    if todo:
        if args.backend == "mock":
            backend = MockBackend()
        else:
            print(f"[model] loading {args.model} (4-bit) ...")
            backend = HFQwenBackend(args.model, args.max_new_tokens, args.no_4bit, args.attn,
                                    args.prefill_chunk)
            args.max_input_tokens = min(args.max_input_tokens, backend.ctx - args.max_new_tokens - 64)
        mode = "a" if done else "w"
        with open(pred_path, mode, encoding="utf-8") as f:
            for i, art in enumerate(todo, 1):
                t0 = time.time()
                res = predict_article(backend, art, labels, args.max_input_tokens, args.long_doc)
                res["field"] = art["field"]
                done[art["manuscript_id"]] = res
                f.write(json.dumps(res, ensure_ascii=False) + "\n"); f.flush()
                ok = set(res["pred"]) == set(res["gt"])
                print(f"[{i}/{len(todo)}] {art['manuscript_id']:<24} {res['mode']:<9} "
                      f"{res['n_input_tokens']:>6} tok  {time.time()-t0:5.1f}s  exact={ok}")

    preds = [done[a["manuscript_id"]] for a in articles]
    for p, a in zip(preds, articles):
        p["field"] = a["field"]
    metrics = evaluate(preds, labels)
    metrics["n_invalid_labels"] = sum(len(p.get("invalid_labels", [])) for p in preds)
    metrics["n_articles_with_invalid"] = sum(bool(p.get("invalid_labels")) for p in preds)
    metrics["invalid_label_counts"] = Counter(l for p in preds for l in p.get("invalid_labels", [])).most_common()
    report = format_report(metrics, labels)
    print("\n" + report)

    out = Path(args.out)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (out / "report.txt").write_text(report, encoding="utf-8")
    _write_per_label(out, metrics, labels)
    print(f"\n[saved] {out}/predictions.jsonl, metrics.json, per_label_metrics.csv, report.txt")
    return 0


def run_pages(args) -> int:
    pages = load_pages(args.data)
    labels = all_labels(pages)
    if args.limit:
        pages = pages[: args.limit]
    print(f"[data] {len(pages)} pages | {len(labels)} target labels: {labels}")
    if args.batch_size > 1:
        print(f"[batch] batch_size={args.batch_size} (lower it if the GPU runs out of memory)")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    pred_path = out / "predictions.jsonl"
    done: Dict[str, dict] = {}
    if pred_path.exists() and not args.no_resume:
        done = {d["sample_id"]: d for d in _read_jsonl(str(pred_path))}
        print(f"[resume] {len(done)} cached page predictions")
    todo = [p for p in pages if p["sample_id"] not in done]

    if todo:
        if args.backend == "mock":
            backend = MockBackend()
        else:
            print(f"[model] loading {args.model} (4-bit) ...")
            backend = HFQwenBackend(args.model, args.max_new_tokens, args.no_4bit, args.attn,
                                    args.prefill_chunk)
            args.max_input_tokens = min(args.max_input_tokens, backend.ctx - args.max_new_tokens - 64)
        mode = "a" if done else "w"
        written = 0
        t_start = time.time()
        with open(pred_path, mode, encoding="utf-8") as f:
            for start in range(0, len(todo), max(1, args.batch_size)):
                batch = todo[start:start + max(1, args.batch_size)]
                t0 = time.time()
                results = predict_pages(
                    backend, batch, labels, args.max_input_tokens,
                    batch_size=args.batch_size,
                    position_context=args.page_position_context,
                    repair=not args.no_repair,
                )
                for res in results:
                    done[res["sample_id"]] = res
                    f.write(json.dumps(res, ensure_ascii=False) + "\n"); f.flush()
                    written += 1
                n_ok = sum(set(r["pred"]) == set(r["gt"]) for r in results)
                print(f"[{written}/{len(todo)}] batch {start // max(1, args.batch_size) + 1} "
                      f"subset={n_ok}/{len(results)}  {time.time()-t0:5.1f}s  "
                      f"elapsed {time.time()-t_start:6.1f}s")

    preds = [done[p["sample_id"]] for p in pages]
    for pred, page in zip(preds, pages):
        pred.setdefault("field", page.get("field"))
        pred.setdefault("label_source", page.get("label_source"))
        pred.setdefault("is_section_start", page.get("is_section_start"))

    metrics = evaluate_pages(preds, labels)
    metrics["model"] = args.model if args.backend != "mock" else "mock"
    metrics["unit"] = "page"
    metrics["split"] = Path(args.data).stem
    metrics["batch_size"] = args.batch_size
    metrics["bootstrap_95ci"] = bootstrap_page_ci(preds, labels, args.bootstrap)
    report = format_page_report(metrics, labels)
    print("\n" + report)

    (out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (out / "report.txt").write_text(report, encoding="utf-8")
    _write_per_label(out, metrics, labels)
    print(f"\n[saved] {out}/predictions.jsonl, metrics.json, per_label_metrics.csv, report.txt")
    return 0


def _write_per_label(out: Path, metrics: dict, labels: Sequence[str]) -> None:
    with open(out / "per_label_metrics.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["label", "precision", "recall", "f1", "accuracy", "support", "predicted",
                    "tp", "fp", "fn", "tn"])
        for l in labels:
            v = metrics["per_label"][l]
            w.writerow([l] + [round(v[k], 4) if isinstance(v[k], float) else v[k]
                              for k in ("precision", "recall", "f1", "accuracy", "support",
                                        "predicted", "tp", "fp", "fn", "tn")])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--unit", choices=["page", "article"], default="page",
                    help="page (default) = one page per prompt; article = whole-article ablation")
    ap.add_argument("--data", default="data/splits/test.jsonl")
    ap.add_argument("--out", default=None,
                    help="output dir (default: results/llm_pages or results/llm_articles)")
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--backend", choices=["hf", "mock"], default="hf")
    ap.add_argument("--no-4bit", action="store_true", help="load in bf16/fp16 instead (debug)")
    ap.add_argument("--attn", default="sdpa", help="sdpa | flash_attention_2 | eager")
    ap.add_argument("--max-input-tokens", type=int, default=None,
                    help="prompt budget (default: 3072 page / 30000 article)")
    ap.add_argument("--max-new-tokens", type=int, default=None,
                    help="generation budget (default: 160 page / 1024 article)")
    ap.add_argument("--prefill-chunk", type=int, default=1024,
                    help="prompt slice size for chunked prefill (lower = less GPU memory)")
    ap.add_argument("--batch-size", type=int, default=4,
                    help="pages per forward pass in page mode (use 1-2 on a small GPU)")
    ap.add_argument("--page-position-context", action="store_true",
                    help="include 'Page X of N' in the page prompt (off by default: text only)")
    ap.add_argument("--no-repair", action="store_true",
                    help="disable the single repair retry for malformed page answers")
    ap.add_argument("--bootstrap", type=int, default=1000,
                    help="article-clustered bootstrap resamples for page metrics (0 = off)")
    ap.add_argument("--long-doc", choices=["chunk", "truncate"], default="chunk",
                    help="article mode: what to do if the article does not fit the context window")
    ap.add_argument("--limit", type=int, default=None,
                    help="only the first N samples (pages in page mode, articles in article mode)")
    ap.add_argument("--no-resume", action="store_true", help="ignore cached predictions")
    args = ap.parse_args(argv)

    if args.max_input_tokens is None:
        args.max_input_tokens = DEFAULT_MAX_INPUT[args.unit]
    if args.max_new_tokens is None:
        args.max_new_tokens = DEFAULT_MAX_NEW[args.unit]
    if args.out is None:
        args.out = DEFAULT_OUT[args.unit]
    Path(args.out).mkdir(parents=True, exist_ok=True)

    if args.unit == "article":
        return run_articles(args)
    return run_pages(args)


if __name__ == "__main__":
    sys.exit(main())
