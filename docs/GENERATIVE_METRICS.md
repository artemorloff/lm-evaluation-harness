# Generative and reference-based metrics (MERA fork)

This document records **implemented features**, dependencies, aggregation behavior, and how to run tests. It applies to the `lm-evaluation-harness` copy under the MERA repository.

## Implementation log

| Date | Change |
|------|--------|
| 2026-04 | **F1_gen fix**: `compute_f1_gen` now calls `transformers.data.metrics.squad_metrics.compute_f1(prediction, ground_truth)` in the correct argument order (was reversed). |
| 2026-04 | **New module** [`lm_eval/api/metrics_generative.py`](../lm_eval/api/metrics_generative.py): optional metrics registered via the same `@register_metric` / `aggregation="mean"` pattern as core metrics. Imported from [`lm_eval/api/metrics.py`](../lm_eval/api/metrics.py) at load time. |
| 2026-04 | **`embedding_cosine`**: OpenAI-compatible `POST {api_base}/v1/embeddings`; cosine similarity between reference and prediction embeddings; optional LRU cache (`use_cache`, default true); env vars `OPENAI_API_KEY` / `LM_EVAL_EMBEDDING_API_KEY`. Requires `lm_eval[api]` (`requests`). |
| 2026-04 | **`levenshtein`**: Normalized similarity `1 - distance/max(len)` by default; `normalized: false` for raw edit distance. Requires `rapidfuzz` (`lm_eval[generative_metrics]`). |
| 2026-04 | **`llm_judge`**: OpenAI-compatible `POST {api_base}/v1/chat/completions`; `judge_prompt` with `{reference}` and `{prediction}`; optional `score_regex`; default score = last float in the reply. Env: `OPENAI_API_KEY` / `LM_EVAL_JUDGE_API_KEY`. Requires `lm_eval[api]`. |
| 2026-04 | **`meteor`**: NLTK METEOR; may auto-download WordNet / punkt. Requires `nltk` (`generative_metrics`). |
| 2026-04 | **`bertscore`**: `bert-score` F1 mean for the pair; `lang` and other kwargs forwarded. Requires `bert-score`, PyTorch (`generative_metrics`). |
| 2026-04 | **`comet`**: `unbabel-comet` with default checkpoint `Unbabel/wmt22-comet-da`; per-doc `predict` with `src=ref`, `mt=pred`, `ref=ref` for monolingual-style evaluation. Requires `unbabel-comet`. |
| 2026-04 | **`bleurt`**: HuggingFace `evaluate.load("bleurt")`; may require TensorFlow or extra BLEURT deps on first use. |
| 2026-04 | **`token_overlap_f1`**: Whitespace token multiset overlap F1 (no extra deps). **Bugfix**: precision/recall denominators use `len(pred)` / `len(ref)` (was broken `pred_set`/`ref_set`). |
| 2026-04 | **`sentence_bleu`**: `sacrebleu.sentence_bleu` per example, aggregated with **`mean`** (distinct from corpus **`bleu`**). |
| 2026-04 | **Dev environment**: [`scripts/bootstrap_dev_venv.sh`](../scripts/bootstrap_dev_venv.sh) creates `.venv` and installs `lm_eval[dev,api,hf,generative_metrics]` **without** `pip install -U` (upgrade flags avoided per team policy). |
| 2026-04 | **Tests**: [`tests/test_generative_metrics.py`](../tests/test_generative_metrics.py) — unit tests for core + generative metrics; mocks for HTTP, COMET, BLEURT; optional NLTK/BERTScore smoke tests. |
| 2026-04 | **MERA CheGeKa**: [`benchmark_tasks/tape/chegeka_generative_metrics.yaml`](../../benchmark_tasks/tape/chegeka_generative_metrics.yaml) — second task config calling [`process_results_generative_metrics`](../../benchmark_tasks/tape/utils.py) (max over `;`-separated references). |
| 2026-04 | **NumPy**: core dependency pinned to **`numpy>=1.26.0,<1.27`** in `pyproject.toml` for ABI compatibility with `pandas` / `scikit-learn` in typical venvs. |
| 2026-04 | **Import hygiene**: `transformers.squad_metrics` and `rouge_score` are **lazy-imported** inside `compute_f1_gen` / `compute_rouge_fn` so `import lm_eval.api.metrics` does not pull sklearn/pandas at import time. |
| 2026-04 | **CI / broken conda**: tests skip with a clear message if the numpy/pandas/sklearn wheels are ABI-incompatible (`numpy.dtype size changed`). Prefer a clean venv via [`scripts/bootstrap_dev_venv.sh`](../scripts/bootstrap_dev_venv.sh); align numpy/pandas/sklearn builds in an existing env only per your team’s packaging policy. |

## Already present in core `metrics.py` (unchanged behavior except F1_gen)

| Metric | Role | Aggregation |
|--------|------|-------------|
| `bleu` | Corpus BLEU (sacrebleu) | **`bleu`** (corpus-level over all pairs) |
| `chrf`, `ter` | Corpus chrF++, TER | **`chrf`**, **`ter`** |
| `exact_match` | Per-example EM | **`mean`** |
| `rouge` | ROUGE F-measure (`rouge_score`) | **`mean`** |
| `f1_gen` | SQuAD-style token F1 between pred and ref | **`mean`** (fixed pred/ref order) |

### Corpus BLEU vs sentence BLEU

- **`bleu`** + aggregation **`bleu`**: one corpus score (passthrough doc pairs, then `sacrebleu.corpus_bleu` in the aggregation step).
- **`sentence_bleu`** + aggregation **`mean`**: one sacreBLEU score per document, then mean over documents.

## Dependencies

Install optional sets:

```bash
pip install "lm_eval[generative_metrics]"   # rapidfuzz, nltk, bert-score, unbabel-comet
pip install "lm_eval[api]"                  # requests — required for embedding_cosine & llm_judge
pip install "lm_eval[hf]"                   # transformers — required to import lm_eval.api.metrics (squad_metrics, etc.)
```

## YAML examples (`generate_until`)

**Levenshtein (normalized similarity)**

```yaml
metric_list:
  - metric: levenshtein
    aggregation: mean
    higher_is_better: true
    normalized: true
```

**Embedding cosine**

```yaml
metric_list:
  - metric: embedding_cosine
    aggregation: mean
    api_base: "https://api.openai.com/v1"
    model: "text-embedding-3-small"
    use_cache: true
```

**LLM judge**

```yaml
metric_list:
  - metric: llm_judge
    aggregation: mean
    api_base: "https://api.openai.com/v1"
    model: "gpt-4o-mini"
    judge_prompt: |
      Compare the prediction to the reference.
      Reference: {reference}
      Prediction: {prediction}
      Reply with one number from 1 to 10.
    score_regex: null
```

**METEOR / BERTScore / COMET / BLEURT**

```yaml
metric_list:
  - metric: meteor
    aggregation: mean
  - metric: bertscore
    aggregation: mean
    lang: en
  - metric: comet
    aggregation: mean
    model_name: "Unbabel/wmt22-comet-da"
  - metric: bleurt
    aggregation: mean
```

## Running tests

From the `lm-evaluation-harness` directory, with dev dependencies installed:

```bash
./scripts/bootstrap_dev_venv.sh
source .venv/bin/activate
python -m pytest tests/test_generative_metrics.py -v
python -m pytest tests/test_metrics.py -v
```

- Tests that need **NLTK WordNet** may `skip` if data are missing (offline CI).
- **`test_bertscore_smoke`** loads BERTScore + torch and may be slow on first run.
- HTTP-based metrics are **mocked** and do not require a live API.

## Files touched

- [`lm_eval/api/metrics.py`](../lm_eval/api/metrics.py) — `compute_f1_gen` fix; import of `metrics_generative`.
- [`lm_eval/api/metrics_generative.py`](../lm_eval/api/metrics_generative.py) — new metrics.
- [`pyproject.toml`](../pyproject.toml) — extra `generative_metrics`.
- [`scripts/bootstrap_dev_venv.sh`](../scripts/bootstrap_dev_venv.sh) — local venv bootstrap.
- [`tests/test_generative_metrics.py`](../tests/test_generative_metrics.py) — tests.
- This doc — [`docs/GENERATIVE_METRICS.md`](GENERATIVE_METRICS.md).
