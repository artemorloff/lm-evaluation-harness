"""Optional generative-evaluation metrics (API-based and heavy NLP metrics).

Install with: pip install lm_eval[generative_metrics] lm_eval[api] (API metrics need requests).
"""

from __future__ import annotations

import logging
import os
import re
from functools import lru_cache
from typing import Any

import numpy as np

from lm_eval.api.registry import register_metric

eval_logger = logging.getLogger(__name__)


def _require_requests():
    try:
        import requests
    except ImportError as e:
        raise ImportError(
            "embedding_cosine and llm_judge require HTTP support. "
            "Install with: pip install lm_eval[api]"
        ) from e
    return requests


def _cosine_similarity_vec(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


@lru_cache(maxsize=4096)
def _cached_embedding(
    text: str,
    api_base: str,
    model: str,
    api_key: str,
    timeout: float,
) -> tuple[float, ...]:
    """Fetch a single text embedding; cached by arguments (api_key should be explicit)."""
    requests = _require_requests()
    url = f"{api_base.rstrip('/')}/v1/embeddings"
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    payload = {"model": model, "input": text}
    resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    emb = data["data"][0]["embedding"]
    return tuple(float(x) for x in emb)


def compute_embedding_cosine(
    predictions,
    references,
    api_base: str,
    model: str,
    api_key: str | None = None,
    timeout: float = 60.0,
    use_cache: bool = True,
    **_: Any,
):
    """Cosine similarity between embeddings from an OpenAI-compatible `/v1/embeddings` API."""
    if not api_base or not model:
        raise ValueError("embedding_cosine requires `api_base` and `model` in metric_list kwargs.")
    key = api_key or os.getenv("OPENAI_API_KEY") or os.getenv("LM_EVAL_EMBEDDING_API_KEY") or ""
    pred = str(predictions[0] if predictions else "")
    ref = str(references[0] if references else "")

    def fetch_uncached(text: str) -> np.ndarray:
        requests = _require_requests()
        url = f"{api_base.rstrip('/')}/v1/embeddings"
        headers = {"Content-Type": "application/json"}
        if key:
            headers["Authorization"] = f"Bearer {key}"
        resp = requests.post(
            url,
            headers=headers,
            json={"model": model, "input": text},
            timeout=timeout,
        )
        resp.raise_for_status()
        return np.array(resp.json()["data"][0]["embedding"], dtype=np.float64)

    if use_cache:
        va = np.array(_cached_embedding(ref, api_base, model, key, timeout), dtype=np.float64)
        vb = np.array(_cached_embedding(pred, api_base, model, key, timeout), dtype=np.float64)
    else:
        va = fetch_uncached(ref)
        vb = fetch_uncached(pred)

    sim = _cosine_similarity_vec(va, vb)
    return {"embedding_cosine": sim}


def compute_levenshtein(
    predictions,
    references,
    normalized: bool = True,
    **_: Any,
):
    try:
        from rapidfuzz.distance import Levenshtein
    except ImportError as e:
        raise ImportError(
            "levenshtein metric requires rapidfuzz. Install with: pip install lm_eval[generative_metrics]"
        ) from e

    pred = predictions[0] if predictions else ""
    ref = references[0] if references else ""
    pred = str(pred)
    ref = str(ref)
    dist = Levenshtein.distance(pred, ref)
    if not normalized:
        return {"levenshtein": float(dist)}
    max_len = max(len(pred), len(ref), 1)
    sim = 1.0 - (dist / max_len)
    return {"levenshtein": float(sim)}


def compute_llm_judge(
    predictions,
    references,
    api_base: str,
    model: str,
    judge_prompt_path: str | None = None,
    instruction: str = "",
    api_key: str | None = None,
    temperature: float = 0.0,
    max_tokens: int = 256,
    timeout: float = 120.0,
    score_regex: str | None = None,
    **_: Any,
):
    if not api_base or not model:
        raise ValueError("llm_judge requires `api_base` and `model`")

    judge_prompt_path = (
        judge_prompt_path
        or os.getenv("LM_EVAL_JUDGE_PROMPT_PATH")
        or os.getenv("pollux_prompt_path")
    )

    if not judge_prompt_path:
        raise ValueError("Need judge_prompt_path")

    with open(judge_prompt_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    prompt_template = data["pollux_prompt"]["template"]
    criteria_name = data["pollux_prompt"]["criteria_name"]
    criteria_rubrics = data["pollux_prompt"]["criteria_rubrics"]

    pred = str(predictions[0] if predictions else "")
    ref = str(references[0] if references else "")

    prompt = prompt_template.format(
        instruction=instruction,
        reference_answer=ref,
        answer=pred,
        criteria_name=criteria_name,
        criteria_rubrics=criteria_rubrics,
    )

    requests = _require_requests()

    key = api_key or os.getenv("LM_EVAL_JUDGE_API_KEY") or ""
    url = f"{api_base.rstrip('/')}/v1/chat/completions"

    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = f"Bearer {key}"

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
    resp.raise_for_status()

    content = resp.json()["choices"][0]["message"]["content"]
    score_regex = r"\b([0-2])\b"
    
    m = re.search(score_regex, content, re.IGNORECASE | re.DOTALL)
    if m:
        return {"llm_judge": float(m.group(1)) / 2}
    
    raise ValueError(f"Cannot parse score: {content[:300]!r}")


def compute_meteor(predictions, references, **_: Any):
    try:
        import nltk
        from nltk.translate.meteor_score import meteor_score
        from nltk.tokenize import word_tokenize
    except ImportError as e:
        raise ImportError(
            "meteor requires nltk. Install with: pip install lm_eval[generative_metrics]"
        ) from e

    pred = predictions[0] if predictions else ""
    ref = references[0] if references else ""
    try:
        nltk.data.find("corpora/wordnet")
    except LookupError:
        eval_logger.warning(
            "NLTK WordNet not found; downloading wordnet (required for METEOR). "
            "For offline use, run: python -c \"import nltk; nltk.download('wordnet'); nltk.download('omw-1.4')\""
        )
        nltk.download("wordnet", quiet=True)
        nltk.download("omw-1.4", quiet=True)
    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        try:
            nltk.download("punkt_tab", quiet=True)
        except Exception:
            nltk.download("punkt", quiet=True)

    ref_t = word_tokenize(str(ref).lower())
    pred_t = word_tokenize(str(pred).lower())
    score = meteor_score([ref_t], pred_t)
    return {"meteor": float(score)}


def compute_bertscore(predictions, references, lang: str = "en", **kwargs: Any):
    try:
        from bert_score import score as bert_score_lib
    except ImportError as e:
        raise ImportError(
            "bertscore requires bert-score and torch. Install with: pip install lm_eval[generative_metrics]"
        ) from e

    pred = predictions[0] if predictions else ""
    ref = references[0] if references else ""
    cands = [str(pred)]
    refs = [str(ref)]
    P, R, F1 = bert_score_lib(cands, refs, lang=lang, verbose=False, **kwargs)
    return {"bertscore": float(F1.mean())}


_comet_model_cache: dict[str, Any] = {}


def _get_comet_model(checkpoint: str):
    if checkpoint in _comet_model_cache:
        return _comet_model_cache[checkpoint]
    try:
        from comet import download_model, load_from_checkpoint
    except ImportError as e:
        raise ImportError(
            "comet requires unbabel-comet. Install with: pip install lm_eval[generative_metrics]"
        ) from e

    path = download_model(checkpoint)
    model = load_from_checkpoint(path)
    _comet_model_cache[checkpoint] = model
    return model


def compute_comet(
    predictions,
    references,
    model_name: str = "Unbabel/wmt22-comet-da",
    **_: Any,
):
    pred = str(predictions[0] if predictions else "")
    ref = str(references[0] if references else "")
    model = _get_comet_model(model_name)
    # Reference-based DA: use reference as both source and reference (monolingual QA / summarization)
    data = [{"src": ref, "mt": pred, "ref": ref}]
    out = model.predict(data, batch_size=1, gpus=0)
    if hasattr(out, "scores"):
        seg_scores = out.scores
    elif isinstance(out, (list, tuple)):
        seg_scores = out
    else:
        seg_scores = [out]
    arr = np.asarray(seg_scores).astype(float).flatten()
    val = float(arr[0]) if arr.size else float("nan")
    return {"comet": val}


_bleurt_metric = None


def _get_bleurt_metric():
    global _bleurt_metric
    if _bleurt_metric is None:
        try:
            import evaluate as hf_evaluate
        except ImportError as e:
            raise ImportError("bleurt requires the evaluate package (already a core dependency).") from e
        try:
            _bleurt_metric = hf_evaluate.load("bleurt", module_type="metric")
        except Exception as e:
            raise RuntimeError(
                "Failed to load HuggingFace `bleurt` metric (may need TensorFlow or extra deps). "
                "See https://huggingface.co/metrics/bleurt"
            ) from e
    return _bleurt_metric


def compute_bleurt(predictions, references, **_: Any):
    pred = predictions[0] if predictions else ""
    ref = references[0] if references else ""
    metric = _get_bleurt_metric()
    out = metric.compute(predictions=[str(pred)], references=[str(ref)])
    if "scores" in out:
        val = float(out["scores"][0])
    else:
        first = next(iter(out.values()))
        val = float(first[0] if isinstance(first, (list, tuple)) else first)
    return {"bleurt": val}


def compute_token_overlap_f1(predictions, references, **_: Any):
    """Token-level overlap F1 (whitespace split, no extra deps)."""
    pred = str(predictions[0] if predictions else "").split()
    ref = str(references[0] if references else "").split()
    if not pred and not ref:
        return {"token_overlap_f1": 1.0}
    if not pred or not ref:
        return {"token_overlap_f1": 0.0}
    from collections import Counter

    c_pred = Counter(pred)
    c_ref = Counter(ref)
    overlap = sum((c_pred & c_ref).values())
    if overlap == 0:
        return {"token_overlap_f1": 0.0}
    precision = overlap / len(pred)
    recall = overlap / len(ref)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"token_overlap_f1": float(f1)}


def compute_sentence_bleu(predictions, references, **_: Any):
    """Corpus BLEU is `metric: bleu`; this is sentence-level sacreBLEU averaged with `mean`."""
    import sacrebleu

    pred = predictions[0] if predictions else ""
    ref = references[0] if references else ""
    sb = sacrebleu.sentence_bleu(str(pred), [str(ref)])
    return {"sentence_bleu": float(sb.score)}


# --- registrations ---


@register_metric(
    metric="embedding_cosine",
    higher_is_better=True,
    output_type="generate_until",
    aggregation="mean",
)
def embedding_cosine_fn(**kwargs):
    return compute_embedding_cosine(**kwargs)


@register_metric(
    metric="levenshtein",
    higher_is_better=True,
    output_type="generate_until",
    aggregation="mean",
)
def levenshtein_metric_fn(**kwargs):
    return compute_levenshtein(**kwargs)


@register_metric(
    metric="llm_judge",
    higher_is_better=True,
    output_type="generate_until",
    aggregation="mean",
)
def llm_judge_fn(**kwargs):
    return compute_llm_judge(**kwargs)


@register_metric(
    metric="meteor",
    higher_is_better=True,
    output_type="generate_until",
    aggregation="mean",
)
def meteor_fn(**kwargs):
    return compute_meteor(**kwargs)


@register_metric(
    metric="bertscore",
    higher_is_better=True,
    output_type="generate_until",
    aggregation="mean",
)
def bertscore_fn(**kwargs):
    return compute_bertscore(**kwargs)


@register_metric(
    metric="comet",
    higher_is_better=True,
    output_type="generate_until",
    aggregation="mean",
)
def comet_fn(**kwargs):
    return compute_comet(**kwargs)


@register_metric(
    metric="bleurt",
    higher_is_better=True,
    output_type="generate_until",
    aggregation="mean",
)
def bleurt_fn(**kwargs):
    return compute_bleurt(**kwargs)


@register_metric(
    metric="token_overlap_f1",
    higher_is_better=True,
    output_type="generate_until",
    aggregation="mean",
)
def token_overlap_f1_fn(**kwargs):
    return compute_token_overlap_f1(**kwargs)


@register_metric(
    metric="sentence_bleu",
    higher_is_better=True,
    output_type="generate_until",
    aggregation="mean",
)
def sentence_bleu_fn(**kwargs):
    return compute_sentence_bleu(**kwargs)
