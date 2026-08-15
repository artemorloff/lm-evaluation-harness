"""One lm-eval backend for OpenRouter: synchronous and batch, no side-car proxy.

Registers three names, all the same class:

    openrouter          synchronous, mode chosen by `mode=` (default sync)
    openrouter-chat     synchronous
    openrouter-batch    asynchronous Batch API (half price where offered)

Nothing in lm_eval is modified. `get_model` resolves names through the registry
that `@register_model` fills, and `import lm_eval.models` below first registers
every stock backend as a lazy placeholder, so importing this file cannot hide
one.

What it folds in, and why each piece exists
-------------------------------------------
* **reasoning** — `local-chat-completions` has no way to send OpenRouter's
  `reasoning` object. Read here from `model_args` or from `--gen_kwargs`.
* **corporate proxy** — `requests` reads HTTP_PROXY from the environment, but
  aiohttp does NOT unless its session is built with `trust_env=True`, and
  `TemplateAPI` does not build it that way. So the concurrent path would ignore
  the proxy entirely and get a Cloudflare 403 instead of the API. `amodel_call`
  is overridden here purely to pass `proxy=` through. Give it as `proxy=` in
  model_args; nothing needs exporting beforehand.
* **empty answers** — a 200 response can carry no text at all, three ways:
  finish_reason `error` (provider gave up), `length` (thinking ate the whole
  budget) or `stop` (finished, said nothing). lm-eval accepts all three as
  answers and scores them zero. Retried here, handing budget back from
  reasoning to the answer.
* **usage journal** — OpenRouter reports the real cost per response and lm-eval
  keeps only the text. Written to `usage.jsonl` next to the output, which is
  what the spend tracking reads.

Every one of those was previously done by a separate HTTP server that had to be
started on its own port, whose log file was truncated whenever two runs shared
an output dir, and which outlived its parent when a run was killed.

Usage — the file doubles as a launcher, so no lm_eval file needs an entry:

    python scripts_gigachat/openrouter.py \\
        --model openrouter --model_args \\
        model=openai/gpt-5-nano,proxy=http://127.0.0.1:12334,reasoning_effort=high \\
        --tasks newreasoning --include_path ./benchmark_tasks \\
        --apply_chat_template --fewshot_as_multiturn --log_samples --predict_only \\
        --output_path ./output/demo --limit 2
"""

import copy
import json
import logging
import os
import sys
import time
from typing import Any, Dict, List, Optional

# `lm_eval` may resolve to an unrelated editable install elsewhere on the
# machine. Point at the harness this file lives in, but only if nothing has
# imported lm_eval yet, so a caller that arranged its own sys.path keeps it.
if "lm_eval" not in sys.modules:
    _HARNESS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _HARNESS not in sys.path:
        sys.path.insert(0, _HARNESS)

import lm_eval.models  # noqa: F401  (registers every stock backend first)
import requests as _requests
from lm_eval.api.registry import register_model
from lm_eval.models.api_models import JsonChatStr
from lm_eval.models.openai_completions import LocalChatCompletion
from tqdm import tqdm

eval_logger = logging.getLogger(__name__)

CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"
BATCH_URL = "https://openrouter.ai/api/beta/batches"
TERMINAL = {"completed", "failed", "cancelled", "expired"}


def _coerce(value: Any) -> Any:
    """model_args values arrive as strings; parse JSON-looking ones."""
    if not isinstance(value, str):
        return value
    text = value.strip()
    if text.startswith(("{", "[")):
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return value
    return value


def _answer_of(body: Any) -> Optional[str]:
    """Assistant text from one chat-completions body, or None if malformed."""
    if not isinstance(body, dict):
        return None
    choices = body.get("choices")
    if not choices:
        return None
    message = (choices[0] or {}).get("message") or {}
    return message.get("content")


def _finish_of(body: Any) -> Optional[str]:
    if not isinstance(body, dict):
        return None
    choices = body.get("choices") or []
    return (choices[0] or {}).get("finish_reason") if choices else None


@register_model("openrouter", "openrouter-chat", "openrouter-batch")
class OpenRouterLM(LocalChatCompletion):
    """OpenRouter through lm-eval, synchronously or as one Batch API job."""

    #: Attempts on an empty answer before it is recorded as empty (sync only).
    EMPTY_RETRIES = 3
    #: Submission answers 202 and the batch becomes queryable a moment later, so
    #: an immediate poll can 404 on a healthy batch. Observed on the first run.
    NOT_FOUND_GRACE_SECONDS = 180.0

    def __init__(
        self,
        base_url: Optional[str] = None,
        mode: Optional[str] = None,
        proxy: Optional[str] = None,
        reasoning: Any = None,
        reasoning_effort: Optional[str] = None,
        reasoning_max_tokens: Optional[int] = None,
        reasoning_exclude: bool = False,
        api_key: Optional[str] = None,
        usage_log: Optional[str] = None,
        poll_seconds: float = 15.0,
        max_requests_per_batch: int = 500,
        max_wait_seconds: float = 86_400.0,
        preflight: Any = True,
        **kwargs,
    ) -> None:
        # The Batch API is selected by the model name carrying OpenRouter's
        # `:batch` suffix — `anthropic/claude-opus-5:batch`. That is the same
        # string the catalogue prices, so naming the model is also how you say
        # which price you expect; an explicit `mode=` still wins if given.
        # Whether a model HAS a batch endpoint, and whether it is actually
        # cheaper, is the caller's business: OpenRouter refuses an unsupported
        # one at submission with a plain 400 and bills nothing.
        named_batch = str(kwargs.get("model") or "").endswith(":batch")
        if mode is None:
            self.mode = "batch" if named_batch else "sync"
        else:
            self.mode = "batch" if str(mode).lower() == "batch" else "sync"

        # Cheap guards against a typo, checked before anything is sent. Neither
        # combination is fatal, but both are almost always a mistake, and the
        # batch one is the expensive kind: the run would go away for hours and
        # come back billed at the ordinary price.
        if self.mode == "batch" and not named_batch:
            eval_logger.warning(
                "openrouter: mode=batch but the model name has no ':batch' "
                "suffix — OpenRouter prices the suffixed name, so this run may "
                "be billed at the ordinary rate. Did you mean "
                f"'{kwargs.get('model')}:batch'?"
            )
        if self.mode != "batch" and named_batch:
            eval_logger.warning(
                "openrouter: the model name ends with ':batch' but this run is "
                "synchronous — the batch endpoint is not being used. Drop the "
                "suffix, or remove mode=sync to let the name choose."
            )
        kwargs.setdefault("tokenized_requests", False)
        kwargs.setdefault("tokenizer_backend", None)
        if self.mode == "batch":
            # One submission carries the whole task; concurrency is the API's
            # problem, not ours.
            kwargs["num_concurrent"] = 1
        super().__init__(base_url=base_url or CHAT_URL, **kwargs)

        self.proxy = proxy or os.environ.get("OPENROUTER_PROXY") or None
        self.batch_root = BATCH_URL
        self.poll_seconds = float(poll_seconds)
        self.max_requests_per_batch = int(max_requests_per_batch)
        self.max_wait_seconds = float(max_wait_seconds)
        self.waited_seconds = 0.0

        # `simple_parse_args_string` splits model_args on every comma and never
        # builds dicts, so a multi-key JSON object cannot survive on the command
        # line at all. Hence the comma-free spellings below; `reasoning=` still
        # accepts a full object when it has a single key, and OPENROUTER_REASONING
        # takes any object at all.
        # `exclude: true` asks the provider not to return the thinking at all.
        # Default false, which is what every measurement so far used: the
        # thinking comes back in `message.reasoning`, where lm-eval ignores it,
        # and that is precisely what keeps `content` a clean answer. Worth
        # turning on only where a provider rejects `exclude: false` or bills for
        # returning the trace — it does NOT reduce the thinking, only its
        # delivery, so it changes cost far less than it looks.
        self.reasoning_exclude = str(reasoning_exclude).lower() in ("1", "true", "yes")
        if reasoning is not None:
            self.reasoning = _coerce(reasoning)
        elif reasoning_max_tokens:
            # An explicit thinking budget. OpenRouter documents `max_tokens`
            # inside `reasoning` for Anthropic and Gemini; elsewhere `effort` is
            # the lever, and this may be ignored.
            self.reasoning = {
                "max_tokens": int(reasoning_max_tokens),
                "exclude": self.reasoning_exclude,
            }
        elif reasoning_effort:
            self.reasoning = {
                "effort": str(reasoning_effort),
                "exclude": self.reasoning_exclude,
            }
        else:
            self.reasoning = _coerce(os.environ.get("OPENROUTER_REASONING")) or None

        self._api_key = (
            api_key
            or os.environ.get("OPENROUTER_API_KEY")
            or self._key_from_repo()
        )
        if not self._api_key:
            raise ValueError(
                "no OpenRouter key: pass api_key= in --model_args or set "
                "OPENROUTER_API_KEY"
            )
        self.usage_log = usage_log
        eval_logger.info(
            f"openrouter: mode={self.mode} proxy={self.proxy or 'none'} "
            f"reasoning={self.reasoning}"
        )
        if str(preflight).lower() not in ("0", "false", "no"):
            self._preflight()

    def _preflight(self) -> None:
        """Fail fast, before any billable work, on the mistakes that cost time.

        The model catalogue is free to fetch, and fetching it exercises exactly
        the things that otherwise fail late and obscurely: the corporate proxy,
        the API key, and whether this model id exists at all. A batch run that
        gets any of these wrong would otherwise be discovered after hours of
        waiting, or — worse — after being billed at the ordinary price.

        Never fatal on a network hiccup: a preflight that cannot reach the
        catalogue must not block a run that would have worked.
        """
        try:
            response = _requests.get(
                "https://openrouter.ai/api/v1/models",
                headers=self.header,
                proxies=self._proxies,
                timeout=60,
            )
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                f"openrouter: cannot reach the model catalogue via "
                f"proxy={self.proxy or 'none'} — {type(exc).__name__}: {exc}. "
                f"Check the `proxy=` argument; without it requests are answered "
                f"by Cloudflare, not the API."
            ) from exc
        if response.status_code == 401:
            raise RuntimeError(
                "openrouter: the API key was rejected (401). Set "
                "OPENROUTER_API_KEY or pass api_key= in --model_args."
            )
        if response.status_code >= 400:
            eval_logger.warning(
                f"openrouter: preflight skipped, catalogue answered HTTP "
                f"{response.status_code}"
            )
            return
        try:
            catalogue = {m["id"]: m for m in response.json().get("data") or []}
        except ValueError:
            eval_logger.warning("openrouter: preflight skipped, catalogue unreadable")
            return

        if self.model not in catalogue:
            plain = self.model.removesuffix(":batch")
            hint = ""
            if self.mode == "batch" and plain in catalogue:
                hint = (
                    f" The model exists but has no batch endpoint; "
                    f"'{plain}' is available synchronously."
                )
            raise ValueError(
                f"openrouter: '{self.model}' is not in the OpenRouter catalogue."
                + hint
            )

        pricing = catalogue[self.model].get("pricing") or {}
        out = float(pricing.get("completion") or 0) * 1e6
        inp = float(pricing.get("prompt") or 0) * 1e6
        note = f"openrouter: {self.model} — ${inp:.2f}/${out:.2f} per 1M in/out"
        # A ':batch' variant is not always a discount. z-ai/glm-5.2:batch is
        # currently 52% MORE expensive than the plain model, because the plain
        # price was cut and the batch one was not. Say so rather than let the
        # run assume it is saving money.
        plain = self.model.removesuffix(":batch")
        if self.mode == "batch" and plain in catalogue and plain != self.model:
            plain_out = float(
                (catalogue[plain].get("pricing") or {}).get("completion") or 0
            ) * 1e6
            if plain_out > 0:
                delta = 100 * (1 - out / plain_out)
                note += f", {delta:+.0f}% vs the sync price"
                if delta < 0:
                    eval_logger.warning(
                        f"openrouter: the batch variant of {plain} is "
                        f"{-delta:.0f}% MORE expensive than the ordinary one"
                    )
        eval_logger.info(note)

    # --- plumbing ------------------------------------------------------------
    @staticmethod
    def _key_from_repo() -> str:
        """Last resort: the repo's api_keys.env, three levels up from here."""
        here = os.path.dirname(os.path.abspath(__file__))
        candidate = os.path.join(os.path.dirname(os.path.dirname(here)), "api_keys.env")
        try:
            with open(candidate, encoding="utf-8") as handle:
                for line in handle:
                    if line.strip().startswith("OPENROUTER_API_KEY="):
                        return line.strip().split("=", 1)[1].strip()
        except OSError:
            pass
        return ""

    @property
    def header(self) -> dict:  # type: ignore[override]
        return {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

    @property
    def _proxies(self) -> Optional[Dict[str, str]]:
        return {"http": self.proxy, "https": self.proxy} if self.proxy else None

    def _record(
        self, body: Any, payload: Dict[str, Any], cost: Optional[float] = None
    ) -> None:
        if not self.usage_log or not isinstance(body, dict):
            return
        usage = body.get("usage") or {}
        text = _answer_of(body) or ""
        row = {
            "model": body.get("model") or payload.get("model"),
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "reasoning_tokens": (usage.get("completion_tokens_details") or {}).get(
                "reasoning_tokens"
            ),
            "cost": usage.get("cost") if cost is None else cost,
            "finish_reason": _finish_of(body),
            "content_chars": len(text),
            "max_tokens": payload.get("max_tokens"),
            "reasoning_arg": payload.get("reasoning"),
        }
        try:
            os.makedirs(os.path.dirname(self.usage_log) or ".", exist_ok=True)
            with open(self.usage_log, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        except OSError:
            pass

    def _tighten(self, payload: Dict[str, Any], attempt: int) -> Dict[str, Any]:
        """Hand budget back from reasoning to the answer, harder each attempt.

        Never `{"enabled": False}`: some endpoints refuse that outright —
        google/gemini-3.7-flash answers 400 "Reasoning is mandatory for this
        endpoint and cannot be disabled" — and a 400 aborts the whole task.
        `effort: low` reaches the same goal and is accepted everywhere.
        """
        body = dict(payload)
        budget = int(body.get("max_tokens") or 32768)
        # Carry the run's own `exclude` into the retry: a provider that rejects
        # exclude:false would reject it here too, turning the rescue attempt
        # into the thing that fails.
        exclude = getattr(self, "reasoning_exclude", False)
        if attempt == 1:
            body["reasoning"] = {
                "max_tokens": max(budget // 4, 1024),
                "exclude": exclude,
            }
        else:
            body["reasoning"] = {"effort": "low", "exclude": exclude}
        return body

    # --- payload -------------------------------------------------------------
    def _create_payload(self, messages, generate=False, gen_kwargs=None, **kwargs):
        gen_kwargs = dict(gen_kwargs or {})
        # `reasoning` may also arrive per-task through --gen_kwargs.
        inline = gen_kwargs.pop("reasoning", None)
        payload = super()._create_payload(
            messages, generate=generate, gen_kwargs=gen_kwargs, **kwargs
        )
        chosen = _coerce(inline) if inline is not None else self.reasoning
        if chosen:
            payload["reasoning"] = chosen
        return payload

    # --- synchronous ---------------------------------------------------------
    def model_call(self, messages, *, generate=True, gen_kwargs=None, **kwargs):
        gen_kwargs = copy.deepcopy(gen_kwargs)
        payload = self._create_payload(
            self.create_message(messages),
            generate=generate,
            gen_kwargs=gen_kwargs,
            seed=self._seed,
            eos=self.eos_string,
            **kwargs,
        )
        attempt_payload = payload
        for attempt in range(1, self.EMPTY_RETRIES + 1):
            response = _requests.post(
                self.base_url,
                json=attempt_payload,
                headers=self.header,
                proxies=self._proxies,
                timeout=self.timeout,
            )
            if response.status_code >= 400 and attempt_payload is not payload:
                # A 4xx caused by OUR OWN edit must never reach lm-eval: it
                # aborts the task, costing far more than the blank it was fixing.
                eval_logger.warning(
                    f"openrouter: HTTP {response.status_code} on a tightened "
                    f"payload; retrying with the original"
                )
                attempt_payload = payload
                response = _requests.post(
                    self.base_url,
                    json=payload,
                    headers=self.header,
                    proxies=self._proxies,
                    timeout=self.timeout,
                )
            response.raise_for_status()
            body = response.json()
            self._record(body, attempt_payload)
            text = _answer_of(body)
            if (text or "").strip() or attempt == self.EMPTY_RETRIES:
                return body
            why = _finish_of(body)
            eval_logger.warning(
                f"openrouter: empty content (finish_reason={why}), retry "
                f"{attempt}/{self.EMPTY_RETRIES - 1}"
            )
            attempt_payload = (
                payload if why == "error" else self._tighten(payload, attempt)
            )
            time.sleep(2 * attempt)
        return body

    async def amodel_call(
        self,
        session,
        sem,
        messages,
        *,
        generate=True,
        cache_keys=None,
        ctxlens=None,
        gen_kwargs=None,
        **kwargs,
    ):
        """Concurrent path. Overridden ONLY to route through `proxy=`.

        aiohttp ignores HTTP_PROXY unless its session was built with
        trust_env=True, and TemplateAPI does not build it that way — so without
        this the concurrent path would leave the corporate proxy unused and get
        a Cloudflare 403 instead of the API.
        """
        import asyncio

        gen_kwargs = copy.deepcopy(gen_kwargs)
        payload = self._create_payload(
            self.create_message(messages),
            generate=generate,
            gen_kwargs=gen_kwargs,
            seed=self._seed,
            eos=self.eos_string,
            **kwargs,
        )
        cache_method = "generate_until" if generate else "loglikelihood"
        await sem.acquire()
        attempt_payload = payload
        try:
            for attempt in range(1, self.EMPTY_RETRIES + 1):
                async with session.post(
                    self.base_url,
                    json=attempt_payload,
                    headers=self.header,
                    proxy=self.proxy,
                ) as response:
                    if not response.ok:
                        error_text = await response.text()
                        eval_logger.warning(
                            f"openrouter: HTTP {response.status}: {error_text[:300]}"
                        )
                        if attempt_payload is not payload:
                            attempt_payload = payload
                            continue
                    response.raise_for_status()
                    body = await response.json()
                self._record(body, attempt_payload)
                text = _answer_of(body)
                if (text or "").strip() or attempt == self.EMPTY_RETRIES:
                    break
                why = _finish_of(body)
                eval_logger.warning(
                    f"openrouter: empty content (finish_reason={why}), retry "
                    f"{attempt}/{self.EMPTY_RETRIES - 1}"
                )
                attempt_payload = (
                    payload if why == "error" else self._tighten(payload, attempt)
                )
                await asyncio.sleep(2 * attempt)

            answers = self.parse_generations(outputs=body)
            if cache_keys:
                for res, cache in zip(answers, cache_keys):
                    self.cache_hook.add_partial(cache_method, cache, res)
            return answers
        finally:
            sem.release()

    # --- batch ---------------------------------------------------------------
    def _submit(self, entries: List[Dict[str, Any]]) -> str:
        response = _requests.post(
            self.batch_root,
            headers=self.header,
            proxies=self._proxies,
            json={
                "endpoint": "/v1/chat/completions",
                "model": self.model,
                "requests": entries,
            },
            timeout=600,
        )
        if response.status_code >= 400:
            raise RuntimeError(
                f"batch submit failed: HTTP {response.status_code} "
                f"{response.text[:400]}"
            )
        batch_id = response.json().get("id")
        if not batch_id:
            raise RuntimeError("batch submit returned no id")
        return batch_id

    def _await(self, batch_id: str) -> Dict[str, Any]:
        started = time.time()
        url = f"{self.batch_root}/{batch_id}"
        while True:
            time.sleep(self.poll_seconds)
            response = _requests.get(
                url, headers=self.header, proxies=self._proxies, timeout=300
            )
            if response.status_code == 404:
                if time.time() - started < self.NOT_FOUND_GRACE_SECONDS:
                    continue
                raise RuntimeError(f"batch {batch_id} never became visible")
            if response.status_code >= 400:
                raise RuntimeError(
                    f"batch poll failed: HTTP {response.status_code} "
                    f"{response.text[:300]}"
                )
            body = response.json()
            if body.get("status") in TERMINAL:
                self.waited_seconds += time.time() - started
                eval_logger.info(
                    f"batch {batch_id}: {body.get('status')} after "
                    f"{time.time() - started:.0f}s"
                )
                return body
            if time.time() - started > self.max_wait_seconds:
                raise TimeoutError(f"batch {batch_id} exceeded max_wait_seconds")

    def generate_until(self, requests_, disable_tqdm: bool = False) -> List[str]:
        if self.mode != "batch":
            return super().generate_until(requests_, disable_tqdm)
        if not requests_:
            return []

        contexts, all_gen_kwargs = zip(*(req.args for req in requests_))
        entries = []
        for index, (context, gen_kwargs) in enumerate(zip(contexts, all_gen_kwargs)):
            messages = (
                json.loads(context.prompt)
                if isinstance(context, JsonChatStr)
                else context
            )
            entries.append(
                {
                    "custom_id": f"req-{index}",
                    "body": self._create_payload(
                        messages,
                        generate=True,
                        gen_kwargs=dict(gen_kwargs or {}),
                        seed=self._seed,
                        eos=self.eos_string,
                    ),
                }
            )

        by_id: Dict[str, str] = {}
        progress = tqdm(total=len(entries), desc="Batch API", disable=disable_tqdm)
        self._run_entries(entries, by_id, progress)

        # A batch answers once, so the per-request retry the synchronous path
        # does is impossible here — the only way to ask again is another batch.
        # Worth it: an empty answer scores zero, and the commonest cause is
        # reasoning eating the whole budget (finish_reason "length"), which a
        # tightened reasoning setting fixes. Measured on gpt-5-nano at
        # max_gen_toks=8192 with effort=high: 2 of 2 came back empty.
        for attempt in range(1, self.EMPTY_RETRIES):
            blanks = [e for e in entries if not (by_id.get(e["custom_id"]) or "").strip()]
            if not blanks:
                break
            eval_logger.warning(
                f"batch: {len(blanks)} empty answers, resubmitting round "
                f"{attempt}/{self.EMPTY_RETRIES - 1} with reduced reasoning"
            )
            retry_entries = [
                {"custom_id": e["custom_id"], "body": self._tighten(e["body"], attempt)}
                for e in blanks
            ]
            self._run_entries(retry_entries, by_id, None)
        progress.close()
        return self._collect(contexts, all_gen_kwargs, by_id, len(entries))

    def _run_entries(self, entries, by_id, progress) -> None:
        """Submit entries in chunks and fold their answers into `by_id`."""
        for start in range(0, len(entries), self.max_requests_per_batch):
            chunk = entries[start : start + self.max_requests_per_batch]
            batch_id = self._submit(chunk)
            eval_logger.info(f"batch {batch_id}: submitted {len(chunk)} requests")
            body = self._await(batch_id)
            results = body.get("results") or []
            # `cost` is reported ONLY for the batch as a whole — the per-result
            # bodies carry token counts and no price at all. Recording them as
            # they come would write a zero into every row and make a batch run
            # look free. Split the batch price across results by total tokens.
            batch_cost = (body.get("usage") or {}).get("cost")
            totals = []
            for result in results:
                inner_usage = (
                    ((result.get("response") or {}).get("body") or {}).get("usage") or {}
                )
                totals.append(inner_usage.get("total_tokens") or 0)
            grand = sum(totals) or 1
            sent = {e["custom_id"]: e["body"] for e in chunk}
            for result, tokens in zip(results, totals):
                inner = (result.get("response") or {}).get("body")
                custom_id = result.get("custom_id")
                if custom_id:
                    by_id[custom_id] = _answer_of(inner) or ""
                    share = (
                        batch_cost * tokens / grand if batch_cost is not None else None
                    )
                    # Journal the body that was actually SENT, not a stub: the
                    # `reasoning_arg` column is what proves a run used one
                    # reasoning setting throughout, and a stub wrote None into
                    # every batch row, making that check silently vacuous.
                    self._record(
                        inner, sent.get(custom_id) or {"model": self.model}, cost=share
                    )
            if progress is not None:
                progress.update(len(chunk))

    def _collect(self, contexts, all_gen_kwargs, by_id, total: int) -> List[str]:
        # By custom_id, never by position: the API does not promise order, and
        # lm-eval consumes this list positionally. Getting it wrong would attach
        # every answer to the wrong document while all format checks still pass.
        out, missing = [], 0
        for index, (context, gen_kwargs) in enumerate(zip(contexts, all_gen_kwargs)):
            text = by_id.get(f"req-{index}")
            if text is None:
                missing += 1
                text = ""
            out.append(text)
            if text:
                self.cache_hook.add_partial(
                    "generate_until", (context, gen_kwargs), text
                )
        if missing:
            eval_logger.warning(f"{missing} requests returned no result")
        eval_logger.info(
            f"batch total wait: {self.waited_seconds:.0f}s for {total} requests"
        )
        return out


def main() -> None:
    from lm_eval.__main__ import cli_evaluate

    cli_evaluate()


if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    main()
