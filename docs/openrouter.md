# OpenRouter backend (`openrouter`)

One lm-eval model backend for [OpenRouter](https://openrouter.ai), covering all
three ways to run a measurement:

| mode | what it does | when to use it |
|---|---|---|
| **synchronous** | one request at a time, answer immediately | debugging, small tasks, reproducible pacing |
| **asynchronous** | N requests in flight at once, answers immediately | the normal way to run a full benchmark |
| **batch** | the whole task submitted as one job, answers later | expensive models — usually half price |

It also folds in four things the stock `local-chat-completions` backend cannot
do, each of which previously required running a separate proxy server:

* sending OpenRouter's `reasoning` object,
* routing through a corporate HTTP proxy **on the concurrent path** (aiohttp
  ignores `HTTP_PROXY` unless its session is built with `trust_env`, which
  lm-eval does not do),
* retrying answers that come back empty,
* journalling what each request actually cost.

Registered names: `openrouter`, `openrouter-chat`, `openrouter-batch` — all the
same class, so pick whichever reads best.

---

## Quick start

```bash
export OPENROUTER_API_KEY=sk-or-v1-...

lm_eval --model openrouter \
  --model_args model=openai/gpt-5-nano,num_concurrent=16,reasoning_effort=high \
  --tasks newreasoning \
  --apply_chat_template --fewshot_as_multiturn \
  --log_samples --predict_only \
  --output_path ./output/demo
```

---

## Parameters

Two groups, and the split matters: **`--model_args` is "where and how we
connect", `--gen_kwargs` is "what we ask the model to generate"**. A parameter
in the wrong group is silently ignored.

### `--model_args` — connection and mode

| parameter | default | meaning |
|---|---|---|
| `model` | — | OpenRouter model id. A `:batch` suffix selects the Batch API (see below). |
| `mode` | from the name | `sync` or `batch`. Normally leave it out; the model name decides. |
| `num_concurrent` | 1 | requests in flight at once. `>1` is the asynchronous mode. Forced to 1 in batch mode. |
| `proxy` | `OPENROUTER_PROXY` | corporate HTTP proxy, e.g. `http://127.0.0.1:3128`. Nothing needs exporting. |
| `api_key` | `OPENROUTER_API_KEY` | falls back to the env var, then to `api_keys.env` in the repo root. |
| `usage_log` | none | path to a JSONL journal of tokens and cost per request. |
| `preflight` | `true` | check key, proxy, model name and price before doing billable work. |
| `timeout` | 1800 | HTTP read timeout, seconds. |
| `max_retries` | 5 | network-level retries (inherited from `TemplateAPI`). |
| `max_length` | 2048 | model context window. Only used for truncation checks, which are off when `tokenized_requests=False`. |

### `--model_args` — reasoning

| parameter | produces |
|---|---|
| `reasoning_effort=high\|medium\|low` | `{"effort": "high", "exclude": false}` |
| `reasoning_max_tokens=8000` | `{"max_tokens": 8000, "exclude": false}` |
| `reasoning_exclude=true` | adds `"exclude": true` to either of the above |
| `reasoning={"effort":"high"}` | the object verbatim — **single-key only**, see the comma trap |
| `OPENROUTER_REASONING` (env) | any object, however many keys |

Precedence: `reasoning` → `reasoning_max_tokens` → `reasoning_effort` → env var.

`exclude` controls whether the thinking is *returned*, not whether it happens.
The default `false` keeps it in `message.reasoning`, where lm-eval ignores it —
which is exactly what keeps `content` a clean answer. Turning it on saves
bandwidth, not money, and costs you the diagnostics: `reasoning_tokens` is how
you tell "the model thought itself out of a budget" from "the model said
nothing".

`reasoning_max_tokens` is documented by OpenRouter for Anthropic and Gemini;
elsewhere `effort` is the lever and a budget may simply be ignored.

### `--model_args` — batch only

| parameter | default | meaning |
|---|---|---|
| `poll_seconds` | 15 | how often to ask whether the job is done |
| `max_requests_per_batch` | 500 | split larger tasks across several jobs |
| `max_wait_seconds` | 86400 | give up on a job that never finishes |
| `max_batch_hold` | 50 | dollars a single job may reserve; see [the credit hold](#the-credit-hold) |

`max_batch_hold` is a ceiling on the *reservation*, not on the spend. Submitting
a job freezes the worst case against your prepaid credit, so on an expensive
model the default `max_requests_per_batch=500` asks for far more than the run
will ever cost and is refused. Raising this is only safe if the credit balance
can absorb it; lowering it makes the run slower but never more expensive.

### `--gen_kwargs` — generation

| parameter | becomes | note |
|---|---|---|
| `max_gen_toks=32768` | `max_tokens` | the **whole** answer budget: thinking *plus* text |
| `temperature=0` | `temperature` | |
| `until=<\|im_end\|>` | `stop` | see the comma trap |
| `do_sample=False` | — | dropped; greedy is expressed by `temperature=0` |
| `reasoning={...}` | `reasoning` | overrides the run-wide setting for this task |

Do not confuse the two `max_tokens`: `max_gen_toks` is the entire output budget,
while `reasoning_max_tokens` caps only the thinking inside it. The second must
be comfortably smaller than the first, or it has no effect.

---

## Two traps in lm-eval's argument parser

Both bite silently, so they are worth knowing before your first run.

**Commas.** `simple_parse_args_string` splits *both* `--model_args` and
`--gen_kwargs` on **every** comma and never builds lists or dicts. So
`reasoning={"effort":"high","exclude":false}` arrives as the string `{"effort"`
plus two junk keys. That is why `reasoning_effort=` exists, and why a
multi-element `until` list cannot be given on the command line at all. For a
full object, use the `OPENROUTER_REASONING` environment variable.

**`until` defaults to the fewshot delimiter.** A task that does not set `until`
gets `["\n\n"]` (`lm_eval/api/task.py`), which cuts every answer at its first
blank line. Pass an explicit end-of-turn marker. A single one is enough — its
job is to displace the default, and markers like `<|im_end|>` are other models'
chat tokens that a model has to emit as literal text to trigger anything.

---

## Mode 1 — synchronous

One request at a time. Slowest, but the pacing is predictable and the log reads
in order, which is what you want while debugging a task.

```bash
lm_eval --model openrouter \
  --model_args model=openai/gpt-5-nano,num_concurrent=1,proxy=http://127.0.0.1:3128,reasoning_effort=high,usage_log=./output/run/usage.jsonl \
  --tasks newreasoning \
  --gen_kwargs 'until=<|im_end|>,do_sample=False,temperature=0,max_gen_toks=32768' \
  --apply_chat_template --fewshot_as_multiturn \
  --log_samples --predict_only \
  --use_cache ./cache/gpt-5-nano/newreasoning \
  --output_path ./output/run --seed 1234
```

## Mode 2 — asynchronous (answers immediately)

The same thing with `num_concurrent` above 1: many requests are in flight at
once and each answer still arrives on its own request. This is the normal way to
run a full benchmark.

```bash
lm_eval --model openrouter \
  --model_args model=openai/gpt-5-nano,num_concurrent=16,proxy=http://127.0.0.1:3128,reasoning_effort=high,usage_log=./output/run/usage.jsonl \
  --tasks newreasoning \
  --gen_kwargs 'until=<|im_end|>,do_sample=False,temperature=0,max_gen_toks=32768' \
  --apply_chat_template --fewshot_as_multiturn \
  --log_samples --predict_only \
  --use_cache ./cache/gpt-5-nano/newreasoning \
  --output_path ./output/run --seed 1234
```

Note that `proxy=` matters most here. On the concurrent path lm-eval talks
through aiohttp, which — unlike `requests` — does not read `HTTP_PROXY` from the
environment. Without this argument the requests leave the machine directly and
come back as a Cloudflare challenge page, not as API answers.

## Mode 3 — batch (answers later)

Add `:batch` to the model name. Nothing else changes.

```bash
lm_eval --model openrouter \
  --model_args model=anthropic/claude-opus-5:batch,proxy=http://127.0.0.1:3128,reasoning_effort=high,poll_seconds=15,max_requests_per_batch=500,usage_log=./output/opus/usage.jsonl \
  --tasks sobhard \
  --gen_kwargs 'until=<|im_end|>,do_sample=False,temperature=0,max_gen_toks=65536' \
  --apply_chat_template --fewshot_as_multiturn \
  --log_samples --predict_only \
  --use_cache ./cache/claude-opus-5/sobhard \
  --output_path ./output/opus --seed 1234
```

---

## How batch works

OpenRouter's Batch API trades latency for price: the same work, usually at half
the per-token rate, delivered whenever the queue gets to it.

The backend does this per task:

1. **Build.** Every request lm-eval hands over is turned into a body by the same
   `_create_payload` the synchronous path uses, so messages, `max_tokens`,
   `temperature`, `stop` and `seed` are identical to what a sync run would have
   sent. Each gets a `custom_id`.
2. **Submit.** `POST /api/beta/batches` with `{"endpoint": "/v1/chat/completions",
   "model": ..., "requests": [...]}`, chunked at whichever is smaller:
   `max_requests_per_batch`, or the number of requests whose combined
   reservation stays under `max_batch_hold`. See [the credit hold](#the-credit-hold).
3. **Poll.** `GET /api/beta/batches/{id}` every `poll_seconds` until the status
   is `completed`, `failed`, `cancelled` or `expired`. Results come back inline.
4. **Match by `custom_id`.** Never by position — the API does not promise order,
   while lm-eval consumes the returned list positionally. Getting this wrong
   would attach every answer to the wrong document while every format check
   still passed.
5. **Retry the blanks.** Any request that came back with no text is resubmitted
   as a second, smaller batch with the thinking budget reduced. Up to two extra
   rounds.
6. **Journal.** The batch reports `cost` only for the job as a whole, so it is
   divided across the results by token count. Without that, every batch row
   would record zero and a batch run would look free.

### The credit hold

Submitting a batch **reserves the worst case**: every request is held as if it
were going to spend all of `max_tokens`, at the model's output price. The
reservation is released when the job finishes and only the real usage is
charged — but until then it has to fit.

```
hold = requests x max_tokens x output price per token
```

Two things about that hold are easy to get wrong, and both were learned the
expensive way:

**It is taken from prepaid credit, not from the key's spending limit.** A key
with a $6000 monthly limit and $149 of credit can submit jobs worth $149. The
balance is `total_credits - total_usage` from `GET /api/v1/credits`;
`GET /api/v1/auth/key` reports the limit and will not tell you why a submission
was refused.

**The refusal is an HTTP 402 that names a number you never intended to spend.**
Measured on `claude-opus-5`, an 825-document task at `max_gen_toks=65536`: 500
requests × 65536 tokens × $12.50/1M ≈ $410, refused with *"$414.61 exceeds your
available balance of $149.05"*. Once chunked, that same task completed for
$103.95 — the hold was four times the bill it was guarding against.

So the backend sizes each chunk to keep the hold under `max_batch_hold` before
submitting, and logs the arithmetic when it does:

```
batch: reserving ~$0.82 per request, so chunking at 61 to keep the hold under $50
```

The cost is wall-clock. Each chunk is a separate job with its own unpredictable
wait, so those 825 documents go out as fourteen jobs of 61 rather than two of
500, and take correspondingly longer. That is the trade: a slower run that
completes, instead of a fast one that is refused. If the account has plenty of
credit, raise `max_batch_hold` and the chunks grow back.

Note that a small `max_gen_toks` shrinks the hold as effectively as a small
chunk does — but it is not a free lever, because a tight thinking budget is
itself a source of blanks (see [Empty answers](#empty-answers)).

### What to expect

**Waiting is unpredictable and unrelated to the amount of work.** Measured on
two-request jobs: 64 s, 107 s, 163 s, 203 s, 213 s. The stated completion window
is 24 hours. A job with more work in it may well come back sooner than a small
one submitted a minute later.

**Be generous with `max_gen_toks`.** There is no per-request retry inside a
batch — the only way to ask again is another job, another wait. If the thinking
budget is tight the model can spend the whole allowance reasoning and return
nothing: measured on `gpt-5-nano` at `max_gen_toks=8192` with `effort=high`,
2 of 2 requests came back empty. The retry rescued them, but a round later.

**`:batch` is not always cheaper.** It is a different catalogue entry with its
own price. At the time of writing `z-ai/glm-5.2:batch` costs 52% *more* than the
plain model, because the ordinary price was cut and the batch one was not. The
preflight prints both and warns when the batch variant is the expensive one.

**Not every model has a batch endpoint.** Submitting one that does not is
refused immediately with a plain 400 and costs nothing, and the preflight
catches it before that.

---

## Preflight

Before any billable work the backend fetches the (free) model catalogue. That
one request exercises the three things that otherwise fail late and obscurely:
the proxy, the API key, and whether the model id exists. It also prints the
price and, in batch mode, compares it against the synchronous one.

It is never fatal on a network hiccup — a preflight that cannot reach the
catalogue must not block a run that would have worked. Disable with
`preflight=false`.

---

## The usage journal

With `usage_log=path.jsonl` every response appends one line:

```json
{"model": "openai/gpt-5-nano", "prompt_tokens": 512, "completion_tokens": 3928,
 "reasoning_tokens": 3904, "cost": 0.00167, "finish_reason": "length",
 "content_chars": 0, "max_tokens": 8192,
 "reasoning_arg": {"effort": "high", "exclude": false}}
```

lm-eval keeps only the answer text, so without this there is no way to know what
a run cost, how much of the output budget went to thinking, or whether every
request really used the same settings. `reasoning_arg` records the body that was
actually sent, which is what makes "all N requests used one reasoning setting" a
checkable claim rather than an assumption.

`finish_reason` is worth watching. An empty answer arrives as HTTP 200 in three
different disguises — `error` (the provider gave up), `length` (thinking ate the
budget) and `stop` (finished, said nothing) — and lm-eval accepts all three as
answers and scores them zero.

---

## Empty answers

An empty response is retried automatically, in both synchronous and batch modes,
by moving budget from thinking back to the answer: first an explicit reasoning
ceiling of a quarter of the budget, then `effort: low`.

Reasoning is never switched off entirely. Some endpoints refuse that outright —
`google/gemini-3.7-flash` answers `400 Reasoning is mandatory for this endpoint
and cannot be disabled` — and a 400 reaches lm-eval as a `ClientResponseError`
that aborts the whole task. A rescue attempt must not be able to destroy the run
it exists to save; for the same reason, a 4xx caused by a modified payload is
retried once with the original.

Some blanks are not the budget and no retry will move them. A model may decline
a prompt it is free to decline: `claude-opus-5` returned `content_filter` on 163
of 8679 MERA documents (1.9%), all of them ordinary API-routing questions with
nothing objectionable in them. Retrying, switching between the Anthropic,
Bedrock, Vertex and Azure providers, adding a benign system prompt and turning
reasoning off all produced the same refusal, which places it in the model rather
than in any one provider. Read `finish_reason` in the journal before spending a
day trying to rescue such answers: `length` is worth another attempt,
`content_filter` is the model's answer.
