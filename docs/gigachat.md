# GigaChat backend (`gigachat-completion`)

Two backends talk to GigaChat and they are not interchangeable:

| model name | class | transport |
|---|---|---|
| `gigachat-completion` | `GigaChatLM` | the `gigachat` SDK |
| `gigachat-chat` | `GigaChatAPI` | OpenAI-compatible `/chat/completions` |

Everything below is about `gigachat-completion`, which is what the MERA text-2.0
runs used. `gigachat-chat` inherits `LocalChatCompletion` and shares none of the
retry or streaming behaviour described here.

## Quick start

```bash
lm_eval --model gigachat-completion \
  --model_args model=GigaChat-3-Pro,base_url=https://gigachat.../v1,profanity_check=false,tools=[],preset=default,timeout=1800,stream=true \
  --tasks rubin --include_path ./benchmark_tasks \
  --gen_kwargs 'until=<|im_end|>,do_sample=False,temperature=0,max_gen_toks=32768' \
  --apply_chat_template --fewshot_as_multiturn --log_samples --predict_only
```

The gateway is internal, so a corporate HTTP proxy usually cannot reach it. Set
`no_proxy` for the gateway host rather than unsetting the proxy globally.

Authentication reads `GIGACHAT_TOKEN` and `GIGACHAT_SCOPE` from the environment.

---

## Parameters

### `--model_args` — named

| parameter | default | meaning |
|---|---|---|
| `model` | `GigaChat` | model name, e.g. `GigaChat-3-Pro` |
| `base_url` | SDK default | gateway URL |
| `scope` | `GIGACHAT_API_PERS` | `GIGACHAT_API_PERS`, `GIGACHAT_API_CORP` or `GIGACHAT_API_B2B`; `GIGACHAT_SCOPE` wins |
| `verify_ssl_certs` | `false` | certificate checking |
| `timeout` | 200 | HTTP read timeout, seconds |
| `max_tokens` | none | passed straight through; the API picks a value when unset |
| `temperature` | none | note the API refuses exactly zero |

`timeout` used to be hard-coded at 200 s. A task that generates tens of
thousands of characters needs far more — `sobhard` asks for
`max_gen_toks=65536` — and a read timeout costs the full wait *and* a back-off
before the retry. Set it to something like 1800 for long-generation tasks.

### `--model_args` — everything else

Any other key becomes part of the request body. Values arrive from the command
line as strings and are converted first: `false` → `False`, `12` → `12`,
`0.9` → `0.9`, `[]` → `[]`, `{...}` → a dict.

Where a key lands then depends on the installed SDK:

* if the SDK's `Chat` model **declares** it, it stays a top-level field —
  `profanity_check`, `top_p`, `repetition_penalty`, `n`, `response_format`,
  `stream`, `reasoning_effort`, `update_interval`, `storage`, `flags`,
  `functions`, `function_call`, `function_ranker`;
* if it does **not**, it is routed through `additional_fields`, which the SDK
  merges back into the top level of the request body.

That second path is the whole point. `Chat` in gigachat 0.2.3 has no `tools`
field and no `model_options` field, so passing them directly meant pydantic
dropped them silently — they never reached the API, and nothing in the logs
said so. Two conveniences build on it:

| you write | what is sent |
|---|---|
| `profanity_check=false` | `profanity_check: false` (declared field, top level) |
| `tools=[]` | `tools: []` (via `additional_fields`) |
| `preset=default` | `model_options: {"preset": "default"}` (via `additional_fields`) |

`stream` is the exception to the routing rule: it is a declared field, but it is
a client-side switch here and is removed from the payload before the request is
built, because the SDK's `stream()` sets it on the request itself. Leaving it in
would send it twice.

### `--gen_kwargs`

`until` is applied client-side, by cutting the generation at the first marker.
`max_gen_toks` becomes the `max_tokens` of the request.

---

## Streaming, and the 300-second wall

The gateway ends a response at about 300 seconds and answers 502. Measured
repeatedly at 288–301 s, with the client timeout set to 1800 s — so the limit is
the gateway's and waiting longer cannot help.

`stream=true` keeps the connection producing chunks, so the read timeout never
fires and long generations complete: requests of 948 s have been measured this
way. It is the only lever that reaches those documents; without it about 5% of
`sobhard` is lost.

Streaming changes what a failure looks like, in two ways that both had to be
handled:

* the gateway reports a mid-generation error as a JSON object **inside** the SSE
  stream — `{"status_code": 500, "message": "Internal Server Error"}` — and the
  SDK feeds that to `ChatCompletionChunk`, which rejects it. The exception is a
  pydantic `ValidationError`, not a `ResponseError`. Left unhandled it escapes
  the retry wrapper and `generate_until` abandons the whole task:
  GigaChat-3-Lightning lost all 825 `sobhard` documents to one such chunk.
* a mid-body disconnect arrives as an httpx protocol error carrying no status.
  It is transient and worth retrying — which is why elapsed time alone is not
  enough to recognise the wall (see below).

---

## Retry policy

Retried: `httpx.ReadTimeout`, `httpx.ConnectTimeout`, `httpx.RemoteProtocolError`,
`gigachat.exceptions.ResponseError`, `pydantic.ValidationError`.

Four rules, each from a measurement rather than an assumption.

**The back-off is capped at 60 s.** The shared helper multiplies its sleep by
1.5 forever, which walks to 3 → 4.5 → … → 173 → 259 s. Under contention the
gateway answers 429 steadily, and by the twelfth refusal a worker would sleep
four minutes for a limit that resets in seconds.

**429 does not consume an attempt.** It means "not now", not "this request is
bad", and waiting it out is the entire remedy. Bursts of 16–22 in a row have
been measured with several workers on one token; counting them would throw away
perfectly good documents.

**401 and 403 abort the run, loudly.** They are not the model failing, they are
us not being allowed to ask, so retrying cannot help — and giving up would write
an empty answer that scores exactly like a wrong one. An expired token silently
turned 241 `sobhard` documents on GigaChat-3-Ultra into blanks while the run
looked healthy throughout.

**The wall is given up on immediately — but only when it really is the wall.**
That means a 502 *and* an elapsed time past 280 s, both. Elapsed time alone used
to be enough, because without streaming nothing else could run that long and
still fail; with streaming a request legitimately lives far longer, and a
mid-body disconnect classified as the wall would give up instantly and write a
blank. Measured at 1 of the first 5 streamed `sobhard` documents, which over 703
of them is roughly 140 blanks no retry policy would ever revisit.

After four attempts the document is recorded as an empty answer rather than
raising. This matters more than it looks: `generate_until` turns a raised
`ResponseError` into a `break`, which abandons every remaining document in the
task. One bad document should cost one document.

---

## Reading a failure

`parse_exception` returns `(status, message)` from a `ResponseError`. Read the
attributes, never `args`: `ResponseError.__init__` keeps url, status_code,
content and headers on the instance but calls
`super().__init__(f"{status_code} {url}")`, so `args` holds exactly one
formatted string. Indexing into it for the body raises
`IndexError: tuple index out of range` for every error it is handed — which
turned a deliberate, correct 401 abort into an unexplained traceback with the
actual cause (an expired token) appearing nowhere. That cost two 20-hour
`sobhard` runs their diagnosis.

---

## Empty answers

An empty answer is scored as a wrong one, in silence. The three sources seen so
far, and what each looks like:

| cause | signature | remedy |
|---|---|---|
| gateway wall at ~300 s | 502 after 288–301 s | `stream=true` |
| expired or wrong-scope token | 401/403 | the run aborts; fix the token |
| SSE-embedded 500 | pydantic `ValidationError` | retried automatically |

Worth auditing a finished run for blanks rather than trusting that it went well:
in every case above the run reported success while the answers were missing.
