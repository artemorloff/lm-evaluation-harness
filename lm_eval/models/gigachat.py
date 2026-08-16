import json
import logging
import os
import time
import warnings
from functools import wraps

import requests  # needs to be imported in order to create gigachat temp acess_token
from tqdm import tqdm

from lm_eval.api.model import LM
from lm_eval.api.registry import register_model
from lm_eval.models.openai_completions import LocalChatCompletion


logging.getLogger("httpx").setLevel(
    logging.WARNING
)  # turn off logging 200 status for each iteration

warnings.filterwarnings(
    "ignore"
)  # turn off insecure connection warning if verify_certificate=False

eval_logger = logging.getLogger(__name__)


def _coerce_model_arg(value):  # noqa: ANN001, ANN201
    """Parse model_args values from CLI strings into Python objects."""
    if not isinstance(value, str):
        return value
    stripped = value.strip()
    if stripped.lower() == "true":
        return True
    if stripped.lower() == "false":
        return False
    if stripped.isnumeric():
        return int(stripped)
    try:
        return float(stripped)
    except ValueError:
        pass
    if stripped.startswith(("{", "[", '"', "'")):
        try:
            return json.loads(stripped.replace("'", '"'))
        except json.JSONDecodeError:
            import ast

            try:
                return ast.literal_eval(stripped)
            except (SyntaxError, ValueError):
                pass
    return value


# model_args that configure the client, or that generate_until passes to Chat()
# itself. None of these belong in the payload built from **kwargs, and
# trust_remote_code is added to model_args by the harness rather than by us.
_NOT_PAYLOAD_KEYS = (
    "model",
    "base_url",
    "scope",
    "verify_ssl_certs",
    "timeout",
    "max_tokens",
    "temperature",
    "trust_remote_code",
)


#: Ceiling for the retry back-off, in seconds. `retry_on_specific_exceptions`
#: multiplies its sleep by 1.5 forever, so a run of 429s walks it up to
#: 3 -> 4.5 -> ... -> 173 -> 259 seconds. Observed live: with several workers on
#: one token the gateway answers 429 steadily, and by the twelfth refusal a
#: worker sleeps four minutes for a limit that resets in seconds. Capping is
#: local to GigaChat on purpose — the shared helper serves every other model.
_MAX_BACKOFF_SECONDS = 60.0
_INITIAL_BACKOFF_SECONDS = 3.0
_BACKOFF_MULTIPLIER = 1.5

#: A 502 that arrives this late did not fail — it ran out of time. The gateway
#: cuts a generation at about 300 seconds and answers 502; measured repeatedly
#: at 288-301s. Retrying such a request replays the same wall, so these are
#: given up on at once. A 502 that comes back quickly (12s was observed) is a
#: transient gateway fault and is worth another attempt.
#:
#: The 502 half of that test is not decoration: under `stream=true` a request
#: routinely outlives this threshold on purpose, and a mid-body disconnect
#: ("incomplete chunked read", no status attached) must still be retried.
_WALL_CLOCK_SECONDS = 280.0

#: Attempts for failures that are *not* the wall. Beyond this the document is
#: recorded as an empty answer so the task can finish; a run that hangs forever
#: on one document measures nothing at all.
_MAX_ATTEMPTS = 4


def _error_status(exc) -> int | None:
    """HTTP status carried by a gigachat ResponseError, if it carries one.

    Read the attribute, not ``args``. ``ResponseError.__init__`` keeps url,
    status_code, content and headers on the instance but calls
    ``super().__init__(f"{status_code} {url}")`` — so ``args`` holds exactly one
    formatted string, and indexing into it for the body raises. (The same
    assumption in this module's ``parse_exception`` is broken for the same
    reason.) Getting this wrong made every 429 look statusless, which spent a
    retry attempt on it and recorded empty answers for documents the gateway had
    merely asked us to send later.
    """
    status = getattr(exc, "status_code", None)
    if status is not None:
        try:
            return int(status)
        except (TypeError, ValueError):
            return None
    return None


def _retry_with_capped_backoff(
    on_exceptions,
    on_exception_callback=None,
    max_attempts=_MAX_ATTEMPTS,
    give_up_value="",
):
    """Retry transient failures; give up on the ones that hit the gateway's wall.

    Three behaviours, each earned from a measurement rather than assumed:

    * the sleep never exceeds :data:`_MAX_BACKOFF_SECONDS`, because unbounded
      exponential back-off walked to 259s under a run of 429s;
    * a failure that took at least :data:`_WALL_CLOCK_SECONDS` is not retried at
      all — around 11% of sobhard documents exceed the gateway's limit at every
      max_tokens tried, and retrying them costs five minutes to learn nothing;
    * anything still failing after `max_attempts` yields `give_up_value` instead
      of raising, because `generate_until` turns a raised ResponseError into a
      `break` that abandons every remaining document in the task.
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            sleep_time = _INITIAL_BACKOFF_SECONDS
            last_error = None
            attempt = 0
            while attempt < max_attempts:
                started = time.time()
                try:
                    return func(*args, **kwargs)
                except tuple(on_exceptions) as e:
                    elapsed = time.time() - started
                    last_error = e
                    # A 429 says "not now", not "this request is bad". Waiting it
                    # out is the whole remedy, so it must not consume an attempt:
                    # bursts of 16-22 in a row were measured under contention,
                    # and counting them would discard perfectly good documents.
                    if _error_status(e) == 429:
                        if on_exception_callback is not None:
                            on_exception_callback(e, sleep_time)
                        time.sleep(sleep_time)
                        sleep_time = min(
                            sleep_time * _BACKOFF_MULTIPLIER, _MAX_BACKOFF_SECONDS
                        )
                        continue
                    # 401/403 is not the model failing, it is us not being
                    # allowed to ask. Retrying cannot help and giving up writes
                    # an empty answer that reads as a wrong answer: an expired
                    # token silently turned 241 sobhard documents on
                    # GigaChat-3-Ultra into blanks, and the run looked healthy
                    # throughout. Fail loudly instead.
                    if _error_status(e) in (401, 403):
                        eval_logger.critical(
                            "gigachat: %s from the API — the token is rejected "
                            "(expired or wrong scope). Aborting instead of "
                            "recording empty answers.",
                            _error_status(e),
                        )
                        raise
                    attempt += 1
                    # The wall is a 502 from nginx at ~300s, and only that.
                    # Elapsed time alone used to be enough to recognise it,
                    # because without streaming nothing else could run that long
                    # and still fail. With streaming a request legitimately lives
                    # far longer (948s measured), and when the gateway drops it
                    # mid-body the failure is an httpx protocol error carrying no
                    # status — transient, and worth retrying. Classifying that as
                    # the wall gave up instantly and wrote an empty answer:
                    # measured at 1 of the first 5 streamed sobhard documents,
                    # which over 703 of them is ~140 blanks that no retry policy
                    # would ever revisit.
                    if elapsed >= _WALL_CLOCK_SECONDS and _error_status(e) == 502:
                        eval_logger.error(
                            "gigachat: request hit the gateway wall after %.0fs; "
                            "recording an empty answer without retrying. %s",
                            elapsed,
                            e,
                        )
                        return give_up_value
                    if attempt == max_attempts:
                        break
                    if on_exception_callback is not None:
                        on_exception_callback(e, sleep_time)
                    time.sleep(sleep_time)
                    sleep_time = min(
                        sleep_time * _BACKOFF_MULTIPLIER, _MAX_BACKOFF_SECONDS
                    )
            eval_logger.error(
                "gigachat: giving up after %d attempts, recording an empty "
                "answer. Last error: %s",
                max_attempts,
                last_error,
            )
            return give_up_value

        return wrapper

    return decorator


def _chat_field_names() -> set | None:
    """Field names gigachat's Chat model declares, or None if it cannot be read."""
    try:
        from gigachat.models import Chat
    except ModuleNotFoundError:
        return None
    fields = getattr(Chat, "model_fields", None)  # pydantic v2
    if fields is None:
        fields = getattr(Chat, "__fields__", None)  # pydantic v1
    return set(fields) if fields else None


def _normalize_gigachat_kwargs(kwargs: dict) -> dict:
    """Map CLI-friendly model_args to GigaChat Chat() payload fields.

    Anything the installed SDK's ``Chat`` model does not declare is routed
    through ``additional_fields``, which ``gigachat.api.chat._build_request_json``
    merges into the top level of the request body. Without that, pydantic drops
    unknown keys silently: ``tools`` and ``model_options`` are not fields of
    ``Chat`` in gigachat 0.2.3, so passing them directly meant they never
    reached the API at all, with nothing in the logs to say so.
    """
    normalized = dict(kwargs)
    for key, value in list(normalized.items()):
        normalized[key] = _coerce_model_arg(value)

    for key in _NOT_PAYLOAD_KEYS:
        normalized.pop(key, None)

    preset = normalized.pop("preset", None)
    if preset is not None and "model_options" not in normalized:
        normalized["model_options"] = {"preset": preset}

    if "tools" in normalized and normalized["tools"] is None:
        normalized["tools"] = []

    known = _chat_field_names()
    if known:
        extra = {k: normalized.pop(k) for k in list(normalized) if k not in known}
        if extra:
            merged = dict(normalized.get("additional_fields") or {})
            merged.update(extra)
            normalized["additional_fields"] = merged

    return normalized


def gigachat_completion(
    client,  #: gigachat.GigaChat,
    model: str,
    prompt: str,
    max_tokens_to_sample: int,
    temperature: float,
    until: list[str],
    chat_template_is_on: bool,
    **kwargs,
) -> str:
    """Wrapper function around the GigaChat API client with exponential back-off
    in case of RateLimitError.
    For authorization set environmental variables "GIGACHAT_CREDENTIALS" and "GIGACHAT_SCOPE" for your API auth_data and scope (GIGACHAT_API_CORP or GIGACHAT_API_PERS) respectively.
    Skip sample after 5 retries if there is an error with GigaChat API occurred.
    params:
        client: gigachat.GigaChat
            GigaChat API client
        model: str
            GigaChat model, possible values: [GigaChat, GigaChat:latest, GigaChat-Plus, GigaChat-Pro]
        prompt: str
            Prompt to feed to the model
        max_tokens: int
            Maximum number of tokens to sample from the model
        temperature: float
            Sampling temperature
        until: List[str]
            List of stop-words
        chat_template_is_on: bool
            Use chat_template or not
        kwargs: Any
            Additional model_args to pass to the API client. May be:
            profanity check: bool, censor status. Default: True
            top_p: float, nucleus params. The default value depends on the selected model and may change with model updates
            repetition_penalty: float, repetition_penalty. The default value depends on the selected model and may change with model updates
            n: int, the number of response options to be generated for each input message. Possible values: [1; 4]. Default: 1
            stream: bool, specifies that messages should be sent in parts in the stream. Default: False
    """
    try:
        import gigachat
        import httpx
        import pydantic
    except ModuleNotFoundError:
        raise Exception(
            "attempted to use 'gigachat' LM type, but packages `gigachat` or `httpx` are not installed. \
please install gigachat via `pip install lm-eval[gigachat]` or `pip install -e .[gigachat]`",
        )

    kwargs = _normalize_gigachat_kwargs(kwargs)
    # Client-side switch, not a payload field: gigachat's `stream()` sets
    # `stream` on the request itself, and leaving it in kwargs would send it
    # twice.
    stream = bool(kwargs.pop("stream", False))

    messages = []
    if not chat_template_is_on:
        messages.append(
            gigachat.models.Messages(
                role=gigachat.models.MessagesRole.USER,
                content=prompt,
            )
        )
    else:
        seq = prompt.split("<role>")[1:]
        for message in seq:
            role, content = message.split("<content>")
            messages.append(
                gigachat.models.Messages(
                    role=role,
                    content=content,
                )
            )

    def _exception_callback(e: Exception, sleep_time: float = 10) -> None:
        eval_logger.warning(
            f"GigaChatError occurred: {e.__str__()}\n Retrying in {sleep_time} seconds"
        )

    @_retry_with_capped_backoff(
        on_exceptions=[
            httpx.ReadTimeout,  # it is like a RateLimitError
            httpx.ConnectTimeout,
            gigachat.exceptions.ResponseError,
            httpx.RemoteProtocolError,
            # Under `stream=true` the gateway reports a mid-generation failure
            # as a JSON object *inside* the SSE stream —
            # {"status_code": 500, "message": "Internal Server Error"} — and the
            # SDK feeds that to ChatCompletionChunk, which rejects it for four
            # missing fields. The exception is a pydantic ValidationError, not a
            # ResponseError, so without this entry it escapes the retry wrapper
            # and `generate_until` aborts the whole task: GigaChat-3-Lightning
            # lost all 825 sobhard documents to one such chunk. It is a
            # transient 500, so it belongs here rather than in a `raise`.
            pydantic.ValidationError,
        ],
        on_exception_callback=_exception_callback,
    )
    def completion():
        payload = gigachat.models.Chat(
            messages=messages,
            model=model,
            max_tokens=max_tokens_to_sample,
            temperature=temperature,
            # profanity_check=False,
            # tools=[],
            # model_options={"preset": "default"},
            **kwargs,
        )

        if stream:
            # The gateway ends a silent response at about 300 seconds and
            # answers 502 — measured at 288-301s across dozens of requests, with
            # the client timeout at 1800s, so the limit is theirs and waiting
            # longer cannot help. Streaming keeps the connection producing
            # chunks, so the read timeout never fires; it is the only lever that
            # reaches those documents. 5% of sobhard is lost without it.
            parts = []
            for chunk in client.stream(payload):
                for choice in chunk.choices or []:
                    piece = getattr(choice.delta, "content", None)
                    if piece:
                        parts.append(piece)
            response = "".join(parts)
        else:
            response = client.chat(payload).choices[0].message.content

        if until:
            response = cut_generation(response, until)
        if not response:
            response = " "  # avoid None in resps
        return response

    return completion()


@register_model("gigachat-completion")
class GigaChatLM(LM):
    def __init__(
        self,
        model: str = "GigaChat",
        max_tokens: int
        | None = None,  # default is None as API will automatically choose the most optimal value
        temperature: float | None = None,
        scope: str = "GIGACHAT_API_PERS",
        verify_ssl_certs: bool = False,
        base_url: str | None = None,
        timeout: float = 200,
        **kwargs,  # top_p,  etc.
    ) -> None:
        """GigaChat API wrapper.

        :param model: str
            GigaChat model, possible values: [GigaChat, GigaChat:latest, GigaChat-Plus, GigaChat-Pro]
        :param max_tokens_to_sample: int
            Maximum number of tokens to sample from the model
        :param temperature: float
            Sampling temperature. Cannot be set to zero!
        :param scope: str
            Set tokenscope. Possible values are: ['GIGACHAT_API_PERS', 'GIGACHAT_API_CORP', 'GIGACHAT_API_B2B']
        :param verify_ssl_certs: bool
            Set this parameter if you have your certificates installed to ensure greater security
        :param timeout: float
            HTTP read timeout in seconds. The default suits short answers; a task
            that generates tens of thousands of characters (sobhard asks for
            max_gen_toks 65536) needs far more, and a read timeout here costs the
            full wait and then an exponential back-off before the retry.
        :param kwargs: Any
            Additional model_args to pass to the API client.
        """
        super().__init__()

        try:
            import gigachat
        except ModuleNotFoundError:
            raise Exception(
                "attempted to use 'gigachat' LM type, but packages `gigachat` or `httpx` are not installed. \
please install gigachat via `pip install lm-eval[gigachat]` or `pip install -e .[gigachat]`",
            )

        self.model = model
        self.client = gigachat.GigaChat(
            base_url=base_url,
            credentials=os.environ.get("GIGACHAT_CREDENTIALS", None),
            access_token=os.environ.get("GIGACHAT_TOKEN", None),
            scope=os.environ.get("GIGACHAT_SCOPE", scope),
            verify_ssl_certs=verify_ssl_certs,
            timeout=timeout,
        )
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.kwargs = _normalize_gigachat_kwargs(kwargs)
        self.chat_template_is_used = False

    @property
    def eot_token_id(self):
        raise NotImplementedError("No idea about GigaChat tokenization.")

    @property
    def max_length(self) -> int:
        return None

    @property
    def max_gen_toks(self) -> int:
        """
        Set max_gen_toks to None as API itself defines max token limit for each model type.
        """
        return None

    @property
    def batch_size(self):
        # Isn't used because we override _loglikelihood_tokens
        raise NotImplementedError("No support for logits.")

    @property
    def device(self):
        # Isn't used because we override _loglikelihood_tokens
        raise NotImplementedError("No support for logits.")

    def tok_encode(self, string: str) -> list[int]:
        return NotImplementedError("No idea about GigaChat tokenization.")

    def tok_decode(self, tokens: list[int]) -> str:
        return NotImplementedError("No idea about GigaChat tokenization.")

    def _loglikelihood_tokens(self, requests, disable_tqdm: bool = False):
        raise NotImplementedError("No support for logits.")

    def generate_until(self, requests, disable_tqdm: bool = False) -> list[str]:
        try:
            import gigachat
        except ModuleNotFoundError:
            raise Exception(
                "attempted to use 'gigachat' LM type, but packages `gigachat` or `httpx` are not installed. \
please install gigachat via `pip install lm-eval[gigachat]` or `pip install -e .[gigachat]`",
            )

        if not requests:
            return []

        _requests: list[tuple[str, dict]] = [req.args for req in requests]

        res = []
        for request in tqdm(_requests, disable=disable_tqdm):
            try:
                inp = request[0]
                request_args = request[1]
                until = request_args.get("until")
                if isinstance(until, str):
                    until = [until]
                # generation_kwargs
                max_gen_toks = request_args.get("max_gen_toks", None)
                temperature = request_args.get("temperature", self.temperature)

                if (
                    "do_sample" in self.kwargs.keys()
                ):  # API does not have do sample option.
                    if not self.kwargs[
                        "do_sample"
                    ]:  # Ensure greedy decoding if do_sample=False
                        self.kwargs["repetition_penalty"] = 1
                        self.kwargs["top_p"] = 0
                    elif temperature == 0:
                        eval_logger.warning(
                            "You cannot set do_sample=True and temperature=0. Automatically setting temperature=1."
                        )
                        temperature = 1.0

                if (
                    temperature == 0
                ):  # Ensure greedy decoding by setting top_p=0 and repetition_penalty = 1
                    temperature = (
                        1.0  # temperature cannot be set to zero. Use top_p instead
                    )
                    self.kwargs["repetition_penalty"] = 1
                    self.kwargs["top_p"] = 0

                if not self.chat_template_is_used:
                    eval_logger.warning(
                        "You are trying to use GigaChat without chat_template. It may lead to inappropriate model behavior. \
                            Please, set `--apply_chat_template` and `--system_instruction`  arguments."
                    )

                response = gigachat_completion(
                    client=self.client,
                    model=self.model,
                    prompt=inp,
                    max_tokens_to_sample=max_gen_toks,
                    temperature=temperature,
                    until=until,
                    chat_template_is_on=self.chat_template_is_used,
                    **self.kwargs,
                )

                res.append(response)

                self.cache_hook.add_partial("generate_until", request, response)
            except (gigachat.exceptions.ResponseError,) as e:
                status, mes = parse_exception(e)
                eval_logger.critical(f"""API error {status}: {mes}""")
                break
        return res

    def apply_chat_template(self, chat_history: list[dict[str, str]], **kwargs) -> str:
        """
        Apply chat template in the gigachat_completion func using gigachat library
        We do not have access to gigachat tokenizer. This is our solution:
        Set chat_history as an attribute and pass it to chat completion func.
        Return a list as a string to avoid raising errors.
        """
        if not self.chat_template_is_used:
            self.chat_template_is_used = True
        prompt = ""
        for dct in chat_history:
            prompt += f"<role>{dct['role']}<content>{dct['content']}"
        return prompt

    @property
    def tokenizer_name(self) -> str:
        """
        Apply chat template in the gigachat_completion func using gigachat library.
        We do not have access to gigachat tokenizer.
        Return gigachat_tokenizer as a name.
        """
        return "gigachat_tokenizer"

    def chat_template(self, chat_template: bool | str = False) -> str:
        """
        Apply chat template in the gigachat_completion func using gigachat library.
        We do not have access to gigachat tokenizer.
        """
        return ""

    def _model_call(self, inps):
        # Isn't used because we override _loglikelihood_tokens
        raise NotImplementedError()

    def _model_generate(self, context, max_length, eos_token_id):
        # Isn't used because we override generate_until
        raise NotImplementedError()

    def loglikelihood(self, requests, disable_tqdm: bool = False):
        raise NotImplementedError("No support for logits.")

    def loglikelihood_rolling(self, requests, disable_tqdm: bool = False):
        raise NotImplementedError("No support for logits.")


@register_model("gigachat-chat")
class GigaChatAPI(LocalChatCompletion):
    def __init__(
        self,
        base_url=None,
        auth_url=None,  # authorization url to get acess_token
        verify_certificate=False,
        **kwargs,
    ):
        super().__init__(
            base_url=base_url,
            verify_certificate=verify_certificate,
            **kwargs,
        )
        self.expiration_time = 0
        self.auth_url = auth_url

    def _create_payload(
        self,
        messages: list[list[int]] | list[dict] | list[str] | str,
        generate=False,
        gen_kwargs: dict | None = None,
        **kwargs,
    ) -> dict:
        if generate:
            temperature = gen_kwargs.pop("temperature", None)
            do_sample = gen_kwargs.pop("do_sample", None)

            if do_sample is not None:  # GigaChat API does not have do sample option.
                if not do_sample:  # Ensure greedy decoding if do_sample=False
                    gen_kwargs["repetition_penalty"] = 1.0
                    gen_kwargs["top_p"] = 0.0
                elif temperature == 0.0:
                    eval_logger.warning(
                        "You cannot set do_sample=True and temperature=0. Automatically setting temperature=1."
                    )
                    temperature = 1.0
            if (
                temperature == 0.0
            ):  # Ensure greedy decoding by setting top_p=0 and repetition_penalty = 1
                temperature = (
                    1.0  # temperature cannot be set to zero. Use top_p instead
                )
                gen_kwargs["repetition_penalty"] = 1.0
                gen_kwargs["top_p"] = 0.0
            return {
                "messages": messages,
                "model": self.model,
                "temperature": temperature,
                **gen_kwargs,
            }
        else:
            return None

    @property  # Don't use cached_property as we need to check that the access_token has not expired.
    def header(self) -> dict:
        """Override this property to return the headers for the API request."""
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Accept": "application/json",
            "Content-Type": "application/json",
            "User-Agent": "Gigaclient",
        }

    @property  # Don't use cached_property as we need to check that the acess_token has not expired.
    def api_key(self):
        self.key = os.environ.get("GIGACHAT_TOKEN", None)  # GigaChat access token.
        if self.key:
            return self.key  # If access token is available, return access token.
        RqUID = os.environ.get(
            "GIGACHAT_RQUID", None
        )  # Unique identification request. Complies with uuid4 format. Value must match regular expression (([0-9a-fA-F-])36)
        auth_token = os.environ.get(
            "GIGACHAT_CREDENTIALS", None
        )  # Client Secret. Credential for GigaChat API.
        scope = os.environ.get(
            "SCOPE", None
        )  # type of your API. Possible values: [GIGACHAT_API_PERS, GIGACHAT_API_B2B, GIGACHAT_API_CORP].
        if not scope:
            scope = "GIGACHAT_API_PERS"
            eval_logger.warning(
                "SCOPE environment variable not found. Automatically set to GIGACHAT_API_PERS."
            )

        if RqUID is None or auth_token is None:
            raise ValueError(
                "Credentials not found. Please set GIGACHAT_RQUID and GIGACHAT_TOKEN environment variables."
            )
        if self.expiration_time == 0 or self.expiration_time < int(
            time.time() * 1000
        ):  # Check if the access token exists and is valid. If not, create a new one
            try:
                token_ = self._get_token_gigachat(RqUID, auth_token, scope)
                self.key, self.expiration_time = (
                    token_["access_token"],
                    token_["expires_at"],
                )
            except Exception as e:
                raise ValueError(
                    f"Invalid credentials: {e}. Please set correct GIGACHAT_RQUID and GIGACHAT_TOKEN environment variables. Or check that the SCOPE was set correctly."
                )
        return self.key

    def _get_token_gigachat(self, rqUID: str, auth_token: str, scope: str) -> str:
        """
        Creates temporal token using credentials.
        rqUID - Unique identification request. Complies with uuid4 format. Value must match regular expression (([0-9a-fA-F-])36)
        auth_token - Client Secret. Credential for GigaChat API.
        scope - type of your API. Possible values: [GIGACHAT_API_PERS, GIGACHAT_API_B2B, GIGACHAT_API_CORP].
        Returns an access token for authorizing API requests. The access token is valid for 30 minutes. Issue it if current time > expiration time.
        """

        payload = f"scope={scope}"
        headers = {
            "Content-Type": "application/x-www-form-urlencoded",
            "Accept": "application/json",
            "RqUID": rqUID,
            "Authorization": f"Basic {auth_token}",
        }

        response = requests.request(
            "POST",
            self.auth_url,
            headers=headers,
            data=payload,
            verify=False,
        )
        return json.loads(response.text)


def cut_generation(generation, stop):
    """
    GigaChat API has no stop argument.
    Use this func in order to cut GigaChat generation.
    """
    if not generation:
        generation = " "
    stop_idxs = [generation.find(sub) for sub in stop if generation.find(sub) != -1]
    if stop_idxs:
        generation = generation[: min(stop_idxs)]
    return generation


def parse_exception(exp):
    """Status code and message from a gigachat ResponseError.

    `ResponseError.__init__` keeps url, status_code, content and headers on the
    instance but calls `super().__init__(f"{status_code} {url}")` — so `args`
    holds exactly ONE formatted string. The previous body indexed `args[2]` and
    `args[1]`, which raises `IndexError: tuple index out of range` for every
    error it is handed.

    That turned the one place this is called — the `except ResponseError` in
    `generate_until` — into a landmine: a deliberate, correct 401 abort came out
    as an unexplained IndexError traceback, and the actual cause (an expired
    token) appeared nowhere in the failure. It cost two 20-hour sobhard runs
    their diagnosis, on GigaChat-3-Ultra at document 206 of 296 and on
    GigaChat-3.5 at 136 of 325.
    """
    status = getattr(exp, "status_code", None)
    content = getattr(exp, "content", None)
    if isinstance(content, (bytes, bytearray)):
        content = content.decode("utf8", errors="replace")
    message = None
    if isinstance(content, str) and content.strip():
        try:
            parsed = json.loads(content)
        except (json.JSONDecodeError, ValueError):
            message = content[:300]
        else:
            if isinstance(parsed, dict):
                status = parsed.get("status", status)
                message = parsed.get("message") or parsed.get("error")
            else:
                message = content[:300]
    return status, message if message is not None else str(exp)
