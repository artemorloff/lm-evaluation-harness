from __future__ import annotations

import inspect
import logging
import warnings
from functools import partial
from random import Random
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence
    from typing import Any, TypeVar

    _T = TypeVar("_T")

eval_logger = logging.getLogger(__name__)


def is_legacy_sampler(cls: type) -> bool:
    """True for samplers written against the pre-0.4.13 sampler API.

    Two sampler generations are in the wild. The current one only *picks*
    documents -- ``sample(n, eval_doc=...)`` -- and ``Task.fewshot_context``
    renders them. The older one renders the context itself, in ``get_context``
    / ``get_chat_context``, and receives the evaluated document as the second
    positional argument of ``sample``.

    A sampler that overrides the rendering methods *and* whose ``sample`` has no
    ``eval_doc`` parameter can only have been written against the old API: the
    current pipeline would never call its rendering methods, and would call its
    ``sample`` with a keyword it does not accept. Such a task is routed through
    the legacy context builder so its prompts stay byte-for-byte what they were.

    A sampler that implements both surfaces (``sample`` takes ``eval_doc`` *and*
    ``get_context`` exists) is deliberately version-agnostic and is left on the
    current path.
    """
    renders = (
        cls.get_context is not ContextSampler.get_context
        or cls.get_chat_context is not ContextSampler.get_chat_context
    )
    if not renders:
        return False
    try:
        params = inspect.signature(cls.sample).parameters
    except (TypeError, ValueError):  # pragma: no cover - builtins/C callables
        return True
    return "eval_doc" not in params


class ContextSampler:
    def __init__(
        self,
        df: Sequence[dict[str, Any]] | None = None,
        *,
        rnd: int | None = None,
        fewshot_indices: list[int] | None = None,
        task: Any | None = None,
        **kwargs,
    ) -> None:
        self.rnd = Random(rnd) if not isinstance(rnd, Random) else rnd
        self.df = df or []
        self.fewshot_indices = fewshot_indices
        self._loaded = False  # to iterate over fewshot_indices when needed
        self._bind_task(task)

    # ------------------------------------------------------------------
    # Legacy sampler surface (lm-eval <= 0.4.9).
    #
    # Samplers of that generation build the context themselves and read the
    # task off the sampler: ``self.task``, ``self.config``, the delimiters and
    # the ``doc_to_*`` callables bound to ``fewshot_config``. Populating them
    # here is what lets such a sampler keep working unchanged; a sampler that
    # only picks documents never touches any of it.
    # ------------------------------------------------------------------
    def _bind_task(self, task: Any | None) -> None:
        self.task = task
        if task is None:
            self.config = None
            self.target_delimiter = " "
            self.fewshot_delimiter = "\n\n"
            return

        self.config = task._config
        self.target_delimiter = self.config.target_delimiter
        self.fewshot_delimiter = self.config.fewshot_delimiter

        fs_cfg = self.config.fewshot_config
        explicit = getattr(fs_cfg, "explicit", None)
        if explicit is None:  # plain dict, e.g. a hand-built config
            explicit = set(fs_cfg or ())
        for field_name in ("doc_to_text", "doc_to_target", "doc_to_choice"):
            task_fn = getattr(task, field_name)
            # Only a template the task spelled out under `fewshot_config` pins the
            # renderer. Binding the inherited task-level template here instead
            # would freeze it, and a sampler that swaps `config.doc_to_text`
            # while assembling shots -- the documented way to give the first
            # shot the instruction and the rest none -- would stop taking effect.
            override = fs_cfg.get(field_name, None) if field_name in explicit else None
            setattr(
                self,
                field_name,
                partial(task_fn, **{field_name: override}) if override else task_fn,
            )

    @property
    def docs(self):
        """The fewshot pool under the name the old API used."""
        return self.fewshot_docs()

    @docs.setter
    def docs(self, value):
        self.df = value
        self._loaded = True

    def get_context(self, doc: dict, num_fewshot: int, gen_prefix: str | None = None):
        """Render the fewshot block as plain text (legacy sampler API)."""
        prefix = gen_prefix + " " if gen_prefix else ""
        # draw an extra fewshot sample if using same split as evaluating on
        n_samples = (
            num_fewshot + 1
            if self.config.fewshot_split == self.config.test_split
            else num_fewshot
        )
        fewshotex = self.sample(n_samples)

        # get rid of the doc that's the one we're evaluating, if it's in the fewshot
        selected_docs = [x for x in fewshotex if x != doc][:num_fewshot]

        labeled_examples = ""
        for doc in selected_docs:
            doc_content = self.doc_to_text(doc)
            doc_target = self.doc_to_target(doc)
            if self.config.doc_to_choice is None or isinstance(doc_content, str):
                labeled_examples += doc_content
            else:
                labeled_examples += self.doc_to_choice(doc)[doc_content]

            if doc_target != "":
                if self.target_delimiter.isspace() and str(doc_target)[0].isspace():
                    warnings.warn(
                        "Both target_delimiter and target start with a space. This may cause issues.",
                        Warning,
                        stacklevel=2,
                    )
                labeled_examples += self.target_delimiter
                labeled_examples += prefix
                labeled_examples += (
                    str(doc_target[0])
                    if isinstance(doc_target, list)
                    else doc_target
                    if self.config.doc_to_choice is None or isinstance(doc_target, str)
                    else str(self.doc_to_choice(doc)[doc_target])
                )
                labeled_examples += self.fewshot_delimiter

        return labeled_examples

    def get_chat_context(
        self,
        doc: dict,
        num_fewshot: int,
        fewshot_as_multiturn: bool = False,
        gen_prefix: str | None = None,
    ):
        """Render the fewshot block as chat turns (legacy sampler API)."""
        prefix = gen_prefix + " " if gen_prefix else ""
        chat_history = []
        n_samples = (
            num_fewshot + 1
            if self.config.fewshot_split == self.config.test_split
            else num_fewshot
        )
        fewshotex = self.sample(n_samples)
        selected_docs = [x for x in fewshotex if x != doc][:num_fewshot]

        if fewshot_as_multiturn:
            for doc in selected_docs:
                doc_content = self.doc_to_text(doc)
                doc_target = self.doc_to_target(doc)
                chat_history.append(
                    {
                        "role": "user",
                        "content": doc_content
                        if self.config.doc_to_choice is None
                        or isinstance(doc_content, str)
                        else self.doc_to_choice(doc)[doc_content],
                    }
                )
                chat_history.append(
                    {
                        "role": "assistant",
                        "content": prefix + str(doc_target[0])
                        if isinstance(doc_target, list)
                        else prefix + doc_target
                        if self.config.doc_to_choice is None
                        or isinstance(doc_target, str)
                        else prefix + str(self.doc_to_choice(doc)[doc_target]),
                    }
                )
        else:
            chat_history.append(
                {
                    "role": "user",
                    "content": self.get_context(doc, num_fewshot, gen_prefix=gen_prefix),
                }
            )

        return chat_history

    def sample(
        self,
        n: int,
        eval_doc: dict[str, Any] | None = None,
        df: Sequence[dict[str, Any]] | None = None,
        **kwargs,
    ) -> Sequence[dict[str, Any]]:
        """
        Sample n documents from the pool.

        Args:
            n: Number of documents to sample
            eval_doc: Optional document to exclude from sampling
            df: Optional list of documents to sample from

        Returns:
            List of sampled documents
        """
        assert n >= 0, "Error: number of samples requested must be >=0"
        if n == 0:
            return []

        if df:
            self.df = df

        assert self.df, "Error: no documents available for sampling."
        res = (
            self.rnd.sample(self.fewshot_docs(), n)
            if not eval_doc
            else self.rm_eval_doc(
                eval_doc, self.rnd.sample(self.fewshot_docs(), n + 1), n
            )
        )
        assert len(res) == n, (
            f"Error: number of fewshot samples returned ({len(res)}) not equal to number requested ({n})."
        )
        return res

    def set_rnd(self, rnd: int | None):
        self.rnd = Random(rnd) if not isinstance(rnd, Random) else rnd
        return self

    def replace_df(self, df: Sequence[dict[str, Any]]):
        self.df = df
        self._loaded = False
        return self

    def fewshot_docs(self):
        """Return cached fewshot docs if available"""
        if self._loaded:
            return self.df
        if self.fewshot_indices and self.df and not self._loaded:
            self.df = [self.df[i] for i in self.fewshot_indices]
        self._loaded = True
        return list(self.df)

    @staticmethod
    def rm_eval_doc(doc: _T, _iter: Iterable[_T], n=None) -> Sequence[_T]:
        return (
            [x for x in _iter if x != doc]
            if n is None
            else [x for x in _iter if x != doc][:n]
        )


class FirstNSampler(ContextSampler):
    def sample(self, n: int, eval_doc=None, df=None, **kwargs):
        """
        Draw the first `n` samples in order from the specified split.
        Used for tasks with "canonical" ordered fewshot examples, such as MMLU and CMMLU.
        """
        pool = self.rm_eval_doc(eval_doc, self.df) if eval_doc is not None else self.df
        assert n <= len(pool), (
            f"Error: number of fewshot samples requested exceeds the {len(pool)} that are available."
        )
        return pool[:n]


class BalancedSampler(ContextSampler):
    def sample(self, n: int, eval_doc=None, df=None, **kwargs):
        """
        TODO: this should return approximately class-balanced samples from our fewshot examples.
        TODO: what order should they be in? maybe random?
        """

        raise NotImplementedError


class ManualSampler(ContextSampler):
    def sample(self, n: int, eval_doc=None, df=None, **kwargs):
        raise NotImplementedError


SAMPLER_REGISTRY: dict[str, type[ContextSampler]] = {
    "default": ContextSampler,
    "first_n": FirstNSampler,
}


def get_sampler(name: str):
    try:
        return SAMPLER_REGISTRY[name]
    except KeyError as e:
        raise KeyError(
            f"Attempted to use contextsampler '{name}', but no sampling strategy for this name found! Supported model names: {', '.join(SAMPLER_REGISTRY.keys())}"
        ) from e
