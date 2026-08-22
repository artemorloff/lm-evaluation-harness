from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field, fields
from inspect import getsource
from typing import TYPE_CHECKING, Any

from lm_eval.defaults import default_gen_kwargs


if TYPE_CHECKING:
    from collections.abc import Callable

    from lm_eval.api.instance import OutputType


eval_logger = logging.getLogger(__name__)


@dataclass
class FewshotConfig:
    """Configuration for few-shot example formatting.

    These fields override the parent TaskConfig fields when formatting
    few-shot examples (as opposed to the test example).

    note: num_fewshot is also runtime dependent, so is not included here.
    """

    sampler: str = "default"
    split: str | None = None
    process_docs: Callable[..., list[dict]] | None = None
    fewshot_indices: list[int] | None = None
    samples: list[dict] | Callable[[], list[dict]] | None = None
    # Override doc formatting for fewshot examples
    doc_to_text: str | Callable[..., str] | None = None
    doc_to_choice: str | Callable[..., str] | dict | list | None = None
    doc_to_target: str | Callable[..., str] | None = None
    gen_prefix: str | None = None
    fewshot_delimiter: str | None = None
    target_delimiter: str | None = None
    #: Keys of `fewshot_config` this dataclass does not name. Task-supplied
    #: samplers are free to define their own, and used to receive them because
    #: `fewshot_config` was a plain dict; dropping them here would silently
    #: change how those tasks build their prompts.
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.split is not None and self.samples is not None:
            eval_logger.warning(
                "Both split and samples are configured; split will take precedence"
            )

    # -- mapping surface, for samplers that read `fewshot_config` as a dict ---
    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self[key]
        except KeyError:
            return default

    def __getitem__(self, key: str) -> Any:
        if key in self.extra:
            return self.extra[key]
        if key in {f.name for f in fields(self) if f.name != "extra"}:
            return getattr(self, key)
        raise KeyError(key)

    def __setitem__(self, key: str, value: Any) -> None:
        if key in {f.name for f in fields(self) if f.name != "extra"}:
            setattr(self, key, value)
        else:
            self.extra[key] = value

    def __contains__(self, key: str) -> bool:
        return key in self.extra or key in {
            f.name for f in fields(self) if f.name != "extra"
        }

    def keys(self):
        return [f.name for f in fields(self) if f.name != "extra"] + list(self.extra)

    @property
    def explicit(self) -> set[str]:
        """Keys the task's own `fewshot_config` block set, as opposed to the
        ones inherited from the enclosing TaskConfig.

        A consumer that asks "did this task override `doc_to_text` for shots?"
        needs the difference: every field here has an inherited value, so a
        plain `is not None` check answers yes for all of them.
        """
        return getattr(self, "_explicit", set())

    def to_dict(self) -> dict[str, Any]:
        d = {f.name: getattr(self, f.name) for f in fields(self) if f.name != "extra"}
        d.update(self.extra)
        return d

    @classmethod
    def from_dict(
        cls,
        cfg: dict,
        *,
        # inherited from TaskConfig if not specified
        fewshot_split: str | None = None,
        process_docs: Callable[..., list[dict]] | None = None,
        fewshot_delimiter: str | None = None,
        target_delimiter: str | None = None,
        gen_prefix: str | None = None,
        doc_to_text: str | Callable[..., str] | None = None,
        doc_to_choice: str | Callable[..., str] | dict | list | None = None,
        doc_to_target: str | Callable[..., str] | None = None,
        **overloads,
    ) -> FewshotConfig:
        cfg_dict = {
            "split": fewshot_split,
            "process_docs": process_docs,
            "fewshot_delimiter": fewshot_delimiter,
            "target_delimiter": target_delimiter,
            "gen_prefix": gen_prefix,
            "doc_to_text": doc_to_text,
            "doc_to_choice": doc_to_choice,
            "doc_to_target": doc_to_target,
            **cfg,
            **overloads,
        }
        cfg_dict.setdefault("sampler", "default")
        known = {f.name for f in fields(cls)}
        extra = {k: v for k, v in cfg_dict.items() if k not in known}
        known_kwargs = {k: v for k, v in cfg_dict.items() if k in known}
        known_kwargs.setdefault("extra", {})
        known_kwargs["extra"] = {**known_kwargs["extra"], **extra}
        obj = cls(**known_kwargs)
        obj._explicit = set(cfg) | set(overloads)
        return obj


@dataclass
class TaskConfig(dict):
    # task naming/registry
    task: str | None = None
    task_alias: str | None = None
    tag: str | list | None = None
    # HF dataset options.
    # which dataset to use,
    # and what splits for what purpose
    custom_dataset: Callable | None = None
    dataset_path: str | None = None
    dataset_name: str | None = None
    dataset_kwargs: dict | None = None
    # context-based flag and funcs
    context_based: bool | None = None
    request_updater: Callable | None = None
    storage_updater: Callable | None = None
    training_split: str | None = None
    validation_split: str | None = None
    test_split: str | None = None
    fewshot_split: str | None = (
        None  # TODO: assert that this not None if num_fewshot > 0. (?) assert if this is same split as one evaluating (?)
    )
    # formatting / prompting options.
    # see docs/advanced_task_guide.md for more info
    process_docs: Callable | None = None
    doc_to_text: Callable | str | None = None
    doc_to_target: Callable | str | None = None
    doc_to_image: Callable | str | None = None
    doc_to_audio: Callable | str | None = None
    unsafe_code: bool = False
    doc_to_choice: Callable | str | dict | list | None = None
    process_results: Callable | str | None = None
    use_prompt: str | None = None
    description: str = ""
    target_delimiter: str = " "
    fewshot_delimiter: str = "\n\n"
    fewshot_config: dict[str, Any] | FewshotConfig | None = None
    # runtime configuration options
    num_fewshot: int | None = None
    # scoring options
    metric_list: list | None = None
    output_type: OutputType = "generate_until"
    generation_kwargs: dict | None = None
    repeats: int = 1
    filter_list: str | list | None = None
    should_decontaminate: bool = False
    doc_to_decontamination_query: str | None = None
    gen_prefix: str | None = None
    metadata: dict | None = (
        None  # by default, not used in the code. allows for users to pass arbitrary info to tasks
    )

    def __post_init__(self) -> None:
        if self.generation_kwargs is not None:
            if self.output_type != "generate_until":
                eval_logger.warning(
                    f"[{self.task}] passed `generation_kwargs`, but not using `output_type: generate_until`!"
                )

            if "temperature" in self.generation_kwargs:
                self.generation_kwargs["temperature"] = float(
                    self.generation_kwargs["temperature"]
                )

            if "until" not in self.generation_kwargs:
                eval_logger.warning(
                    f"{self.task}: No `until` specified in `generation_kwargs`! Defaulting to the fewshot_delimiter={repr(self.fewshot_delimiter)}"
                )
                self.generation_kwargs["until"] = [self.fewshot_delimiter]
        else:
            if self.output_type == "generate_until":
                # ensure that we greedily generate in absence of explicit arguments otherwise
                # No `max_gen_toks`: a task that does not spell out
                # `generation_kwargs` must keep taking its length cap from the
                # model backend (or from `--gen_kwargs`). Pinning one here
                # would cut long answers short in tasks that never asked for a
                # cap, and it would show up in every logged request.
                self.generation_kwargs = default_gen_kwargs(
                    self.fewshot_delimiter, max_gen_toks=None
                )
                eval_logger.warning(
                    f"{self.task}: No `generation_kwargs` specified in task config, defaulting to {self.generation_kwargs}"
                )
        self.fewshot_config = (
            FewshotConfig.from_dict(
                self.fewshot_config or {},
                fewshot_split=self.fewshot_split,
                process_docs=self.process_docs,
                fewshot_delimiter=self.fewshot_delimiter,
                target_delimiter=self.target_delimiter,
                gen_prefix=self.gen_prefix,
                doc_to_text=self.doc_to_text,
                doc_to_choice=self.doc_to_choice,
                doc_to_target=self.doc_to_target,
            )
            if (isinstance(self.fewshot_config, dict) or self.fewshot_config is None)
            else self.fewshot_config
        )

    def __getitem__(self, item):
        return getattr(self, item)

    def __setitem__(self, item, value):
        return setattr(self, item, value)

    def to_dict(self, keep_callable: bool = False) -> dict:
        """Dumps the current config as a dictionary object, as a printable format.

        null fields will not be printed.
        Used for dumping results alongside full task configuration

        :return: dict
            A printable dictionary version of the TaskConfig object.

        # TODO: should any default value in the TaskConfig not be printed?
        """
        cfg_dict = asdict(self)
        # remove values that are `None`
        for k, v in list(cfg_dict.items()):
            if v is None:
                cfg_dict.pop(k)
            elif k == "metric_list":
                for metric_dict in v:
                    for metric_key, metric_value in metric_dict.items():
                        if callable(metric_value):
                            metric_dict[metric_key] = self.serialize_function(
                                metric_value, keep_callable=keep_callable
                            )
                cfg_dict[k] = v
            elif callable(v):
                cfg_dict[k] = self.serialize_function(v, keep_callable=keep_callable)
        return cfg_dict

    def serialize_function(
        self, value: Callable | str, keep_callable=False
    ) -> Callable | str:
        """Serializes a given function or string.

        If 'keep_callable' is True, the original callable is returned.
        Otherwise, attempts to return the source code of the callable using 'getsource'.
        """
        if keep_callable:
            return value
        else:
            try:
                return getsource(value)  # type:ignore[invalid-argument-type]
            except (TypeError, OSError):
                return str(value)
