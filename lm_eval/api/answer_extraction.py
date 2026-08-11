"""Reading a model response two ways, and keeping the better score.

MERA prompts ask the model to end with ``Ответ: <answer>``, but a response is
free text: the model may skip the marker, emit it several times, or put the
answer before it and something else after. Committing to one reading throws
away the other, and there is no reading that is right for every response.

So the answer is never extracted in a filter. The filter chain
(``remove_whitespace_and_nones`` then ``take_first``) only normalises
whitespace and unwraps the batch dimension; the text that reaches
``process_results`` — and the text written to the submission — is the model's
own. The metric then scores both readings and keeps the higher one:

    from lm_eval.api.answer_extraction import answer_candidates, best_score

    cands = answer_candidates(results[0])
    exact = best_score(lambda c: squad_metrics.compute_exact(gold, c), cands)

Scoring the truncated reading alone would zero every response that broke the
format; scoring the full one alone would zero every response that reasoned out
loud before answering. The maximum is right in both cases, and it cannot
inflate a score: a wrong answer stays wrong under either reading.
"""

from typing import Callable, List, Optional, Sequence, TypeVar

ANSWER_MARKER = "Ответ:"

T = TypeVar("T")


def answer_candidates(
    response: Optional[str], marker: str = ANSWER_MARKER
) -> List[str]:
    """The readings of ``response`` worth scoring, de-duplicated.

    Returns the whole response and, when ``marker`` occurs in it, the part
    after its **last** occurrence — so a model that restates "Ответ:" while
    thinking out loud is read by its final answer. The two collapse to one
    entry when the marker is absent or contributes nothing, which is what keeps
    an expensive metric (the LLM judge) from being charged twice for a response
    that has only one reading.

    Whitespace around the marker is not significant: the split is on ``Ответ:``
    itself, so ``Ответ: X`` and ``Ответ:X`` both yield ``X``.
    """
    full = (response or "").strip()
    if not full or marker not in full:
        return [full]
    tail = full.rsplit(marker, 1)[-1].strip()
    if not tail or tail == full:
        return [full]
    return [full, tail]


def best_score(
    score_fn: Callable[[str], float], candidates: Sequence[str]
) -> float:
    """Highest score ``score_fn`` gives any candidate; ``0.0`` for none."""
    return max((score_fn(c) for c in candidates), default=0.0)
