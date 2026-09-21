"""Stage functions for the default pipeline. Each is `(item, ctx) -> Route`.

Keeping them as free functions (with small factories where a model is needed)
means another program can import just the stages it wants and wire its own graph.
"""

from __future__ import annotations

import logging
import re

from .models import Decision, PromptItem, Verdict
from .pipeline import DONE, PipelineContext, Route

log = logging.getLogger("promptscope.stages")

_WS = re.compile(r"[ \t]+")
_BLANK_LINES = re.compile(r"\n{3,}")
_CHATTER = re.compile(
    r"^(hey|hi|hello|yo|please|pls|can you|could you|i need you to|i want you to)[,\s]+",
    re.IGNORECASE,
)


# 1. ingest ----------------------------------------------------------------
def ingest(item: PromptItem, ctx: PipelineContext) -> Route:
    item.metadata.setdefault("ingested", True)
    item.metadata.setdefault("orig_len", len(item.text))
    return None


# 2. normalize -------------------------------------------------------------
def normalize(item: PromptItem, ctx: PipelineContext) -> Route:
    text = item.text.replace("\r\n", "\n").replace(" ", " ")
    text = _WS.sub(" ", text)
    text = _BLANK_LINES.sub("\n\n", text).strip()
    # strip one leading conversational opener; keeps the imperative core
    text = _CHATTER.sub("", text, count=1)
    if text and text[0].islower():
        text = text[0].upper() + text[1:]
    max_len = int(ctx.extras.get("max_len", 6000))
    if len(text) > max_len:
        text = text[:max_len].rsplit(" ", 1)[0] + " ..."
        item.metadata["truncated"] = True
    item.text = text
    return None


# 3. screen (rules) --------------------------------------------------------
def make_screen(model):
    def screen(item: PromptItem, ctx: PipelineContext) -> Route:
        verdict = model.rules.evaluate(item.text)
        item.apply(verdict)
        if verdict.decision == Decision.ESCALATE:
            return "review"
        if verdict.decision == Decision.REJECT:
            return "emit"
        # ACCEPT or REWRITE -> validate (rewrite gets re-checked there)
        return "validate"
    return screen


# 4. review (LLM) ----------------------------------------------------------
def make_review(model):
    def review(item: PromptItem, ctx: PipelineContext) -> Route:
        sem = ctx.llm_semaphore
        if sem:
            sem.acquire()
        try:
            verdict = model.llm_review(item)
        finally:
            if sem:
                sem.release()
        item.apply(verdict)
        if verdict.decision == Decision.REJECT:
            return "emit"
        return "validate"
    return review


# 5. validate (+ loop decision) -------------------------------------------
def make_validate(model):
    escalate_t = float(model.rules.scoring.get("escalate_threshold", 0.3))

    def validate(item: PromptItem, ctx: PipelineContext) -> Route:
        last = item.last
        check = model.rules.evaluate(item.text)
        changed = item.text != item.original

        # 1. The current text passes the rules cleanly -> done.
        if check.decision == Decision.ACCEPT:
            item.final = Verdict(
                Decision.REWRITE if changed else Decision.ACCEPT,
                score=max(check.score, last.score if last else 0.0),
                reasons=(last.reasons if last else []) + ["validated by rules"],
                rewritten=item.text if changed else None,
                source="merge",
            )
            return "emit"

        # 2. The model returned a rewrite it was confident in -> trust it instead of
        #    re-imposing the rules' keyword checks on the model's finished work and
        #    looping into a reject. The rules gate raw prompts, not model rewrites.
        if (last is not None and last.source.startswith("llm:")
                and last.decision == Decision.REWRITE and changed
                and last.score >= escalate_t):
            item.final = Verdict(
                Decision.REWRITE,
                score=max(check.score, last.score),
                reasons=last.reasons + ["accepted model rewrite"],
                rewritten=item.text,
                source="merge",
            )
            return "emit"

        # 3. Otherwise keep improving within the loop budget, then give up.
        item.iteration += 1
        if item.iteration < ctx.max_iterations:
            if check.decision == Decision.REWRITE:
                item.apply(check)
                return "validate"          # rules produced another fix; re-check
            return "review"                # ask the LLM for a real rewrite
        item.final = Verdict(
            Decision.REJECT,
            score=check.score,
            reasons=check.reasons + [f"exceeded {ctx.max_iterations} loop iterations"],
            source="merge",
        )
        return "emit"
    return validate


# 6. emit ------------------------------------------------------------------
def emit(item: PromptItem, ctx: PipelineContext) -> Route:
    if item.final is None:
        item.final = item.last
    sink = ctx.extras.get("sink")
    if callable(sink):
        sink(item)
    return DONE
