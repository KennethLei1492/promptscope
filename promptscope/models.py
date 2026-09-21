"""Core data types shared by every stage of the pipeline."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Dict, List, Optional


class Decision(str, Enum):
    """Outcome of a scope analysis pass."""

    ACCEPT = "accept"      # in scope as written
    REWRITE = "rewrite"    # in scope after edits (rewritten text is attached)
    ESCALATE = "escalate"  # rules could not decide -> send to LLM pass
    REJECT = "reject"      # out of scope and not salvageable


@dataclass
class Verdict:
    """Result produced by a single analysis pass (rules or LLM)."""

    decision: Decision
    score: float                      # 0..1 confidence that the prompt is in scope
    reasons: List[str] = field(default_factory=list)
    rewritten: Optional[str] = None   # populated when decision == REWRITE
    source: str = "rules"             # "rules" | "llm:<backend>" | "merge"

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["decision"] = self.decision.value
        return d


@dataclass
class PromptItem:
    """A unit of work flowing through the pipeline.

    `text` is always the *current* version of the prompt. The original is kept in
    `original` so the final output can show the diff.
    """

    text: str
    original: str = ""
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    metadata: Dict[str, Any] = field(default_factory=dict)
    history: List[Verdict] = field(default_factory=list)
    iteration: int = 0                # how many times it has gone around the loop
    final: Optional[Verdict] = None   # set when the item leaves the pipeline
    created_at: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        if not self.original:
            self.original = self.text

    @property
    def last(self) -> Optional[Verdict]:
        return self.history[-1] if self.history else None

    def apply(self, verdict: Verdict) -> None:
        """Record a verdict and, if it carries a rewrite, adopt the rewritten text."""
        self.history.append(verdict)
        if verdict.decision == Decision.REWRITE and verdict.rewritten:
            self.text = verdict.rewritten.strip()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "original": self.original,
            "text": self.text,
            "iteration": self.iteration,
            "final": self.final.to_dict() if self.final else None,
            "history": [v.to_dict() for v in self.history],
            "metadata": self.metadata,
        }
