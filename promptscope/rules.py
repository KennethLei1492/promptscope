"""Deterministic rule layer (first pass of the hybrid model).

Zero external dependencies; everything is driven by a JSON rubric so the scope can
be re-targeted to a different project by swapping the file.

Evaluation order:
  1. forbidden patterns   -> REJECT immediately
  2. domain scoring       -> how strongly the text reads as this project's work
  3. rewrite rules        -> mechanical cleanups (slang, vague criteria, terminology)
  4. quality checks       -> target / deliverable / constraints present?
  5. decision:
       score >= accept_threshold and no missing quality  -> ACCEPT
       score >= accept_threshold and rewrites applied    -> REWRITE
       score >= escalate_threshold                        -> ESCALATE (LLM decides)
       otherwise                                          -> REJECT
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .models import Decision, Verdict

DEFAULT_RUBRIC = Path(__file__).with_name("rubric_csaw.json")


def load_rubric(path: Optional[str | Path] = None) -> Dict[str, Any]:
    with open(path or DEFAULT_RUBRIC, "r", encoding="utf-8") as f:
        return json.load(f)


class RuleEngine:
    def __init__(self, rubric: Optional[Dict[str, Any]] = None):
        self.rubric = rubric or load_rubric()
        terms = self.rubric["domain_terms"]
        self._term_res = {
            k: re.compile(r"\b(" + "|".join(re.escape(t) for t in v) + r")\b", re.IGNORECASE)
            for k, v in terms.items()
        }
        self._forbidden = [
            (re.compile(f["pattern"], re.IGNORECASE), f["reason"], f.get("action", "reject"))
            for f in self.rubric.get("forbidden", [])
        ]
        self._rewrites = [
            (re.compile(r["pattern"]), r["replacement"], r["reason"])
            for r in self.rubric.get("rewrites", [])
        ]
        self._quality = [
            (q["id"], re.compile(q["pattern"]), q["missing_reason"], float(q.get("severity", 0.1)))
            for q in self.rubric.get("quality_checks", [])
        ]
        self.scoring = self.rubric["scoring"]

    # ------------------------------------------------------------------ pieces
    def domain_score(self, text: str) -> Tuple[float, Dict[str, int]]:
        s = self.scoring
        hits = {k: len(set(m.lower() for m in rx.findall(text))) for k, rx in self._term_res.items()}
        cap = float(s.get("saturate_at", 2))
        # saturating credit: `saturate_at` distinct terms in a bucket = full marks
        sat = lambda n: min(n, cap) / cap
        score = (
            s["hardware_weight"] * sat(hits.get("hardware", 0))
            + s["toolchain_weight"] * sat(hits.get("toolchain", 0))
            + s["competition_weight"] * sat(hits.get("competition", 0))
            + s["verb_weight"] * sat(hits.get("engineering_verbs", 0))
        )
        return round(min(score, 1.0), 3), hits

    def apply_rewrites(self, text: str) -> Tuple[str, List[str]]:
        reasons = []
        for rx, repl, reason in self._rewrites:
            new = rx.sub(repl, text)
            if new != text:
                reasons.append(f"rewrite: {reason}")
                text = new
        text = re.sub(r"[ \t]{2,}", " ", text).strip()
        text = re.sub(r"\s+([,.;:])", r"\1", text)
        if text and text[0].islower():
            text = text[0].upper() + text[1:]
        return text, reasons

    def quality(self, text: str) -> Tuple[float, List[str]]:
        """Return (structure_score in 0..1, list of missing-check reasons).

        structure_score is the fraction of quality checks the text passes, which is
        the machine-checkable proxy for "professional": names a target, states a
        deliverable, states constraints.
        """
        if not self._quality:
            return 1.0, []
        passed, missing = 0, []
        for qid, rx, reason, _sev in self._quality:
            if rx.search(text):
                passed += 1
            else:
                missing.append(f"{qid}: {reason}")
        return passed / len(self._quality), missing

    # ------------------------------------------------------------------ verdict
    def evaluate(self, text: str) -> Verdict:
        reasons: List[str] = []

        for rx, reason, action in self._forbidden:
            if rx.search(text):
                return Verdict(Decision.REJECT, 0.0, [f"forbidden: {reason}"], source="rules")

        base, hits = self.domain_score(text)
        reasons.append("domain hits: " + ", ".join(f"{k}={v}" for k, v in hits.items()))

        # No project-domain signal at all -> not this team's work; don't waste an LLM call.
        if base == 0.0:
            reasons.append("no engineering/domain content for this project's scope")
            return Verdict(Decision.REJECT, 0.0, reasons, source="rules")

        if len(text.split()) < int(self.scoring.get("min_words", 6)):
            return Verdict(Decision.ESCALATE, round(base, 3),
                           reasons + ["too short to judge; escalating"], source="rules")

        rewritten, rw_reasons = self.apply_rewrites(text)
        reasons += rw_reasons

        structure, missing = self.quality(rewritten)
        dw = float(self.scoring.get("domain_blend", 0.65))
        sw = float(self.scoring.get("structure_blend", 0.35))
        score = round(min(1.0, dw * base + sw * structure), 3)
        reasons.append(f"structure {structure:.2f} (domain {base:.2f})")

        accept_t = self.scoring["accept_threshold"]
        escalate_t = self.scoring["escalate_threshold"]

        if score >= accept_t and not missing:
            if rewritten != text:
                return Verdict(Decision.REWRITE, score, reasons, rewritten=rewritten, source="rules")
            return Verdict(Decision.ACCEPT, score, reasons, source="rules")

        # In-domain but under-specified, or borderline -> let the LLM rewrite it.
        reasons += missing
        if score >= escalate_t:
            v = Verdict(Decision.ESCALATE, score, reasons, source="rules")
            if rewritten != text:
                v.rewritten = rewritten  # carry the mechanical cleanup forward
            return v

        reasons.append("insufficient engineering/domain content for this project's scope")
        return Verdict(Decision.REJECT, score, reasons, source="rules")
