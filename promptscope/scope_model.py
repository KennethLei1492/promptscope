"""PromptScopeModel: the hybrid analyzer/editor.

  analyze(text)      -> Verdict        rules first; LLM only when rules escalate
  analyze_many(texts)-> [Verdict]      thread-pool parallel version
  llm_review(item)   -> Verdict        used by the pipeline's `review` stage

The LLM is asked for strict JSON so the answer can be validated and merged with
the rule verdict; anything unparseable degrades to ESCALATE->REJECT rather than
silently accepting text.
"""

from __future__ import annotations

import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, Iterable, List, Optional

from .llm import LLMBackend, get_backend
from .models import Decision, PromptItem, Verdict
from .rules import RuleEngine, load_rubric

log = logging.getLogger("promptscope.model")

SYSTEM_TEMPLATE = """You are a prompt-scope reviewer for an engineering team.

{scope}

Task: decide whether the PROMPT below fits that scope and, if it is salvageable,
rewrite it into a professional engineering prompt that:
  - names the target (file / module / design) explicitly
  - states the concrete deliverable (patch, diff, file, testbench, report, list)
  - states constraints (functional equivalence, interface unchanged, tool flow, limits)
  - uses standard terminology, no slang, no urgency filler, imperative voice
  - keeps the author's technical intent; do NOT add new requirements they did not imply
  - is out of scope -> do not rewrite; reject with reasons

Respond with ONLY a JSON object, no prose, no code fences:
{{"decision": "accept" | "rewrite" | "reject",
  "score": <0.0-1.0 confidence the (rewritten) prompt is in scope>,
  "reasons": ["short reason", ...],
  "rewritten": "<rewritten prompt or null>"}}"""

_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)


def extract_json_object(raw: str) -> Optional[dict]:
    """Pull the verdict object out of an LLM reply that may wrap it in prose or a
    ```json fence, or precede it with other braces. Tries, in order:
      1. a fenced ```json {...}``` block,
      2. each balanced {...} span, newest first, that parses and has "decision",
      3. the whole string.
    Returns the parsed dict, or None if nothing usable is found.
    """
    if not raw:
        return None

    def _load(s: str) -> Optional[dict]:
        try:
            obj = json.loads(s)
            return obj if isinstance(obj, dict) else None
        except (ValueError, TypeError):
            return None

    m = _FENCE_RE.search(raw)
    if m:
        obj = _load(m.group(1))
        if obj is not None:
            return obj

    # Scan for balanced top-level {...} spans; prefer the last one containing "decision".
    spans, depth, start = [], 0, -1
    for i, ch in enumerate(raw):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and depth > 0:
            depth -= 1
            if depth == 0 and start >= 0:
                spans.append(raw[start:i + 1])
    for span in reversed(spans):
        obj = _load(span)
        if obj is not None and "decision" in obj:
            return obj
    for span in reversed(spans):
        obj = _load(span)
        if obj is not None:
            return obj
    return _load(raw.strip())


class PromptScopeModel:
    def __init__(self, backend: Optional[LLMBackend] = None,
                 rubric: Optional[Dict[str, Any]] = None, rubric_path: Optional[str] = None,
                 max_workers: int = 4):
        self.rubric = rubric or load_rubric(rubric_path)
        self.rules = RuleEngine(self.rubric)
        self.backend = backend or get_backend("mock")
        self.max_workers = max_workers
        self.system_prompt = SYSTEM_TEMPLATE.format(scope=self.rubric["llm_scope_statement"])

    # ------------------------------------------------------------------ LLM pass
    def build_user_message(self, item: PromptItem) -> str:
        notes = ""
        if item.last and item.last.reasons:
            notes = "Rule-layer notes:\n- " + "\n- ".join(item.last.reasons) + "\n\n"
        return f"{notes}PROMPT:\n{item.text}"

    def parse_llm(self, raw: str) -> Verdict:
        data = extract_json_object(raw)
        if data is None:
            return Verdict(Decision.REJECT, 0.0, ["LLM returned no parseable JSON verdict"],
                           source=f"llm:{self.backend.name}")
        try:
            decision = Decision(str(data.get("decision", "reject")).lower())
        except ValueError:
            return Verdict(Decision.REJECT, 0.0, [f"LLM gave unknown decision {data.get('decision')!r}"],
                           source=f"llm:{self.backend.name}")
        if decision == Decision.ESCALATE:      # the LLM may not punt back
            decision = Decision.REJECT
        score = float(data.get("score", 0.0))
        reasons = [str(r) for r in data.get("reasons", [])]
        rewritten = data.get("rewritten") or None
        if decision == Decision.REWRITE and not rewritten:
            decision = Decision.REJECT
            reasons.append("rewrite requested without text")
        return Verdict(decision, max(0.0, min(1.0, score)), reasons,
                       rewritten=rewritten, source=f"llm:{self.backend.name}")

    def llm_review(self, item: PromptItem) -> Verdict:
        # carry the rules' mechanical cleanup into the text the LLM sees
        if item.last and item.last.decision == Decision.ESCALATE and item.last.rewritten:
            item.text = item.last.rewritten
        try:
            raw = self.backend.complete(self.system_prompt, self.build_user_message(item))
        except Exception as exc:
            log.warning("LLM backend %s failed on %s: %s", self.backend.name, item.id, exc)
            return Verdict(Decision.REJECT, 0.0, [f"LLM backend error: {exc}"], source=f"llm:{self.backend.name}")
        return self.parse_llm(raw)

    # ------------------------------------------------------------------ one-shot API
    def analyze(self, text: str, max_iterations: int = 3) -> PromptItem:
        """Run the full rules -> (LLM) -> validate loop for one prompt, synchronously."""
        item = PromptItem(text=text)
        for _ in range(max_iterations):
            v = self.rules.evaluate(item.text)
            item.apply(v)
            if v.decision == Decision.REJECT:
                item.final = v
                return item
            if v.decision in (Decision.ACCEPT, Decision.REWRITE):
                check = self.rules.evaluate(item.text)
                if check.decision == Decision.ACCEPT or v.decision == Decision.ACCEPT:
                    changed = item.text != item.original
                    item.final = Verdict(Decision.REWRITE if changed else Decision.ACCEPT, check.score,
                                         v.reasons, rewritten=item.text if changed else None, source="merge")
                    return item
            lv = self.llm_review(item)
            item.apply(lv)
            item.iteration += 1
            if lv.decision == Decision.REJECT:
                item.final = lv
                return item
            if lv.decision == Decision.ACCEPT:
                item.final = lv
                return item
            if lv.decision == Decision.REWRITE:
                # Trust a confident model rewrite instead of looping the rules'
                # keyword checks over it until the iteration budget runs out.
                escalate_t = float(self.rules.scoring.get("escalate_threshold", 0.3))
                if lv.score >= escalate_t:
                    check = self.rules.evaluate(item.text)
                    changed = item.text != item.original
                    tag = ("validated by rules" if check.decision == Decision.ACCEPT
                           else "accepted model rewrite")
                    item.final = Verdict(
                        Decision.REWRITE if changed else Decision.ACCEPT,
                        max(check.score, lv.score), lv.reasons + [tag],
                        rewritten=item.text if changed else None, source="merge")
                    return item
                # low-confidence rewrite: loop for another pass
        item.final = Verdict(Decision.REJECT, item.last.score if item.last else 0.0,
                             [f"exceeded {max_iterations} iterations"], source="merge")
        return item

    def analyze_many(self, texts: Iterable[str], max_iterations: int = 3) -> List[PromptItem]:
        with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
            return list(ex.map(lambda t: self.analyze(t, max_iterations), texts))
