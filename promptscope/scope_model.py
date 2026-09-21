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

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


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
        m = _JSON_RE.search(raw or "")
        if not m:
            return Verdict(Decision.REJECT, 0.0, ["LLM returned no JSON"], source=f"llm:{self.backend.name}")
        try:
            data = json.loads(m.group(0))
            decision = Decision(str(data.get("decision", "reject")).lower())
        except (ValueError, TypeError) as exc:
            return Verdict(Decision.REJECT, 0.0, [f"LLM JSON invalid: {exc}"], source=f"llm:{self.backend.name}")
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
            # REWRITE -> loop: re-validate the new text with the rules
        item.final = Verdict(Decision.REJECT, item.last.score if item.last else 0.0,
                             [f"exceeded {max_iterations} iterations"], source="merge")
        return item

    def analyze_many(self, texts: Iterable[str], max_iterations: int = 3) -> List[PromptItem]:
        with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
            return list(ex.map(lambda t: self.analyze(t, max_iterations), texts))
