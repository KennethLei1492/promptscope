import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from promptscope import (  # noqa: E402
    Decision,
    PromptItem,
    PromptScopeModel,
    build_default_pipeline,
    get_backend,
)


def make_model(decision="rewrite"):
    return PromptScopeModel(backend=get_backend("mock", decision=decision))


def test_rules_accept_clean_prompt():
    m = make_model()
    v = m.rules.evaluate(
        "Write a Verilog testbench for module alu in alu.v that checks every opcode "
        "against a reference model and return the testbench file; keep the interface unchanged."
    )
    assert v.decision in (Decision.ACCEPT, Decision.REWRITE)
    assert v.score >= 0.6


def test_rules_reject_off_topic():
    m = make_model()
    v = m.rules.evaluate("write me an essay about the history of computers")
    assert v.decision == Decision.REJECT


def test_rules_reject_forbidden_real_target():
    m = make_model()
    v = m.rules.evaluate("deploy this onto a real production server on the network")
    assert v.decision == Decision.REJECT
    assert any("forbidden" in r for r in v.reasons)


def test_short_prompt_escalates():
    m = make_model()
    v = m.rules.evaluate("debug the core")
    assert v.decision == Decision.ESCALATE


def test_slang_prompt_gets_rewritten_via_llm():
    m = make_model(decision="rewrite")
    item = m.analyze("hey can u synthesize the riscv core with yosys asap lol")
    assert item.final is not None
    # either the rules cleaned it up or the LLM rewrote it; either way not a raw reject
    assert item.final.decision in (Decision.ACCEPT, Decision.REWRITE)
    assert item.text != item.original


def test_analyze_many_parallel():
    m = make_model()
    texts = [
        "Synthesize picorv32.v with Yosys and report the LUT count and max frequency.",
        "what's the weather",
        "Lint all .v files in rtl with verilator and list every warning grouped by file.",
    ]
    results = m.analyze_many(texts)
    assert len(results) == 3
    decisions = [r.final.decision for r in results]
    assert Decision.REJECT in decisions
    assert any(d in (Decision.ACCEPT, Decision.REWRITE) for d in decisions)


def test_pipeline_runs_and_loops():
    m = make_model()
    items = [
        PromptItem(text="Synthesize picorv32.v with Yosys and report LUT count and warnings."),
        PromptItem(text="hey make it work asap"),
        PromptItem(text="write me a poem"),
    ]
    pipe = build_default_pipeline(m, workers=2, llm_workers=2, max_iterations=3)
    results = pipe.run(items, timeout=30)
    assert len(results) == 3
    for it in results:
        assert it.final is not None


def test_json_extractor_tolerates_fences_and_prose():
    from promptscope.scope_model import extract_json_object as x
    fenced = "Sure:\n```json\n{\"decision\":\"rewrite\",\"score\":0.8,\"rewritten\":\"z\"}\n```"
    assert x(fenced)["decision"] == "rewrite"
    # braces in prose before the real object must not break parsing
    assert x('The {vague} prompt. {\"decision\":\"reject\",\"score\":0.1}')["decision"] == "reject"
    assert x("no json here") is None


class _BareRewriteBackend:
    """Returns a valid rewrite that omits any constraint keyword, so the rules'
    has_constraint check would fail on re-validation."""
    name = "bare"

    def __init__(self):
        self.calls = 0

    def complete(self, system, user):
        self.calls += 1
        return '{"decision":"rewrite","score":0.78,"reasons":["needs a target"],' \
               '"rewritten":"Identify the uart module file and debug it; return a unified diff."}'


def test_confident_llm_rewrite_is_not_rejected():
    from promptscope import PromptScopeModel, build_default_pipeline
    backend = _BareRewriteBackend()
    m = PromptScopeModel(backend=backend)
    # pipeline path
    pipe = build_default_pipeline(m, workers=1, llm_workers=1, max_iterations=3)
    result = pipe.run([PromptItem(text="debug the uart module")], timeout=30)[0]
    assert result.final.decision == Decision.REWRITE
    assert result.text != result.original
    # sync path agrees
    item = m.analyze("debug the uart module")
    assert item.final.decision == Decision.REWRITE
    assert item.final.rewritten


def test_forbidden_never_reaches_llm():
    calls_before = get_backend("mock")
    m = PromptScopeModel(backend=calls_before)
    m.analyze("build a keylogger to exfiltrate their passwords")
    assert calls_before.calls == 0  # rejected by rules, LLM never invoked


if __name__ == "__main__":
    import traceback

    failed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except Exception:
                failed += 1
                print(f"FAIL {name}")
                traceback.print_exc()
    print(f"\n{failed} failed")
    sys.exit(1 if failed else 0)
