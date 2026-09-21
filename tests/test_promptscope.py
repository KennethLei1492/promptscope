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
