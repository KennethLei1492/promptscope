# promptscope

> **New here and not a programmer?** Read [GETTING_STARTED.md](GETTING_STARTED.md) — a plain-language guide to downloading and running the tool.

A hybrid model that **analyzes and edits prompts so they fit a professional engineering scope**, wrapped in a disassembled, parallel, loop-capable pipeline. Built for the CSAW AI Hardware Attack (AHA!) Challenge workflow, but the scope rubric and the LLM backend are both pluggable, so the same program drives Claude, OpenAI/Codex, any OpenAI-compatible server, or a local CLI agent.

Standard library only. Python 3.9+. No install required to run.

```bash
python -m promptscope "synthesize picorv32.v with yosys and report LUT count"
python -m promptscope -f examples/prompts.txt --backend openai --model gpt-4o-mini
python tests/test_promptscope.py
```

## The order of operations (disassembled)

The scoping logic is broken into independent **stages**. Each stage owns its own input queue and its own pool of worker threads, so stages run concurrently with one another *and* each stage processes many prompts at once. A stage returns the *name* of the next stage, which is how the loop-back edge is expressed without any special cases.

```
 ingest --> normalize --> screen --> [ review ] --> validate --> emit
                            ^  (rules)    (LLM)         |
                            |                           |
                            +----------- loop ----------+
                                    (bounded by max_iterations)
```

| Stage | Kind | What it does |
|-------|------|--------------|
| `ingest`    | 1 worker  | Wrap raw input in a `PromptItem`, record metadata. |
| `normalize` | parallel  | Encoding/whitespace cleanup, strip conversational openers, length cap. |
| `screen`    | parallel  | **Rule pass.** Fast deterministic scope check → `accept` / `rewrite` / `escalate` / `reject`. |
| `review`    | bounded   | **LLM pass.** Only for escalated items; concurrency capped by a semaphore. |
| `validate`  | parallel  | Re-check the (rewritten) text against the rules; decide accept, loop again, or give up. |
| `emit`      | 1 worker  | Freeze the final verdict, hand the item to an optional sink. |

**Hybrid by design:** cheap, deterministic rules handle the clear cases (an off-topic prompt is rejected in microseconds and never costs an API call); the LLM is spent only on genuinely ambiguous, in-domain prompts that need a real rewrite. The loop lets a rewrite be re-validated and, if still under-specified, sent back around up to `max_iterations` times.

## Two ways to run it

**Staged pipeline** — parallel across stages, best for throughput and for plugging into an existing loop:

```python
from promptscope import PromptScopeModel, build_default_pipeline, get_backend, PromptItem

model = PromptScopeModel(backend=get_backend("anthropic", model="claude-sonnet-4-5"))
pipe  = build_default_pipeline(model, workers=4, llm_workers=2, max_iterations=3)
results = pipe.run([PromptItem(text=p) for p in prompts])
for item in results:
    print(item.final.decision, "->", item.text)
```

**Direct model** — one call, thread-pooled internally, best for embedding as a single step:

```python
model = PromptScopeModel(backend=get_backend("openai", model="gpt-4o-mini"))
items = model.analyze_many(prompts)      # parallel
one   = model.analyze("debug the alu")   # synchronous, runs the full loop
```

The pipeline is a plain stage graph (`promptscope/pipeline.py`), so to drop it into a framework you already have, either call `pipe.submit(item, stage="screen")` from your loop, or import the stage functions from `promptscope/stages.py` and wire them into your own graph.

## Backends — run it on other AI models

Select with `--backend` (CLI) or `get_backend(name, ...)` (API). Every backend implements one method, `complete(system, user) -> str`.

| `--backend` | Talks to | Configure with |
|-------------|----------|----------------|
| `anthropic` | Claude Messages API | `ANTHROPIC_API_KEY`, `--model` |
| `openai`    | OpenAI Chat Completions | `OPENAI_API_KEY`, `--model` |
| `codex`     | OpenAI (Codex-capable model) | same as `openai`, pick a code model |
| `openai` + `--base-url` | **Any OpenAI-compatible server** — Ollama, vLLM, LM Studio, OpenRouter, Azure | `--base-url http://localhost:11434/v1` |
| `subprocess`| **Any local CLI agent** — pipes the prompt to stdin, reads stdout | `--cmd "codex exec -"` · `--cmd "claude -p"` · `--cmd "ollama run llama3.1"` |
| `mock`      | Nothing (offline, deterministic) | for tests and dry runs |

Examples:

```bash
# OpenAI Codex CLI as the reviewer
python -m promptscope -f examples/prompts.txt --backend subprocess --cmd "codex exec -"

# a local model through Ollama's OpenAI-compatible endpoint
python -m promptscope -f examples/prompts.txt --backend openai \
    --base-url http://localhost:11434/v1 --model llama3.1

# Claude, writing full results to JSON Lines
python -m promptscope -f examples/prompts.jsonl --backend anthropic \
    --model claude-sonnet-4-5 --jsonl out.jsonl
```

To add a backend, write a class with a `complete(system, user)` method and register it in `BACKENDS` in `promptscope/llm.py`.

## Re-targeting the scope

The professional-scope definition lives entirely in `promptscope/rubric_csaw.json` — no code change needed to point it at a different project:

- `domain_terms` — vocabulary buckets (hardware, toolchain, competition, engineering verbs) that score how strongly a prompt reads as this project's work.
- `forbidden` — regex patterns that hard-reject (real/deployed targets, malware, rule-circumvention, off-topic).
- `rewrites` — mechanical cleanups applied before scoring (slang, urgency filler, vague success criteria, terminology).
- `quality_checks` — the machine-checkable proxy for "professional": does the prompt name a **target** (file/module), state a **deliverable**, and state **constraints**?
- `scoring` — weights, blend of domain-vs-structure, and the accept/escalate thresholds.
- `llm_scope_statement` — the scope description handed to the LLM in the system prompt.

Point the CLI at your own file with `--rubric path/to/rubric.json`, or pass `rubric_path=` to `PromptScopeModel`.

### How a verdict is reached (rule pass)

1. Any `forbidden` match → **reject** immediately.
2. No domain signal at all → **reject** (not this project's work; no LLM call).
3. Score = `domain_blend · domain_score + structure_blend · structure_score`, after applying `rewrites`.
4. `score ≥ accept_threshold` and all quality checks pass → **accept** (or **rewrite** if cleanups changed the text).
5. `score ≥ escalate_threshold` → **escalate** to the LLM (carrying the mechanical cleanup forward).
6. Otherwise → **reject**.

## Layout

```
promptscope/
  models.py       PromptItem, Verdict, Decision
  rules.py        RuleEngine (deterministic first pass)
  llm.py          backends: anthropic, openai/codex, subprocess, mock
  scope_model.py  PromptScopeModel: analyze / analyze_many / llm_review
  pipeline.py     Pipeline + Stage (thread-pool stage graph, loop-back)
  stages.py       the six default stage functions
  cli.py          command line
  rubric_csaw.json  the CSAW scope definition
examples/prompts.txt
tests/test_promptscope.py
```
