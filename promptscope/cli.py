"""Command-line entry point.

    python -m promptscope "insert a bug in the alu"                 # one prompt, mock backend
    python -m promptscope -f prompts.txt --backend openai --model gpt-4o-mini
    python -m promptscope -f prompts.jsonl --backend anthropic --model claude-sonnet-4-5 --jsonl out.jsonl
    python -m promptscope -f prompts.txt --backend subprocess --cmd "codex exec -"
    python -m promptscope -f prompts.txt --backend openai --base-url http://localhost:11434/v1 --model llama3.1

Input formats: one prompt per line (.txt), or JSON Lines with a "text" key (.jsonl).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import List

from .llm import BACKENDS, get_backend
from .models import PromptItem
from .pipeline import build_default_pipeline
from .scope_model import PromptScopeModel


def _read_prompts(path: str) -> List[PromptItem]:
    p = Path(path)
    items: List[PromptItem] = []
    if p.suffix == ".jsonl":
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip():
                d = json.loads(line)
                items.append(PromptItem(text=d["text"], metadata={k: v for k, v in d.items() if k != "text"}))
    else:
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip() and not line.lstrip().startswith("#"):
                items.append(PromptItem(text=line.strip()))
    return items


def _print_item(item: PromptItem, verbose: bool) -> None:
    f = item.final
    tag = f.decision.value.upper() if f else "?"
    print(f"[{tag:8}] score={f.score if f else 0:.2f}  id={item.id}  iters={item.iteration}")
    print(f"   in : {item.original}")
    if f and f.decision.value == "rewrite":
        print(f"   out: {item.text}")
    if verbose and f:
        for r in f.reasons:
            print(f"      - {r}")
        for v in item.history:
            print(f"      · {v.source:14} {v.decision.value:8} {v.score:.2f}")
    print()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="promptscope", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("prompt", nargs="?", help="a single prompt to analyze")
    ap.add_argument("-f", "--file", help="file of prompts (.txt one per line, or .jsonl with 'text')")
    ap.add_argument("--backend", default="mock", choices=sorted(BACKENDS))
    ap.add_argument("--model", help="model name for HTTP backends")
    ap.add_argument("--base-url", help="override API base URL (OpenAI-compatible servers, proxies)")
    ap.add_argument("--cmd", help="command for the subprocess backend, e.g. 'codex exec -'")
    ap.add_argument("--rubric", help="path to a rubric JSON (default: bundled CSAW rubric)")
    ap.add_argument("--workers", type=int, default=4, help="threads per CPU-bound stage")
    ap.add_argument("--llm-workers", type=int, default=2, help="max concurrent LLM calls")
    ap.add_argument("--max-iterations", type=int, default=3, help="loop budget per prompt")
    ap.add_argument("--jsonl", help="write full results as JSON Lines to this path")
    ap.add_argument("--no-pipeline", action="store_true",
                    help="use PromptScopeModel.analyze_many instead of the staged pipeline")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.WARNING,
                        format="%(asctime)s %(threadName)s %(levelname)s %(message)s")

    if not args.prompt and not args.file:
        ap.error("give a prompt or -f FILE")
    items = _read_prompts(args.file) if args.file else [PromptItem(text=args.prompt)]

    backend = get_backend(args.backend, model=args.model, base_url=args.base_url, cmd=args.cmd)
    model = PromptScopeModel(backend=backend, rubric_path=args.rubric, max_workers=args.workers)

    t0 = time.time()
    if args.no_pipeline:
        results = model.analyze_many([i.text for i in items], max_iterations=args.max_iterations)
    else:
        pipe = build_default_pipeline(model, workers=args.workers, llm_workers=args.llm_workers,
                                      max_iterations=args.max_iterations)
        results = pipe.run(items)
    elapsed = time.time() - t0

    order = {it.id: n for n, it in enumerate(items)}
    results.sort(key=lambda it: order.get(it.id, 0))
    for it in results:
        _print_item(it, args.verbose)

    counts = {}
    for it in results:
        k = it.final.decision.value if it.final else "none"
        counts[k] = counts.get(k, 0) + 1
    print(f"{len(results)} prompts in {elapsed:.2f}s  backend={backend.name}  " +
          "  ".join(f"{k}={v}" for k, v in sorted(counts.items())), file=sys.stderr)

    if args.jsonl:
        with open(args.jsonl, "w", encoding="utf-8") as f:
            for it in results:
                f.write(json.dumps(it.to_dict()) + "\n")
        print(f"wrote {args.jsonl}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
