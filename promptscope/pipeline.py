"""Disassembled, parallel, loop-capable pipeline.

The order of operations is broken into independent *stages*. Each stage owns an
input queue and a pool of worker threads, so stages run concurrently with each
other *and* each stage processes several items at once. A stage function may
route an item to any other stage by name, which is how the loop-back edge
(validate -> screen) is expressed without special-casing.

Default order of operations for prompt scoping:

    ingest -> normalize -> screen -> [review] -> validate -> emit
                            ^                        |
                            +-------- loop-back ------+   (bounded by max_iterations)

    ingest     : wrap raw input in a PromptItem, attach metadata
    normalize  : whitespace/encoding cleanup, strip chatter, length caps
    screen     : fast deterministic rule pass (accept / rewrite / escalate / reject)
    review     : LLM pass, only for escalated items (bounded concurrency)
    validate   : re-run rules on the rewritten text; decide whether to loop
    emit       : freeze the final verdict and hand the item to the sink

Every stage is a plain callable `(PromptItem, PipelineContext) -> Route`, so the
graph can be re-wired, extended, or partially reused by another program.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional, Union

from .models import Decision, PromptItem, Verdict

log = logging.getLogger("promptscope.pipeline")

DONE = "__done__"
Route = Union[str, None]  # next stage name, DONE, or None (= next stage in declared order)


@dataclass
class PipelineContext:
    """Shared, read-mostly state handed to every stage call."""

    max_iterations: int = 3
    llm_semaphore: Optional[threading.Semaphore] = None
    extras: Dict[str, object] = field(default_factory=dict)


StageFn = Callable[[PromptItem, PipelineContext], Route]


@dataclass
class Stage:
    name: str
    fn: StageFn
    workers: int = 2
    queue: "queue.Queue[Optional[PromptItem]]" = field(default_factory=queue.Queue, repr=False)
    threads: List[threading.Thread] = field(default_factory=list, repr=False)


class Pipeline:
    """A stage graph executed by per-stage thread pools.

    Items enter at the first declared stage. A `_pending` counter (not
    `queue.join`) tracks completion because loop-back edges make per-queue
    joins unreliable.
    """

    def __init__(self, stages: Iterable[Stage], context: Optional[PipelineContext] = None):
        self.stages: List[Stage] = list(stages)
        if not self.stages:
            raise ValueError("pipeline needs at least one stage")
        self.by_name: Dict[str, Stage] = {s.name: s for s in self.stages}
        self.ctx = context or PipelineContext()
        self.results: List[PromptItem] = []
        self._results_lock = threading.Lock()
        self._pending = 0
        self._pending_cv = threading.Condition()
        self._running = False

    # ---------------------------------------------------------------- lifecycle
    def start(self) -> None:
        if self._running:
            return
        self._running = True
        for stage in self.stages:
            for i in range(stage.workers):
                t = threading.Thread(
                    target=self._worker, args=(stage,), name=f"{stage.name}-{i}", daemon=True
                )
                t.start()
                stage.threads.append(t)
        log.debug("pipeline started with stages %s", [s.name for s in self.stages])

    def stop(self) -> None:
        if not self._running:
            return
        self._running = False
        for stage in self.stages:
            for _ in stage.threads:
                stage.queue.put(None)  # sentinel
        for stage in self.stages:
            for t in stage.threads:
                t.join(timeout=5)
            stage.threads.clear()

    def __enter__(self) -> "Pipeline":
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop()

    # ---------------------------------------------------------------- submission
    def submit(self, item: PromptItem, stage: Optional[str] = None) -> None:
        target = self.by_name[stage] if stage else self.stages[0]
        with self._pending_cv:
            self._pending += 1
        target.queue.put(item)

    def submit_many(self, items: Iterable[PromptItem]) -> None:
        for it in items:
            self.submit(it)

    def wait(self, timeout: Optional[float] = None) -> bool:
        """Block until every submitted item has reached DONE. Returns False on timeout."""
        deadline = None if timeout is None else time.time() + timeout
        with self._pending_cv:
            while self._pending > 0:
                remaining = None if deadline is None else max(0.0, deadline - time.time())
                if remaining == 0.0:
                    return False
                self._pending_cv.wait(remaining)
        return True

    def run(self, items: Iterable[PromptItem], timeout: Optional[float] = None) -> List[PromptItem]:
        """Convenience: start, submit everything, wait, stop, return results."""
        with self:
            self.submit_many(items)
            self.wait(timeout)
        return list(self.results)

    # ---------------------------------------------------------------- internals
    def _next_in_order(self, stage: Stage) -> str:
        idx = self.stages.index(stage)
        return self.stages[idx + 1].name if idx + 1 < len(self.stages) else DONE

    def _finish(self, item: PromptItem) -> None:
        with self._results_lock:
            self.results.append(item)
        with self._pending_cv:
            self._pending -= 1
            if self._pending <= 0:
                self._pending_cv.notify_all()

    def _worker(self, stage: Stage) -> None:
        while True:
            item = stage.queue.get()
            if item is None:
                return
            try:
                route = stage.fn(item, self.ctx)
            except Exception as exc:  # a crashed stage must not hang the pipeline
                log.exception("stage %s failed on item %s", stage.name, item.id)
                item.apply(
                    Verdict(Decision.REJECT, 0.0, [f"stage '{stage.name}' error: {exc}"], source="pipeline")
                )
                route = DONE
            if route is None:
                route = self._next_in_order(stage)
            if route == DONE:
                if item.final is None:
                    item.final = item.last
                self._finish(item)
            else:
                self.by_name[route].queue.put(item)


# ---------------------------------------------------------------------------
# Default stage implementations
# ---------------------------------------------------------------------------

def build_default_pipeline(scope_model, *, workers: int = 4, llm_workers: int = 2,
                           max_iterations: int = 3) -> Pipeline:
    """Wire the standard six-stage graph around a PromptScopeModel.

    `scope_model` must expose `.rules` (RuleEngine) and `.llm_review(item)`.
    """
    from .stages import ingest, normalize, make_screen, make_review, make_validate, emit

    ctx = PipelineContext(
        max_iterations=max_iterations,
        llm_semaphore=threading.Semaphore(llm_workers),
    )
    stages = [
        Stage("ingest", ingest, workers=1),
        Stage("normalize", normalize, workers=workers),
        Stage("screen", make_screen(scope_model), workers=workers),
        Stage("review", make_review(scope_model), workers=llm_workers),
        Stage("validate", make_validate(scope_model), workers=workers),
        Stage("emit", emit, workers=1),
    ]
    return Pipeline(stages, ctx)
