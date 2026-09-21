"""promptscope - hybrid prompt-scope analysis/rewrite model with a parallel, looped pipeline.

Built for the CSAW AI Hardware Attack (AHA!) Challenge workflow, but the rubric and
LLM backend are pluggable so the same program runs against Claude, OpenAI/Codex,
any OpenAI-compatible server, or a local CLI tool.
"""

from .models import PromptItem, Verdict, Decision
from .rules import RuleEngine, load_rubric
from .scope_model import PromptScopeModel
from .pipeline import Pipeline, Stage, build_default_pipeline
from .llm import get_backend

__all__ = [
    "PromptItem",
    "Verdict",
    "Decision",
    "RuleEngine",
    "load_rubric",
    "PromptScopeModel",
    "Pipeline",
    "Stage",
    "build_default_pipeline",
    "get_backend",
]

__version__ = "0.1.0"
