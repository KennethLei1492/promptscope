"""Pluggable LLM backends for the second (escalation) pass.

All backends implement one method: `complete(system: str, user: str) -> str`.
Only the standard library is used, so the package runs anywhere Python 3.9+ does.

Backends
--------
  anthropic   Claude via the Messages API           (ANTHROPIC_API_KEY)
  openai      OpenAI / Codex via Chat Completions    (OPENAI_API_KEY, OPENAI_BASE_URL)
              Also works with any OpenAI-compatible server (Ollama, vLLM, LM Studio,
              OpenRouter, Azure w/ base_url) by setting OPENAI_BASE_URL.
  subprocess  Any local CLI agent: the prompt is piped to stdin, stdout is the reply.
              e.g.  --backend subprocess --cmd "codex exec -"
                    --backend subprocess --cmd "claude -p"
                    --backend subprocess --cmd "ollama run llama3.1"
  mock        Deterministic stand-in for tests / dry runs (no network).

Select with `get_backend(name, model=..., **kw)` or the CLI `--backend` flag.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import subprocess
import urllib.error
import urllib.request
from typing import Optional, Protocol


class LLMBackend(Protocol):
    name: str
    def complete(self, system: str, user: str) -> str: ...


def _post_json(url: str, headers: dict, payload: dict, timeout: float) -> dict:
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", **headers}, method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {e.code} from {url}: {body[:500]}") from e


# --------------------------------------------------------------------------- Anthropic
class AnthropicBackend:
    name = "anthropic"

    def __init__(self, model: str = "claude-sonnet-5", api_key: Optional[str] = None,
                 base_url: Optional[str] = None, max_tokens: int = 1024, timeout: float = 60.0):
        self.model = model
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise RuntimeError("ANTHROPIC_API_KEY not set")
        self.base_url = (base_url or os.environ.get("ANTHROPIC_BASE_URL") or "https://api.anthropic.com").rstrip("/")
        self.max_tokens = max_tokens
        self.timeout = timeout

    def complete(self, system: str, user: str) -> str:
        data = _post_json(
            f"{self.base_url}/v1/messages",
            {"x-api-key": self.api_key, "anthropic-version": "2023-06-01"},
            {"model": self.model, "max_tokens": self.max_tokens, "system": system,
             "messages": [{"role": "user", "content": user}]},
            self.timeout,
        )
        return "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text")


# --------------------------------------------------------------------------- OpenAI / Codex / compatible
class OpenAIBackend:
    name = "openai"

    def __init__(self, model: str = "gpt-4o-mini", api_key: Optional[str] = None,
                 base_url: Optional[str] = None, max_tokens: int = 1024, timeout: float = 60.0,
                 temperature: float = 0.0):
        self.model = model
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY") or "not-needed"  # local servers ignore it
        self.base_url = (base_url or os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1").rstrip("/")
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.temperature = temperature

    def complete(self, system: str, user: str) -> str:
        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        }
        # Newer OpenAI models reject `max_tokens`/`temperature`; compatible servers accept them.
        if not self.model.startswith(("gpt-5", "o1", "o3", "o4")):
            payload["max_tokens"] = self.max_tokens
            payload["temperature"] = self.temperature
        data = _post_json(
            f"{self.base_url}/chat/completions",
            {"Authorization": f"Bearer {self.api_key}"},
            payload, self.timeout,
        )
        return data["choices"][0]["message"]["content"] or ""


# --------------------------------------------------------------------------- local CLI (Codex CLI, Claude Code, ollama, ...)
class SubprocessBackend:
    """Pipe `<system>\\n\\n<user>` to a command's stdin and read stdout.

    Works with any agent CLI that can take a prompt on stdin, e.g.
        codex exec -            (OpenAI Codex CLI)
        claude -p               (Claude Code)
        ollama run <model>      (local models)
    """

    name = "subprocess"

    def __init__(self, cmd: str, timeout: float = 300.0, cwd: Optional[str] = None):
        if not cmd:
            raise RuntimeError("subprocess backend needs --cmd")
        self.cmd = cmd
        self.timeout = timeout
        self.cwd = cwd

    def complete(self, system: str, user: str) -> str:
        argv = shlex.split(self.cmd, posix=(os.name != "nt"))
        # On Windows, npm-installed CLIs (codex, claude) are .cmd shims that
        # CreateProcess can only launch by their resolved path.
        resolved = shutil.which(argv[0])
        if resolved:
            argv[0] = resolved
        proc = subprocess.run(
            argv, input=f"{system}\n\n{user}", capture_output=True,
            text=True, timeout=self.timeout, cwd=self.cwd,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"{self.cmd!r} exited {proc.returncode}: {proc.stderr[:500]}")
        return proc.stdout


# --------------------------------------------------------------------------- mock
class MockBackend:
    """Deterministic backend for tests. Returns a JSON verdict that rewrites the prompt
    into a professional template so the whole loop can be exercised offline."""

    name = "mock"

    def __init__(self, decision: str = "rewrite", **_):
        self.decision = decision
        self.calls = 0

    def complete(self, system: str, user: str) -> str:
        self.calls += 1
        # last line of the user message is the prompt (see scope_model.build_user_message)
        prompt = user.rsplit("PROMPT:\n", 1)[-1].strip().rstrip(".")
        prompt = re.sub(r"\bu\b", "you", prompt)
        prompt = re.sub(r"\bur\b", "your", prompt)
        if prompt and prompt[0].islower():
            prompt = prompt[0].upper() + prompt[1:]
        rewritten = (
            f"{prompt}. "
            "Preserve functional equivalence with the original design and keep the module interface unchanged. "
            "Return a unified diff and a short explanation."
        )
        return json.dumps({
            "decision": self.decision,
            "score": 0.85,
            "reasons": ["mock backend"],
            "rewritten": rewritten if self.decision == "rewrite" else None,
        })


# --------------------------------------------------------------------------- factory
BACKENDS = {
    "anthropic": AnthropicBackend,
    "openai": OpenAIBackend,
    "codex": OpenAIBackend,        # alias: same wire protocol, pick a codex-capable model
    "subprocess": SubprocessBackend,
    "mock": MockBackend,
}


def get_backend(name: str = "mock", **kwargs) -> LLMBackend:
    try:
        cls = BACKENDS[name]
    except KeyError:
        raise ValueError(f"unknown backend {name!r}; choose from {sorted(BACKENDS)}")
    # drop kwargs the class does not take (e.g. `cmd` for HTTP backends, `model` for subprocess)
    import inspect
    accepted = inspect.signature(cls.__init__).parameters
    if "_" in accepted or any(p.kind == inspect.Parameter.VAR_KEYWORD for p in accepted.values()):
        return cls(**kwargs)
    return cls(**{k: v for k, v in kwargs.items() if k in accepted and v is not None})
