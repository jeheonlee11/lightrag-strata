"""LLM backends for LightRAG.

LightRAG calls ``llm_model_func(prompt, system_prompt=None, history_messages=[],
keyword_extraction=False, **kwargs)`` and expects a string back. Three backends:

- ``claude-cli``: shells out to the ``claude`` CLI in print mode, so a Claude
  Pro/Max login works with no API key.
- ``openai``: any OpenAI-compatible endpoint (``OPENAI_API_KEY``, optional
  ``OPENAI_BASE_URL``).
- ``ollama``: a local Ollama server.
"""

from __future__ import annotations

import asyncio
import os
import shutil
import tempfile

DEFAULT_MODELS = {
    "claude-cli": "haiku",
    "openai": "gpt-4o-mini",
    "ollama": "qwen2.5:7b",
}


def format_prompt(prompt: str, history_messages: list[dict] | None = None) -> str:
    """Fold chat history into one prompt for single-shot backends."""
    if not history_messages:
        return prompt
    lines = ["Conversation so far:"]
    for msg in history_messages:
        lines.append(f"{msg.get('role', 'user')}: {msg.get('content', '')}")
    lines += ["", "Current request:", prompt]
    return "\n".join(lines)


def make_claude_cli_llm(model: str, timeout: float = 300.0):
    exe = shutil.which("claude")
    if not exe:
        raise RuntimeError("`claude` CLI not found on PATH; install Claude Code or pick another --llm")
    # Run outside any project so no CLAUDE.md / settings leak into extraction prompts.
    workdir = tempfile.mkdtemp(prefix="lightrag-claude-")

    async def llm(prompt, system_prompt=None, history_messages=None, keyword_extraction=False, **kwargs):
        args = [
            exe, "-p",
            "--model", model,
            "--output-format", "text",
            "--tools", "",
            "--strict-mcp-config",
            "--setting-sources", "",
            "--no-session-persistence",
            "--system-prompt", system_prompt or "You are a precise assistant. Follow the output format exactly.",
        ]
        proc = await asyncio.create_subprocess_exec(
            *args,
            cwd=workdir,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            out, err = await asyncio.wait_for(
                proc.communicate(format_prompt(prompt, history_messages).encode()), timeout
            )
        except asyncio.TimeoutError:
            proc.kill()
            raise RuntimeError(f"claude CLI timed out after {timeout}s")
        if proc.returncode != 0:
            raise RuntimeError(f"claude CLI failed ({proc.returncode}): {err.decode()[:500]}")
        return out.decode().strip()

    return llm


def make_openai_llm(model: str):
    from lightrag.llm.openai import openai_complete_if_cache

    async def llm(prompt, system_prompt=None, history_messages=None, keyword_extraction=False, **kwargs):
        return await openai_complete_if_cache(
            model,
            prompt,
            system_prompt=system_prompt,
            history_messages=history_messages or [],
            api_key=os.environ.get("OPENAI_API_KEY"),
            base_url=os.environ.get("OPENAI_BASE_URL"),
            **kwargs,
        )

    return llm


def make_ollama_llm(model: str):
    from lightrag.llm.ollama import ollama_model_complete

    async def llm(prompt, system_prompt=None, history_messages=None, keyword_extraction=False, **kwargs):
        kwargs.setdefault("host", os.environ.get("OLLAMA_HOST", "http://localhost:11434"))
        return await ollama_model_complete(
            prompt,
            system_prompt=system_prompt,
            history_messages=history_messages or [],
            keyword_extraction=keyword_extraction,
            **kwargs,
        )

    return llm


def make_llm(backend: str, model: str | None = None):
    if backend not in DEFAULT_MODELS:
        raise ValueError(f"unknown LLM backend: {backend}")
    model = model or DEFAULT_MODELS[backend]
    if backend == "claude-cli":
        return make_claude_cli_llm(model), model
    if backend == "openai":
        return make_openai_llm(model), model
    if backend == "ollama":
        return make_ollama_llm(model), model
    raise ValueError(f"unknown LLM backend: {backend}")
