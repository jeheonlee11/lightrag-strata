"""Fast unit tests: no LLM calls, no model downloads."""

from pathlib import Path

import pytest

from lightrag_strata.__main__ import parse_args
from lightrag_strata.core import collect_documents, read_document
from lightrag_strata.llm import format_prompt, make_llm


def test_collect_documents_filters_and_recurses(tmp_path: Path):
    (tmp_path / "a.md").write_text("# A")
    (tmp_path / "skip.png").write_bytes(b"\x89PNG")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "b.txt").write_text("B")
    found = [p.name for p in collect_documents([tmp_path])]
    assert found == ["a.md", "b.txt"]


def test_collect_documents_accepts_files_and_dedupes(tmp_path: Path):
    f = tmp_path / "a.md"
    f.write_text("x")
    assert collect_documents([f, tmp_path]) == [f]


def test_read_document_text(tmp_path: Path):
    f = tmp_path / "n.txt"
    f.write_text("hello")
    assert read_document(f) == "hello"


def test_format_prompt_without_history():
    assert format_prompt("q") == "q"


def test_format_prompt_folds_history():
    out = format_prompt("now?", [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "yo"}])
    assert "user: hi" in out and "assistant: yo" in out and out.endswith("now?")


def test_make_llm_rejects_unknown_backend():
    with pytest.raises(ValueError):
        make_llm("nope")


def test_cli_parses_compare():
    args = parse_args(["--workdir", "/tmp/x", "compare", "what?", "--modes", "local,global"])
    assert args.cmd == "compare" and args.modes == "local,global" and args.llm == "claude-cli"
    assert args.graph_store == "strata"
