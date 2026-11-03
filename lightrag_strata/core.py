"""Build a LightRAG instance on StrataDB and load documents into it."""

from __future__ import annotations

from pathlib import Path

from lightrag import LightRAG, QueryParam

from .embed import DEFAULT_EMBED_MODEL, make_embedding_func
from .llm import make_llm
from .strata_graph import register_graph
from .strata_storage import register

SUPPORTED_SUFFIXES = {".txt", ".md", ".markdown", ".rst", ".pdf"}
MODES = ("local", "global", "hybrid", "mix", "naive")
GRAPH_STORES = {"strata": "StrataGraphStorage", "networkx": "NetworkXStorage"}


async def build_rag(
    workdir: str | Path,
    llm_backend: str = "claude-cli",
    llm_model: str | None = None,
    embed_model: str = DEFAULT_EMBED_MODEL,
    max_async: int = 4,
    graph_store: str = "strata",
) -> LightRAG:
    """Create and initialize LightRAG with StrataDB storage.

    Entity, relation and chunk vectors live in an embedded StrataDB under
    ``<workdir>/stratadb``. The entity/relation graph lives there too
    (``graph_store="strata"``), or in LightRAG's NetworkX GraphML file
    (``graph_store="networkx"``).
    """
    if graph_store not in GRAPH_STORES:
        raise ValueError(f"graph_store must be one of {tuple(GRAPH_STORES)}")
    workdir = Path(workdir).expanduser()
    workdir.mkdir(parents=True, exist_ok=True)
    register()
    register_graph()

    llm_func, model_name = make_llm(llm_backend, llm_model)
    rag = LightRAG(
        working_dir=str(workdir),
        llm_model_func=llm_func,
        llm_model_name=model_name,
        llm_model_max_async=max_async,
        embedding_func=make_embedding_func(embed_model),
        vector_storage="StrataVectorDBStorage",
        graph_storage=GRAPH_STORES[graph_store],
        cosine_better_than_threshold=0.2,
    )
    await rag.initialize_storages()
    return rag


def read_document(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        from pypdf import PdfReader

        return "\n\n".join(page.extract_text() or "" for page in PdfReader(path).pages)
    return path.read_text(encoding="utf-8", errors="replace")


def collect_documents(paths: list[str | Path]) -> list[Path]:
    """Expand files and folders into a sorted list of supported documents."""
    found: set[Path] = set()
    for raw in paths:
        p = Path(raw).expanduser()
        if p.is_dir():
            found.update(f for f in p.rglob("*") if f.is_file() and f.suffix.lower() in SUPPORTED_SUFFIXES)
        elif p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES:
            found.add(p)
    return sorted(found)


async def ingest(rag: LightRAG, paths: list[str | Path]) -> list[Path]:
    """Insert documents. LightRAG dedupes by content hash, so re-running is cheap."""
    docs = collect_documents(paths)
    texts, names = [], []
    for doc in docs:
        text = read_document(doc).strip()
        if text:
            texts.append(text)
            names.append(doc.name)
    if texts:
        await rag.ainsert(texts, file_paths=names)
    return docs


async def query(rag: LightRAG, question: str, mode: str = "mix", context_only: bool = False) -> str:
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    param = QueryParam(mode=mode, only_need_context=context_only, enable_rerank=False)
    return await rag.aquery(question, param=param)
