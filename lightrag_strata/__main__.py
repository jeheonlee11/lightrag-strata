"""CLI: python -m lightrag_strata {ingest,query,compare,stats}."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import time
from pathlib import Path

from .core import MODES, build_rag, ingest, query
from .embed import DEFAULT_EMBED_MODEL


def parse_args(argv=None):
    parser = argparse.ArgumentParser(prog="lightrag_strata", description="LightRAG graph RAG on StrataDB")
    parser.add_argument("--workdir", default="./rag_storage", help="LightRAG state + StrataDB storage dir")
    parser.add_argument("--llm", default="claude-cli", choices=["claude-cli", "openai", "ollama"])
    parser.add_argument("--model", default=None, help="LLM model (default depends on --llm)")
    parser.add_argument("--embed-model", default=DEFAULT_EMBED_MODEL)
    parser.add_argument("--graph-store", default="strata", choices=["strata", "networkx"],
                        help="where the entity/relation graph lives (default: StrataDB)")
    parser.add_argument("--max-async", type=int, default=4, help="concurrent LLM calls")
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("ingest", help="extract entities/relations from files or folders")
    p.add_argument("paths", nargs="+")

    p = sub.add_parser("query", help="ask a question")
    p.add_argument("question")
    p.add_argument("--mode", default="mix", choices=MODES)
    p.add_argument("--context-only", action="store_true", help="print retrieved context, skip generation")

    p = sub.add_parser("compare", help="answer one question in several modes side by side")
    p.add_argument("question")
    p.add_argument("--modes", default="naive,local,global", help="comma-separated modes")

    sub.add_parser("stats", help="show graph size")
    return parser.parse_args(argv)


def graph_stats(workdir: Path) -> dict:
    workdir = workdir.expanduser()
    strata_graph = workdir / "stratadb" / "graph_chunk_entity_relation"
    if (strata_graph / "config.json").exists():
        from stratadb import Collection

        col = Collection(strata_graph)
        try:
            g = col.graph
            degrees = [(n, len(g.neighbor_ids(n))) for n in g.node_ids()]
            top = sorted(degrees, key=lambda kv: (-kv[1], kv[0]))[:10]
            return {"entities": g.node_count, "relations": len(g), "most_connected": top}
        finally:
            col.close()

    import networkx as nx

    graphml = next(workdir.rglob("graph_chunk_entity_relation.graphml"), None)
    if graphml is None:
        return {"entities": 0, "relations": 0}
    g = nx.read_graphml(graphml)
    top = sorted(g.degree, key=lambda kv: kv[1], reverse=True)[:10]
    return {"entities": g.number_of_nodes(), "relations": g.number_of_edges(), "most_connected": top}


async def run(args) -> None:
    workdir = Path(args.workdir)
    if args.cmd == "stats":
        print(json.dumps(graph_stats(workdir), indent=2))
        return

    rag = await build_rag(
        workdir,
        llm_backend=args.llm,
        llm_model=args.model,
        embed_model=args.embed_model,
        max_async=args.max_async,
        graph_store=args.graph_store,
    )
    try:
        if args.cmd == "ingest":
            start = time.time()
            docs = await ingest(rag, args.paths)
            print(f"Ingested {len(docs)} document(s) in {time.time() - start:.0f}s")
        elif args.cmd == "query":
            print(await query(rag, args.question, args.mode, args.context_only))
        elif args.cmd == "compare":
            for mode in [m.strip() for m in args.modes.split(",") if m.strip()]:
                print(f"\n===== {mode} =====")
                print(await query(rag, args.question, mode))
    finally:
        await rag.finalize_storages()
    if args.cmd == "ingest":
        # After finalize: a StrataDB graph can only be opened once at a time.
        print(json.dumps(graph_stats(workdir), indent=2))


def main(argv=None) -> None:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)
    if not args.verbose:
        logging.getLogger("lightrag").setLevel(logging.WARNING)
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
