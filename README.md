# lightrag-strata

Graph RAG over your own documents with [LightRAG](https://github.com/HKUDS/LightRAG), stored in
[StrataDB](https://github.com/jeheonlee11-create/stratadb): an embedded, graph-native vector
database, so LightRAG's vectors **and** its entity/relation graph live in one place.

Drop in a folder of Markdown, text or PDF files. LightRAG extracts entities and relationships
into a knowledge graph and answers questions at two levels:

- **local** mode: finds the entities a question names and walks to their neighbours. Good for specific questions.
- **global** mode: retrieves relationships across the whole graph. Good for "what are the main themes" questions.
- **hybrid / mix** combine the two (mix also adds plain vector search over chunks); **naive** is plain vector RAG, for comparison.

It runs fully locally with no API key:

| Piece | Default | Alternatives |
|---|---|---|
| Vector store | **StrataDB** (embedded, under the work dir) | |
| Graph store | **StrataDB** (same database) | LightRAG's NetworkX GraphML file via `--graph-store networkx` |
| Embeddings | `BAAI/bge-small-en-v1.5` via fastembed (ONNX, CPU, ~64 MB) | other fastembed models via `--embed-model` |
| LLM | Claude through the `claude` CLI (uses your Claude Pro/Max login) | `--llm openai` (`OPENAI_API_KEY`) or `--llm ollama` |

## Setup

Python 3.10–3.12.

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install git+https://github.com/jeheonlee11-create/stratadb@v0.7.0
```

For the default LLM backend, install [Claude Code](https://claude.com/claude-code) and sign in once
(`claude`). The first run downloads the embedding model to `~/.cache/fastembed`.

## Usage

```bash
# build the graph from the bundled docs (StrataDB's own documentation) or any folder
python -m lightrag_strata ingest sample_docs

# ask questions
python -m lightrag_strata query "How does crash recovery work?" --mode local
python -m lightrag_strata query "What are the main components and how do they fit together?" --mode global

# same question, several retrieval modes side by side
python -m lightrag_strata compare "What happens when a segment is sealed?" --modes naive,local,global

# only show what was retrieved, skip the answer
python -m lightrag_strata query "What is a write-ahead log?" --context-only

# graph size and most connected entities
python -m lightrag_strata stats
```

Global options go before the subcommand: `--workdir` (default `./rag_storage`), `--llm`, `--model`,
`--embed-model`, `--graph-store`, `--max-async`, `-v`.

Re-running `ingest` on the same files is cheap: LightRAG skips documents it has already processed
and caches LLM responses.

## How StrataDB is used

Everything LightRAG stores besides its JSON key-value files lives in `<workdir>/stratadb/`: three
vector collections (entities, relations, chunks) and a graph-only collection holding the
entity/relation graph. Relationship vectors are linked to their endpoint entities with StrataDB
edges, so deleting an entity's relations is a neighbour lookup instead of a scan.

`tests/test_strata_graph.py` drives the StrataDB graph backend and LightRAG's NetworkX backend
through the same random operation histories and checks that every read method agrees, including
BFS subgraphs and truncation flags.

## Layout

```
lightrag_strata/
  core.py            build LightRAG on StrataDB, ingest files, query
  strata_storage.py  LightRAG vector storage on StrataDB
  strata_graph.py    LightRAG graph storage on StrataDB
  llm.py             claude-cli / openai / ollama backends
  embed.py           local fastembed embeddings
  __main__.py        CLI
sample_docs/         StrataDB's documentation, as a demo corpus
tests/               unit tests (no network, no LLM)
```

Run tests with `python -m pytest -q`.

## Notes

- With the `claude-cli` backend every LLM call is a `claude -p` subprocess, so ingest is slower
  than an HTTP API (roughly a few seconds per extraction call). Raise `--max-async` to run more in
  parallel, or switch to `--llm openai` for large corpora.

## License

MIT.
