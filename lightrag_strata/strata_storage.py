"""LightRAG vector storage backed by StrataDB (embedded, graph-native).

Each LightRAG namespace (entities, relationships, chunks) becomes a StrataDB
collection under ``<working_dir>/stratadb``. Relationship records are also
linked to their endpoint entities with StrataDB edges, so deleting an
entity's relations is a graph lookup instead of a full scan.

Call ``register()`` once before constructing LightRAG, then pass
``vector_storage="StrataVectorDBStorage"``.
"""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass
from typing import Any, final

import numpy as np
from lightrag.base import BaseVectorStorage
from lightrag.utils import compute_mdhash_id, logger
from stratadb import Database

STORAGE_NAME = "StrataVectorDBStorage"
CREATED_AT = "__created_at__"

_databases: dict[str, Database] = {}
_db_lock = threading.Lock()


def register() -> None:
    """Make LightRAG's storage factory aware of this backend."""
    from lightrag import kg

    impls = kg.STORAGE_IMPLEMENTATIONS["VECTOR_STORAGE"]["implementations"]
    if STORAGE_NAME not in impls:
        impls.append(STORAGE_NAME)
    kg.STORAGES[STORAGE_NAME] = __name__
    kg.STORAGE_ENV_REQUIREMENTS.setdefault(STORAGE_NAME, [])


def _database(path: str) -> Database:
    with _db_lock:
        if path not in _databases:
            _databases[path] = Database(path)
        return _databases[path]


def _entity_node(name: str) -> str:
    return f"entity:{name}"


@final
@dataclass
class StrataVectorDBStorage(BaseVectorStorage):
    def __post_init__(self):
        self._validate_embedding_func()
        kwargs = self.global_config.get("vector_db_storage_cls_kwargs", {})
        self.cosine_better_than_threshold = kwargs.get(
            "cosine_better_than_threshold", self.cosine_better_than_threshold
        )
        base = os.path.join(self.global_config["working_dir"], self.workspace or "", "stratadb")
        self._db_path = os.path.normpath(base)
        suffix = self._generate_collection_suffix()
        self._collection_name = f"{self.namespace}_{suffix}" if suffix else self.namespace
        self._batch = int(self.global_config.get("embedding_batch_num", 32))
        self._col = None

    async def initialize(self):
        if self._col is None:
            db = _database(self._db_path)
            self._col = db.create_collection(
                self._collection_name, dim=self.embedding_func.embedding_dim, metric="cosine",
                seal_threshold=2048,
            )

    async def finalize(self):
        if self._col is not None:
            self._col.close()
            self._col = None

    # ---------------------------------------------------------------- helpers
    async def _embed(self, texts: list[str]) -> np.ndarray:
        batches = [texts[i:i + self._batch] for i in range(0, len(texts), self._batch)]
        parts = [await self.embedding_func(batch) for batch in batches]
        return np.concatenate([np.asarray(p, dtype=np.float32) for p in parts])

    @staticmethod
    def _format(rid: str, meta: dict) -> dict[str, Any]:
        return {**{k: v for k, v in meta.items() if k != CREATED_AT}, "id": rid, "created_at": meta.get(CREATED_AT)}

    # ------------------------------------------------------------- write API
    async def upsert(self, data: dict[str, dict[str, Any]]) -> None:
        if not data:
            return
        ids = list(data)
        now = int(time.time())
        metas = [
            {**{k: v for k, v in data[rid].items() if k in self.meta_fields}, CREATED_AT: now}
            for rid in ids
        ]
        vectors = await self._embed([data[rid]["content"] for rid in ids])
        self._col.upsert(ids, vectors, metas)

        edges = []
        for rid in ids:
            rec = data[rid]
            if "src_id" in rec and "tgt_id" in rec:
                edges.append({"src": _entity_node(rec["src_id"]), "dst": rid, "rel": "endpoint"})
                edges.append({"src": _entity_node(rec["tgt_id"]), "dst": rid, "rel": "endpoint"})
        if edges:
            self._col.link(edges)

    async def delete(self, ids: list[str]):
        if ids:
            self._col.delete(ids)

    async def delete_entity(self, entity_name: str) -> None:
        self._col.delete([compute_mdhash_id(entity_name, prefix="ent-")])

    async def delete_entity_relation(self, entity_name: str) -> None:
        node = _entity_node(entity_name)
        linked = {n["id"] for n in self._col.neighbors(node, rel="endpoint")}
        # Records written before linking existed (or by another tool) are caught by metadata.
        scanned = {
            r["id"] for r in self._col.scan({"$or": [{"src_id": entity_name}, {"tgt_id": entity_name}]})
        }
        targets = sorted(linked | scanned)
        self._col.delete(targets + [node])
        logger.debug(f"[{self.workspace}] deleted {len(targets)} relations of {entity_name}")

    # -------------------------------------------------------------- read API
    async def query(self, query: str, top_k: int, query_embedding: list[float] = None) -> list[dict[str, Any]]:
        if query_embedding is None:
            query_embedding = (await self.embedding_func([query], context="query"))[0]
        hits = self._col.search(query_embedding, k=top_k)
        return [
            {**self._format(h["id"], h["metadata"]), "distance": h["score"]}
            for h in hits
            if h["score"] > self.cosine_better_than_threshold
        ]

    async def get_by_id(self, id: str) -> dict[str, Any] | None:
        found = self._col.get([id])
        return self._format(found[0]["id"], found[0]["metadata"]) if found else None

    async def get_by_ids(self, ids: list[str]) -> list[dict[str, Any] | None]:
        found = {r["id"]: r for r in self._col.get(ids)}
        return [self._format(i, found[i]["metadata"]) if i in found else None for i in ids]

    async def get_vectors_by_ids(self, ids: list[str]) -> dict[str, list[float]]:
        return {r["id"]: r["vector"] for r in self._col.get(ids, include_vectors=True)}

    # ------------------------------------------------------------ lifecycle
    async def index_done_callback(self) -> bool:
        self._col.checkpoint()
        return True

    async def drop(self) -> dict[str, str]:
        try:
            db = _database(self._db_path)
            db.drop_collection(self._collection_name)
            self._col = db.create_collection(
                self._collection_name, dim=self.embedding_func.embedding_dim, metric="cosine",
                seal_threshold=2048,
            )
            return {"status": "success", "message": "data dropped"}
        except Exception as e:  # noqa: BLE001 - LightRAG expects a status dict, not an exception
            return {"status": "error", "message": str(e)}
