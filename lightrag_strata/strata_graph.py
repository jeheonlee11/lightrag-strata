"""LightRAG knowledge-graph storage backed by a StrataDB graph-only collection.

With this and ``StrataVectorDBStorage``, LightRAG keeps both its vectors and
its entity/relation graph in one embedded StrataDB database. Semantics match
LightRAG's default NetworkX backend: an undirected simple graph where nodes
and edges carry string-keyed property dicts.

Edges are stored once, in the direction they were first written, under the
relation ``"related"``; every lookup checks both directions.
"""

from __future__ import annotations

import os
from collections import deque
from dataclasses import dataclass
from typing import final

from lightrag.base import BaseGraphStorage
from lightrag.types import KnowledgeGraph, KnowledgeGraphEdge, KnowledgeGraphNode
from lightrag.utils import logger

from .strata_storage import _database

GRAPH_STORAGE_NAME = "StrataGraphStorage"
REL = "related"


def register_graph() -> None:
    from lightrag import kg

    impls = kg.STORAGE_IMPLEMENTATIONS["GRAPH_STORAGE"]["implementations"]
    if GRAPH_STORAGE_NAME not in impls:
        impls.append(GRAPH_STORAGE_NAME)
    kg.STORAGES[GRAPH_STORAGE_NAME] = __name__
    kg.STORAGE_ENV_REQUIREMENTS.setdefault(GRAPH_STORAGE_NAME, [])


@final
@dataclass
class StrataGraphStorage(BaseGraphStorage):
    def __post_init__(self):
        base = os.path.join(self.global_config["working_dir"], self.workspace or "", "stratadb")
        self._db_path = os.path.normpath(base)
        self._collection_name = f"graph_{self.namespace}"
        self._col = None

    async def initialize(self):
        if self._col is None:
            self._col = _database(self._db_path).create_graph(self._collection_name)

    async def finalize(self):
        if self._col is not None:
            self._col.close()
            self._col = None

    # ---------------------------------------------------------------- helpers
    @property
    def _g(self):
        return self._col.graph

    def _edge_key(self, a: str, b: str) -> tuple[str, str] | None:
        """The stored (src, dst) for the undirected edge a-b, if any."""
        if self._g.get_edge(a, b, REL) is not None:
            return a, b
        if self._g.get_edge(b, a, REL) is not None:
            return b, a
        return None

    def _degree(self, node: str) -> int:
        return len(self._g.neighbor_ids(node))

    # ------------------------------------------------------------- read API
    async def has_node(self, node_id: str) -> bool:
        return self._g.has_node(node_id)

    async def has_edge(self, source_node_id: str, target_node_id: str) -> bool:
        return self._edge_key(source_node_id, target_node_id) is not None

    async def node_degree(self, node_id: str) -> int:
        return self._degree(node_id) if self._g.has_node(node_id) else 0

    async def edge_degree(self, src_id: str, tgt_id: str) -> int:
        return (await self.node_degree(src_id)) + (await self.node_degree(tgt_id))

    async def get_node(self, node_id: str) -> dict[str, str] | None:
        props = self._g.get_node(node_id)
        return dict(props) if props is not None else None

    async def get_edge(self, source_node_id: str, target_node_id: str) -> dict[str, str] | None:
        key = self._edge_key(source_node_id, target_node_id)
        if key is None:
            return None
        _, props = self._g.get_edge(key[0], key[1], REL)
        return dict(props)

    async def get_node_edges(self, source_node_id: str) -> list[tuple[str, str]] | None:
        if not self._g.has_node(source_node_id):
            return None
        return [(source_node_id, nbr) for nbr in sorted(self._g.neighbor_ids(source_node_id))]

    async def get_all_labels(self) -> list[str]:
        return sorted(self._g.node_ids())

    async def get_all_nodes(self) -> list[dict]:
        return [{**props, "id": nid} for nid in self._g.node_ids() if (props := self._g.get_node(nid)) is not None]

    async def get_all_edges(self) -> list[dict]:
        return [{**props, "source": src, "target": dst} for src, rel, dst, _, props in self._g.all_edges() if rel == REL]

    async def get_popular_labels(self, limit: int = 300) -> list[str]:
        nodes = sorted(self._g.node_ids(), key=lambda n: (-self._degree(n), n))
        return nodes[:limit]

    async def search_labels(self, query: str, limit: int = 50) -> list[str]:
        q = query.lower().strip()
        if not q:
            return []
        matches = []
        for node in self._g.node_ids():
            low = node.lower()
            if q not in low:
                continue
            if low == q:
                score = 1000
            elif low.startswith(q):
                score = 500
            else:
                score = 100 - len(node)
                if f" {q}" in low or f"_{q}" in low:
                    score += 50
            matches.append((node, score))
        matches.sort(key=lambda m: (-m[1], m[0]))
        return [m[0] for m in matches[:limit]]

    async def get_knowledge_graph(self, node_label: str, max_depth: int = 3, max_nodes: int = None) -> KnowledgeGraph:
        cap = self.global_config.get("max_graph_nodes", 1000)
        max_nodes = cap if max_nodes is None else min(max_nodes, cap)
        result = KnowledgeGraph()

        if node_label == "*":
            ordered = sorted(self._g.node_ids(), key=lambda n: (-self._degree(n), n))
            if len(ordered) > max_nodes:
                result.is_truncated = True
            chosen = ordered[:max_nodes]
        else:
            if not self._g.has_node(node_label):
                logger.warning(f"[{self.workspace}] Node {node_label} not found in the graph")
                return KnowledgeGraph()
            chosen, truncated = self._bfs(node_label, max_depth, max_nodes)
            result.is_truncated = truncated

        selected = set(chosen)
        for node in chosen:
            props = dict(self._g.get_node(node) or {})
            result.nodes.append(KnowledgeGraphNode(id=node, labels=[node], properties=props))
        seen = set()
        for src, rel, dst, _, props in self._g.all_edges():
            if rel != REL or src not in selected or dst not in selected:
                continue
            a, b = sorted((src, dst))
            edge_id = f"{a}-{b}"
            if edge_id in seen:
                continue
            seen.add(edge_id)
            result.edges.append(KnowledgeGraphEdge(id=edge_id, type="DIRECTED", source=a, target=b,
                                                   properties=dict(props)))
        return result

    def _bfs(self, start: str, max_depth: int, max_nodes: int) -> tuple[list[str], bool]:
        """Level-by-level BFS, highest-degree nodes first within a level (as NetworkX backend)."""
        visited: set[str] = set()
        order: list[str] = []
        queue = deque([(start, 0)])
        unexplored_beyond_depth = False
        cut_short = False
        while queue and len(order) < max_nodes:
            depth = queue[0][1]
            level = []
            while queue and queue[0][1] == depth:
                level.append(queue.popleft()[0])
            level.sort(key=lambda n: (-self._degree(n), n))
            for idx, node in enumerate(level):
                if node in visited:
                    continue
                visited.add(node)
                order.append(node)
                unvisited = [n for n in self._g.neighbor_ids(node) if n not in visited]
                if depth < max_depth:
                    queue.extend((n, depth + 1) for n in unvisited)
                elif unvisited:
                    unexplored_beyond_depth = True
                if len(order) >= max_nodes:
                    cut_short = any(n not in visited for n in level[idx + 1:])
                    break
        pending = any(n not in visited for n, _ in queue)
        truncated = len(order) >= max_nodes and (pending or cut_short or unexplored_beyond_depth)
        return order, truncated

    # ------------------------------------------------------------ write API
    async def upsert_node(self, node_id: str, node_data: dict[str, str]) -> None:
        existing = self._g.get_node(node_id) or {}
        self._col.put_nodes([{"id": node_id, "props": {**existing, **node_data}}])

    async def upsert_edge(self, source_node_id: str, target_node_id: str, edge_data: dict[str, str]) -> None:
        for node in (source_node_id, target_node_id):
            if not self._g.has_node(node):
                self._col.put_nodes([{"id": node, "props": {}}])  # like networkx add_edge
        key = self._edge_key(source_node_id, target_node_id) or (source_node_id, target_node_id)
        current = self._g.get_edge(key[0], key[1], REL)
        props = {**(current[1] if current else {}), **edge_data}
        try:
            weight = float(props.get("weight", 1.0))
        except (TypeError, ValueError):
            weight = 1.0
        self._col.link([{"src": key[0], "dst": key[1], "rel": REL, "weight": weight, "props": props}])

    async def delete_node(self, node_id: str) -> None:
        self._col.delete([node_id])

    async def remove_nodes(self, nodes: list[str]):
        if nodes:
            self._col.delete(nodes)

    async def remove_edges(self, edges: list[tuple[str, str]]):
        to_remove = []
        for a, b in edges:
            to_remove.append({"src": a, "dst": b, "rel": REL})
            to_remove.append({"src": b, "dst": a, "rel": REL})
        if to_remove:
            self._col.unlink(to_remove)

    # ------------------------------------------------------------ lifecycle
    async def index_done_callback(self) -> bool:
        self._col.checkpoint()
        return True

    async def drop(self) -> dict[str, str]:
        try:
            db = _database(self._db_path)
            db.drop_collection(self._collection_name)
            self._col = db.create_graph(self._collection_name)
            return {"status": "success", "message": "data dropped"}
        except Exception as e:  # noqa: BLE001 - LightRAG expects a status dict
            return {"status": "error", "message": str(e)}
