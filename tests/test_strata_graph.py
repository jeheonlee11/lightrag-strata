"""StrataGraphStorage must behave like LightRAG's default NetworkX backend.

Each test drives both backends through the same operations and compares
every read method.
"""

import asyncio
import random

import pytest
from lightrag.kg.networkx_impl import NetworkXStorage
from lightrag.kg.shared_storage import initialize_share_data

from lightrag_strata.strata_graph import StrataGraphStorage, register_graph


def run(coro):
    return asyncio.run(coro)


def make_pair(tmp_path):
    cfg_s = {"working_dir": str(tmp_path / "s"), "max_graph_nodes": 1000}
    cfg_n = {"working_dir": str(tmp_path / "n"), "max_graph_nodes": 1000}
    (tmp_path / "s").mkdir()
    (tmp_path / "n").mkdir()
    strata = StrataGraphStorage(namespace="chunk_entity_relation", workspace="", global_config=cfg_s,
                                embedding_func=None)
    nx = NetworkXStorage(namespace="chunk_entity_relation", workspace="", global_config=cfg_n,
                         embedding_func=None)
    return strata, nx


async def init(*stores):
    initialize_share_data()
    for s in stores:
        await s.initialize()


async def snapshot(store, nodes, pairs):
    out = {
        "labels": await store.get_all_labels(),
        "popular": await store.get_popular_labels(10),
        "search": await store.search_labels("n1", 10),
        "nodes": sorted((n["id"], tuple(sorted(n.items()))) for n in await store.get_all_nodes()),
        "edges": sorted(tuple(sorted((min(e["source"], e["target"]), max(e["source"], e["target"])))) +
                        (e.get("w"),) for e in await store.get_all_edges()),
    }
    for n in nodes:
        out[f"has {n}"] = await store.has_node(n)
        out[f"deg {n}"] = await store.node_degree(n)
        out[f"get {n}"] = await store.get_node(n)
        edges = await store.get_node_edges(n)
        out[f"edges {n}"] = None if edges is None else sorted(tuple(sorted(e)) for e in edges)
    for a, b in pairs:
        out[f"has {a}-{b}"] = await store.has_edge(a, b)
        out[f"edge {a}-{b}"] = await store.get_edge(a, b)
        out[f"edeg {a}-{b}"] = await store.edge_degree(a, b)
    kg = await store.get_knowledge_graph(nodes[0], max_depth=2, max_nodes=6)
    out["kg nodes"] = sorted(n.id for n in kg.nodes)
    out["kg edges"] = sorted(e.id for e in kg.edges)
    out["kg truncated"] = kg.is_truncated
    star = await store.get_knowledge_graph("*", max_nodes=5)
    out["star"] = (sorted(n.id for n in star.nodes), star.is_truncated)
    return out


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_matches_networkx_on_random_operations(tmp_path, seed):
    rng = random.Random(seed)
    nodes = [f"n{i}" for i in range(12)]

    async def go():
        strata, nx = make_pair(tmp_path)
        await init(strata, nx)
        for step in range(120):
            op = rng.random()
            a, b = rng.sample(nodes, 2)
            etype = rng.choice("xyz")
            for store in (strata, nx):
                if op < 0.35:
                    await store.upsert_node(a, {"entity_type": etype, "step": str(step)})
                elif op < 0.75:
                    await store.upsert_edge(a, b, {"w": str(step % 5), "weight": "1.0"})
                elif op < 0.83:
                    await store.remove_edges([(a, b)])
                elif op < 0.9:
                    await store.delete_node(a)
                else:
                    await store.remove_nodes([a, b])
        pairs = [(a, b) for a in nodes[:6] for b in nodes[6:]]
        s_snap = await snapshot(strata, nodes, pairs)
        n_snap = await snapshot(nx, nodes, pairs)
        for key in n_snap:
            assert s_snap[key] == n_snap[key], key
        await strata.finalize()

    run(go())


def test_persistence_and_drop(tmp_path):
    async def go():
        strata, _ = make_pair(tmp_path)
        await init(strata)
        await strata.upsert_node("A", {"entity_type": "person"})
        await strata.upsert_edge("A", "B", {"description": "knows", "weight": "2"})
        await strata.index_done_callback()
        await strata.finalize()

        again = StrataGraphStorage(namespace="chunk_entity_relation", workspace="",
                                   global_config={"working_dir": str(tmp_path / "s")}, embedding_func=None)
        await again.initialize()
        assert await again.get_edge("B", "A") == {"description": "knows", "weight": "2"}
        assert await again.get_node("B") == {}  # created implicitly by the edge, like networkx
        assert (await again.drop())["status"] == "success"
        assert await again.get_all_labels() == []
        await again.finalize()

    run(go())


def test_register_graph():
    from lightrag import kg

    register_graph()
    assert "StrataGraphStorage" in kg.STORAGE_IMPLEMENTATIONS["GRAPH_STORAGE"]["implementations"]
