"""StrataVectorDBStorage against LightRAG's vector-storage contract, with a fake embedder."""

import asyncio
import hashlib

import numpy as np
import pytest
from lightrag.utils import EmbeddingFunc, compute_mdhash_id

from lightrag_strata.strata_storage import StrataVectorDBStorage, register

DIM = 32


async def fake_embed(texts, **kwargs):
    out = []
    for t in texts:
        seed = int(hashlib.md5(t.encode()).hexdigest()[:8], 16)
        out.append(np.random.default_rng(seed).normal(size=DIM))
    return np.array(out, dtype=np.float32)


def make_storage(tmp_path, namespace="entities", meta_fields=("content", "entity_name", "src_id", "tgt_id")):
    return StrataVectorDBStorage(
        namespace=namespace,
        workspace="",
        global_config={
            "working_dir": str(tmp_path),
            "embedding_batch_num": 4,
            "vector_db_storage_cls_kwargs": {"cosine_better_than_threshold": -1.0},
        },
        embedding_func=EmbeddingFunc(embedding_dim=DIM, func=fake_embed),
        meta_fields=set(meta_fields),
    )


def run(coro):
    return asyncio.run(coro)


def test_register_adds_backend():
    from lightrag import kg

    register()
    assert "StrataVectorDBStorage" in kg.STORAGE_IMPLEMENTATIONS["VECTOR_STORAGE"]["implementations"]
    assert kg.STORAGES["StrataVectorDBStorage"] == "lightrag_strata.strata_storage"


def test_upsert_query_get(tmp_path):
    async def go():
        st = make_storage(tmp_path)
        await st.initialize()
        data = {
            compute_mdhash_id(name, prefix="ent-"): {"content": f"{name} description", "entity_name": name}
            for name in ["Alpha", "Beta", "Gamma", "Delta", "Epsilon"]
        }
        await st.upsert(data)
        await st.index_done_callback()
        alpha = compute_mdhash_id("Alpha", prefix="ent-")
        hits = await st.query("Alpha description", top_k=2)
        assert hits[0]["id"] == alpha and hits[0]["entity_name"] == "Alpha"
        assert hits[0]["distance"] == pytest.approx(1.0, abs=1e-4)
        assert isinstance(hits[0]["created_at"], int)
        got = await st.get_by_ids([alpha, "missing"])
        assert got[0]["content"] == "Alpha description" and got[1] is None
        assert (await st.get_by_id("missing")) is None
        vecs = await st.get_vectors_by_ids([alpha])
        assert len(vecs[alpha]) == DIM
        await st.delete_entity("Alpha")
        assert (await st.get_by_id(alpha)) is None
        await st.finalize()

    run(go())


def test_relations_are_linked_and_deleted_by_entity(tmp_path):
    async def go():
        st = make_storage(tmp_path, namespace="relationships")
        await st.initialize()
        rels = {
            "rel-1": {"content": "A knows B", "src_id": "A", "tgt_id": "B"},
            "rel-2": {"content": "B owns C", "src_id": "B", "tgt_id": "C"},
            "rel-3": {"content": "C likes D", "src_id": "C", "tgt_id": "D"},
        }
        await st.upsert(rels)
        await st.delete_entity_relation("B")
        remaining = await st.get_by_ids(["rel-1", "rel-2", "rel-3"])
        assert remaining[0] is None and remaining[1] is None and remaining[2]["src_id"] == "C"
        await st.finalize()

    run(go())


def test_persists_across_restart_and_drop(tmp_path):
    async def go():
        st = make_storage(tmp_path, namespace="chunks", meta_fields=("content",))
        await st.initialize()
        await st.upsert({"c1": {"content": "hello world"}})
        await st.index_done_callback()
        await st.finalize()

        st2 = make_storage(tmp_path, namespace="chunks", meta_fields=("content",))
        await st2.initialize()
        assert (await st2.get_by_id("c1"))["content"] == "hello world"
        assert (await st2.drop())["status"] == "success"
        assert (await st2.get_by_id("c1")) is None
        await st2.finalize()

    run(go())
