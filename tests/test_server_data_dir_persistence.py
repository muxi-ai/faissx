#!/usr/bin/env python3

import numpy as np

from faissx.server.server import FaissIndex


def test_data_dir_persists_index_across_restart(tmp_path):
    data_dir = tmp_path / "faissx-data"
    index_id = "persisted_index"

    server_a = FaissIndex(data_dir=str(data_dir))
    create_response = server_a.create_index(index_id, 4, "L2")
    assert create_response.get("success") is True

    vectors = np.array(
        [[0.1, 0.2, 0.3, 0.4], [0.2, 0.1, 0.4, 0.3]],
        dtype=np.float32,
    ).tolist()
    add_response = server_a.add_vectors(index_id, vectors)
    assert add_response.get("success") is True
    assert server_a.indexes[index_id].ntotal == 2

    # Writes are debounced and performed by a background thread; flush to
    # guarantee durability before handing the data dir to another instance.
    server_a.flush()

    server_b = FaissIndex(data_dir=str(data_dir))
    assert index_id in server_b.indexes
    assert server_b.indexes[index_id].ntotal == 2


def test_persist_interval_debounces_writes(tmp_path):
    data_dir = tmp_path / "faissx-data"
    index_id = "debounced_index"

    server = FaissIndex(data_dir=str(data_dir), persist_interval=3600)
    server.create_index(index_id, 4, "L2")
    server.flush()

    vectors = np.random.rand(2, 4).astype(np.float32).tolist()
    server.add_vectors(index_id, vectors)
    server.add_vectors(index_id, vectors)

    # Within the interval the adds stay dirty; the persisted file still holds
    # the pre-add snapshot.
    with server._persist_cv:
        server._persist_cv.wait_for(
            lambda: not server._pending_writes and server._writing_id is None
        )
    restored = FaissIndex(data_dir=str(data_dir))
    assert restored.indexes[index_id].ntotal < 4

    # flush() persists the outstanding dirty state.
    server.flush()
    restored = FaissIndex(data_dir=str(data_dir))
    assert restored.indexes[index_id].ntotal == 4


def test_persist_interval_zero_writes_synchronously(tmp_path):
    data_dir = tmp_path / "faissx-data"
    index_id = "sync_index"

    server = FaissIndex(data_dir=str(data_dir), persist_interval=0)
    server.create_index(index_id, 4, "L2")
    vectors = np.random.rand(3, 4).astype(np.float32).tolist()
    server.add_vectors(index_id, vectors)

    # No flush needed: every mutation is written before the call returns.
    restored = FaissIndex(data_dir=str(data_dir))
    assert restored.indexes[index_id].ntotal == 3


def test_delete_index_cancels_pending_writes(tmp_path):
    data_dir = tmp_path / "faissx-data"
    index_id = "deleted_index"

    server = FaissIndex(data_dir=str(data_dir))
    server.create_index(index_id, 4, "L2")
    vectors = np.random.rand(2, 4).astype(np.float32).tolist()
    server.add_vectors(index_id, vectors)
    server.delete_index(index_id)
    server.flush()

    base_path = server._persisted_index_base_path(index_id)
    assert not base_path.with_suffix(".faiss").exists()
    assert not base_path.with_suffix(".json").exists()

    restored = FaissIndex(data_dir=str(data_dir))
    assert index_id not in restored.indexes


def test_data_dir_expands_user_home(monkeypatch, tmp_path):
    home_dir = tmp_path / "home"
    home_dir.mkdir()
    monkeypatch.setenv("HOME", str(home_dir))

    server = FaissIndex(data_dir="~/faissx-home-data")
    assert str(server.data_dir).startswith(str(home_dir))
