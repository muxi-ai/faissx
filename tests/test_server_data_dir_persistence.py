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

    server_b = FaissIndex(data_dir=str(data_dir))
    assert index_id in server_b.indexes
    assert server_b.indexes[index_id].ntotal == 2


def test_data_dir_expands_user_home(monkeypatch, tmp_path):
    home_dir = tmp_path / "home"
    home_dir.mkdir()
    monkeypatch.setenv("HOME", str(home_dir))

    server = FaissIndex(data_dir="~/faissx-home-data")
    assert str(server.data_dir).startswith(str(home_dir))
