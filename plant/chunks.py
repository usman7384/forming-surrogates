"""Chunk encoding of high-rate signals for OPC-UA array nodes (layout is defined in docs/signal-interface.yaml).

A chunk is a flat float64 array [part_id, chunk_seq, t0, v0, q0, t1, v1, q1, ...]. One array write is atomic,
so a subscriber never sees timestamps from one write and values from another.
"""
from __future__ import annotations

import numpy as np


def encode_chunks(part_id: int, t, v, q, t_start: float, chunk_s: float) -> list[np.ndarray]:
    """Split one signal of one part into time chunks (empty chunks are skipped; seq numbers stay consecutive)."""
    t, v, q = (np.asarray(a, float) for a in (t, v, q))
    idx = np.floor((t - t_start) / chunk_s).astype(int)
    out = []
    for seq, k in enumerate(np.unique(idx)):
        m = idx == k
        body = np.column_stack([t[m], v[m], q[m]]).ravel()
        out.append(np.concatenate([[float(part_id), float(seq)], body]))
    return out


def decode_chunk(arr) -> tuple[int, int, np.ndarray, np.ndarray, np.ndarray]:
    a = np.asarray(arr, float)
    body = a[2:].reshape(-1, 3)
    return int(a[0]), int(a[1]), body[:, 0], body[:, 1], body[:, 2]
