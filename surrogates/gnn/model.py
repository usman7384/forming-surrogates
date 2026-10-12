"""Encode-process-decode GNN (MeshGraphNets style) in PyTorch Geometric (task N3).

Why this shape: the mesh is small (50-560 nodes) so a modest latent width and 10-15 residual
message-passing blocks are enough for information to cross the ring; layer norm keeps deep stacks
trainable. Global inputs (m_f, m, s) are concatenated to every node feature at the encoder, which is
the simplest way to condition on them. The force head mean-pools nodes, then an MLP.
"""
from __future__ import annotations

import torch
from torch import nn
from torch_geometric.nn import MessagePassing, global_mean_pool

from .graph import N_EDGE_IN, N_GLOBAL, N_NODE_IN, N_NODE_OUT


def mlp(n_in: int, n_hidden: int, n_out: int, layer_norm: bool = True) -> nn.Sequential:
    layers = [nn.Linear(n_in, n_hidden), nn.SiLU(), nn.Linear(n_hidden, n_out)]
    if layer_norm:
        layers.append(nn.LayerNorm(n_out))
    return nn.Sequential(*layers)


class Block(MessagePassing):
    """Edge update from (edge, sender, receiver) latents, sum-aggregate, node update; both residual."""

    def __init__(self, d: int):
        super().__init__(aggr="sum")
        self.edge_mlp = mlp(3 * d, d, d)
        self.node_mlp = mlp(2 * d, d, d)

    def forward(self, h, edge_index, e):
        src, dst = edge_index
        e_new = e + self.edge_mlp(torch.cat([e, h[src], h[dst]], dim=-1))
        agg = self.propagate(edge_index, e=e_new, size=(h.shape[0], h.shape[0]))
        h_new = h + self.node_mlp(torch.cat([h, agg], dim=-1))
        return h_new, e_new

    def message(self, e):
        return e


class MeshGNN(nn.Module):
    def __init__(self, latent: int = 64, steps: int = 10):
        super().__init__()
        self.node_enc = mlp(N_NODE_IN + N_GLOBAL, latent, latent)
        self.edge_enc = mlp(N_EDGE_IN, latent, latent)
        self.blocks = nn.ModuleList([Block(latent) for _ in range(steps)])
        self.node_dec = mlp(latent, latent, N_NODE_OUT, layer_norm=False)
        self.force_head = mlp(latent, latent, 1, layer_norm=False)

    def forward(self, data):
        batch = data.batch if data.batch is not None else torch.zeros(data.x.shape[0], dtype=torch.long)
        g = data.glob[batch]
        h = self.node_enc(torch.cat([data.x, g], dim=-1))
        e = self.edge_enc(data.edge_attr)
        for blk in self.blocks:
            h, e = blk(h, data.edge_index, e)
        out = self.node_dec(h)
        force = self.force_head(global_mean_pool(h, batch))
        return out[:, :2], out[:, 2:3], force      # displacement, strain, force (all normalised)
