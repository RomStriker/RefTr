# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Positional encodings for the transformer.
"""
import torch
import numpy as np
from torch import nn


class PositionEmbeddingLearned3D(nn.Module):
    """
    Learned positional embeddings for the transformer.
    """
    def __init__(self, vol_size=32, dim_size=256):
        super().__init__()
        self.dim_size = dim_size
        self.num_pos_feats = int(np.ceil(dim_size / 6) * 2)
        self.row_embed = nn.Embedding(vol_size, self.num_pos_feats)
        self.col_embed = nn.Embedding(vol_size, self.num_pos_feats)
        self.dep_embed = nn.Embedding(vol_size, self.num_pos_feats)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.row_embed.weight)
        nn.init.uniform_(self.col_embed.weight)
        nn.init.uniform_(self.dep_embed.weight)

    def forward(self, tensor_list):
        x = tensor_list
        h, w, d = x.shape[-3:]

        # Create embeddings for each dimension
        x_emb = self.col_embed(torch.arange(w, device=x.device))  # (w, num_feats)
        y_emb = self.row_embed(torch.arange(h, device=x.device))  # (h, num_feats)
        z_emb = self.dep_embed(torch.arange(d, device=x.device))  # (d, num_feats)
        x_emb = x_emb.unsqueeze(0).unsqueeze(1).expand(d, h, -1, -1)  # (d, h, w, num_feats)
        y_emb = y_emb.unsqueeze(0).unsqueeze(2).expand(d, -1, w, -1)  # (d, h, w, num_feats)
        z_emb = z_emb.unsqueeze(1).unsqueeze(2).expand(-1, h, w, -1)  # (d, h, w, num_feats)

        # Concatenate and reshape
        pos = torch.cat([x_emb, y_emb, z_emb], dim=-1)  # (d, h, w, 3*num_feats)
        pos = pos.permute(3, 0, 1, 2)  # (3*num_feats, d, h, w)
        pos = pos.unsqueeze(0).expand(x.shape[0], -1, -1, -1, -1)  # (bs, 3*num_feats, d, h, w)

        # Trim to desired dimension if needed
        pos = pos[:, :self.dim_size, ...]
        return pos