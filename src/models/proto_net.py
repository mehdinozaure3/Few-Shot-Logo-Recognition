import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.embedder import ResNetEmbedder


class ProtoNet(nn.Module):
    def __init__(self, embedding_dim: int = 256, pretrained: bool = True):
        super().__init__()
        self.embedder = ResNetEmbedder(embedding_dim=embedding_dim, pretrained=pretrained)

    def forward(self, x):
        return self.embedder(x)


def prototypical_logits(z_support, y_support, z_query, n_way: int):
    """
    z_support: (N*K, D)
    y_support: (N*K,) in [0..N-1] episode labels
    z_query:   (N*Q, D)
    returns logits: (N*Q, N) where higher is better (negative distance)
    """
    D = z_support.size(-1)

    prototypes = []
    for c in range(n_way):
        prototypes.append(z_support[y_support == c].mean(dim=0))
    prototypes = torch.stack(prototypes, dim=0)  # (N, D)

    dists = (z_query.unsqueeze(1) - prototypes.unsqueeze(0)).pow(2).sum(dim=2)  # (NQ, N)
    logits = -dists
    return logits