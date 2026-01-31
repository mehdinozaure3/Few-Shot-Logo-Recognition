import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights

class ResNetEmbedder(nn.Module):
    def __init__(self, embedding_dim: int = 256, pretrained: bool = True):
        super().__init__()
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        backbone = resnet18(weights=weights)

        self.backbone = nn.Sequential(*list(backbone.children())[:-1])  # (B,512,1,1)
        self.proj = nn.Linear(512, embedding_dim)

    def forward(self, x):
        feat = self.backbone(x).flatten(1)
        emb = self.proj(feat)
        emb = F.normalize(emb, p=2, dim=1)
        return emb

class CEModel(nn.Module):
    def __init__(self, num_classes: int, embedding_dim: int = 256, pretrained: bool = True):
        super().__init__()
        self.embedder = ResNetEmbedder(embedding_dim=embedding_dim, pretrained=pretrained)
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(self, x):
        z = self.embedder(x)
        logits = self.classifier(z)
        return logits, z