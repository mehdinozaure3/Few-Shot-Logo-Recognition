import argparse
from pathlib import Path
import json

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
from sklearn.metrics import confusion_matrix

from src.datasets.logodet3k import LogoPatchDataset
from src.datasets.transforms import build_transforms
from src.models.embedder import CEModel

DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    splits_dir = Path("data/processed/splits")
    test_csv = splits_dir / "base_test.csv"

    ckpt = torch.load(args.ckpt, map_location=DEVICE)
    class_to_idx = ckpt["class_to_idx"]
    num_classes = len(class_to_idx)

    tf = build_transforms(train=False, aug=False)
    ds = LogoPatchDataset(str(test_csv), class_to_idx, transform=tf)
    dl = DataLoader(ds, batch_size=64, shuffle=False, num_workers=0)

    model = CEModel(num_classes=num_classes, embedding_dim=ckpt["config"]["embedding_dim"], pretrained=False)
    model.load_state_dict(ckpt["model_state"])
    model.to(DEVICE)
    model.eval()

    y_true, y_pred = [], []
    loss_fn = nn.CrossEntropyLoss()
    total_loss, n = 0.0, 0

    for x, y, _ in dl:
        x = x.to(DEVICE)
        y = y.to(DEVICE)
        logits, _ = model(x)
        loss = loss_fn(logits, y)
        total_loss += loss.item() * x.size(0)
        n += x.size(0)
        pred = logits.argmax(dim=1)
        y_true.extend(y.cpu().tolist())
        y_pred.extend(pred.cpu().tolist())

    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    (out_dir / "confusion_matrix.json").write_text(json.dumps(cm.tolist()))
    (out_dir / "base_test_loss.json").write_text(json.dumps({"loss": total_loss / n}, indent=2))

    print("Saved confusion matrix to:", out_dir / "confusion_matrix.json")

if __name__ == "__main__":
    main()