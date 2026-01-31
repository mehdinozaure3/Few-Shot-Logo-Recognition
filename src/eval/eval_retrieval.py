import argparse
from pathlib import Path
import json

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.datasets.logodet3k import LogoPatchDataset
from src.datasets.transforms import build_transforms
from src.models.embedder import CEModel

DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


@torch.no_grad()
def compute_embeddings(model, loader):
    embs = []
    labels = []
    for x, y, _ in loader:
        x = x.to(DEVICE)
        _, z = model(x)  # (B, D) L2-normalized
        embs.append(z.cpu().numpy())
        labels.append(y.numpy())
    return np.vstack(embs), np.concatenate(labels)


def average_precision(ranked_relevant: np.ndarray) -> float:
    # ranked_relevant: boolean array where True indicates relevant at rank i
    if ranked_relevant.sum() == 0:
        return 0.0
    precisions = []
    hit = 0
    for i, rel in enumerate(ranked_relevant, start=1):
        if rel:
            hit += 1
            precisions.append(hit / i)
    return float(np.mean(precisions))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--split", type=str, default="novel_test", choices=["novel_val", "novel_test"])
    parser.add_argument("--out_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    splits_dir = Path("data/processed/splits")
    gallery_csv = splits_dir / f"{args.split}_gallery.csv"
    query_csv = splits_dir / f"{args.split}_query.csv"

    assert gallery_csv.exists(), f"Missing {gallery_csv}"
    assert query_csv.exists(), f"Missing {query_csv}"

    ckpt = torch.load(args.ckpt, map_location=DEVICE)

    # Build label mapping from novel split (gallery + query) to avoid KeyError on novel classes
    import pandas as pd
    df_g = pd.read_csv(gallery_csv)
    df_q = pd.read_csv(query_csv)
    novel_classes = sorted(set(df_g["class_name"].unique()) | set(df_q["class_name"].unique()))
    novel_class_to_idx = {c: i for i, c in enumerate(novel_classes)}

    # Model must be built with base num_classes to load the CE checkpoint state_dict correctly
    base_num_classes = len(ckpt["class_to_idx"])
    embedding_dim = int(ckpt["config"]["embedding_dim"])

    tf = build_transforms(train=False, aug=False)

    ds_g = LogoPatchDataset(str(gallery_csv), novel_class_to_idx, transform=tf)
    ds_q = LogoPatchDataset(str(query_csv), novel_class_to_idx, transform=tf)
    dl_g = DataLoader(ds_g, batch_size=args.batch_size, shuffle=False, num_workers=0)
    dl_q = DataLoader(ds_q, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = CEModel(num_classes=base_num_classes, embedding_dim=embedding_dim, pretrained=False)
    model.load_state_dict(ckpt["model_state"])
    model.to(DEVICE)
    model.eval()

    G, yG = compute_embeddings(model, dl_g)
    Q, yQ = compute_embeddings(model, dl_q)

    # Cosine similarity since embeddings are L2 normalized: sim = dot
    sims = Q @ G.T

    recall1, recall5 = 0, 0
    aps = []

    for i in range(sims.shape[0]):
        scores = sims[i]
        ranking = np.argsort(-scores)  # descending
        rel = (yG[ranking] == yQ[i])
        recall1 += int(rel[:1].any())
        recall5 += int(rel[:5].any())
        aps.append(average_precision(rel))

    nQ = sims.shape[0]
    metrics = {
        "split": args.split,
        "num_gallery": int(len(yG)),
        "num_query": int(len(yQ)),
        "recall@1": float(recall1 / nQ),
        "recall@5": float(recall5 / nQ),
        "mAP": float(np.mean(aps)),
    }

    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()