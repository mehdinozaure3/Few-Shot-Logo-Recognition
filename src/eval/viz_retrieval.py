import argparse
from pathlib import Path
import random

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from PIL import Image, ImageDraw

from src.datasets.logodet3k import LogoPatchDataset
from src.datasets.transforms import build_transforms
from src.models.embedder import CEModel
from src.models.proto_net import ProtoNet

DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


@torch.no_grad()
def compute_embeddings(model, loader, model_type: str):
    """
    Returns:
      Z: (N, D) embeddings
      Y: (N,) integer labels (as produced by dataset class_to_idx)
    """
    Z, Y = [], []
    for x, y, _ in loader:
        x = x.to(DEVICE)

        if model_type == "ce":
            # CEModel returns (logits, z)
            _, z = model(x)
        else:
            # ProtoNet returns z
            z = model(x)

        Z.append(z.cpu().numpy())
        Y.append(y.numpy())

    return np.vstack(Z), np.concatenate(Y)


def make_row_canvas(query_img, gallery_imgs, correct, tile_size=224):
    query_img = query_img.resize((tile_size, tile_size))
    gallery_imgs = [im.resize((tile_size, tile_size)) for im in gallery_imgs]

    W, H = tile_size, tile_size
    canvas = Image.new("RGB", ((1 + len(gallery_imgs)) * W, H), (255, 255, 255))

    # query
    canvas.paste(query_img, (0, 0))

    d = ImageDraw.Draw(canvas)

    # gallery
    for i, (img, ok) in enumerate(zip(gallery_imgs, correct)):
        x0 = (i + 1) * W
        canvas.paste(img, (x0, 0))
        color = (0, 200, 0) if ok else (220, 0, 0)
        d.rectangle([x0, 0, x0 + W - 1, H - 1], outline=color, width=6)

    return canvas


def build_class_to_idx_from_split(gallery_csv: Path, query_csv: Path):
    df_g = pd.read_csv(gallery_csv)
    df_q = pd.read_csv(query_csv)
    classes = sorted(set(df_g["class_name"].unique()) | set(df_q["class_name"].unique()))
    return {c: i for i, c in enumerate(classes)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--model", type=str, required=True, choices=["proto", "ce"])
    parser.add_argument("--split", type=str, required=True, choices=["novel_val", "novel_test"])
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--num_queries", type=int, default=6)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)

    splits_dir = Path("data/processed/splits")
    gallery_csv = splits_dir / f"{args.split}_gallery.csv"
    query_csv = splits_dir / f"{args.split}_query.csv"

    assert gallery_csv.exists(), f"Missing {gallery_csv}"
    assert query_csv.exists(), f"Missing {query_csv}"

    # Build mapping from the novel split itself
    class_to_idx = build_class_to_idx_from_split(gallery_csv, query_csv)

    df_g = pd.read_csv(gallery_csv)
    df_q = pd.read_csv(query_csv)

    tf = build_transforms(train=False, aug=False)
    ds_g = LogoPatchDataset(str(gallery_csv), class_to_idx, transform=tf)
    ds_q = LogoPatchDataset(str(query_csv), class_to_idx, transform=tf)

    dl_g = DataLoader(ds_g, batch_size=args.batch_size, shuffle=False, num_workers=0)
    dl_q = DataLoader(ds_q, batch_size=args.batch_size, shuffle=False, num_workers=0)

    ckpt = torch.load(args.ckpt, map_location=DEVICE)

    # Build model depending on args.model
    if args.model == "proto":
        emb_dim = ckpt["config"]["embedding_dim"]
        model = ProtoNet(embedding_dim=emb_dim, pretrained=False)
        model.load_state_dict(ckpt["model_state"])
        model.to(DEVICE).eval()
        model_type = "proto"
    else:
        # CE checkpoint must contain class_to_idx + model_state + config
        emb_dim = ckpt["config"]["embedding_dim"]
        num_classes = len(ckpt["class_to_idx"])
        model = CEModel(num_classes=num_classes, embedding_dim=emb_dim, pretrained=False)
        model.load_state_dict(ckpt["model_state"])
        model.to(DEVICE).eval()
        model_type = "ce"

    # Compute embeddings
    G, yG = compute_embeddings(model, dl_g, model_type=model_type)
    Q, yQ = compute_embeddings(model, dl_q, model_type=model_type)

    # Similarity matrix (cosine because embeddings normalized)
    sims = Q @ G.T

    # Choose query indices (random in query pool)
    nQ = len(df_q)
    if nQ == 0:
        raise RuntimeError("Query split is empty.")
    k = min(args.num_queries, nQ)
    q_indices = rng.sample(list(range(nQ)), k=k)

    rows = []
    for qi in q_indices:
        scores = sims[qi]
        ranking = np.argsort(-scores)[: args.topk]

        q_row = df_q.iloc[qi]
        q_img = Image.open(q_row.patch_path).convert("RGB")

        g_imgs = []
        correct = []
        for gi in ranking:
            g_row = df_g.iloc[gi]
            img = Image.open(g_row.patch_path).convert("RGB")
            g_imgs.append(img)
            correct.append(g_row.class_name == q_row.class_name)

        rows.append(make_row_canvas(q_img, g_imgs, correct, tile_size=224))

    if len(rows) == 0:
        print("No rows generated.")
        return

    W = rows[0].size[0]
    H = rows[0].size[1]
    final = Image.new("RGB", (W, H * len(rows)), (255, 255, 255))
    for i, rimg in enumerate(rows):
        final.paste(rimg, (0, i * H))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"qual_retrieval_{args.model}_{args.split}_top{args.topk}.png"
    final.save(out_path)
    print("Saved:", out_path)


if __name__ == "__main__":
    main()
