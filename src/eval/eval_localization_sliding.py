import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader

from src.datasets.logodet3k import LogoPatchDataset
from src.datasets.transforms import build_transforms
from src.models.proto_net import ProtoNet
from src.eval.iou import iou_xyxy

DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


@torch.no_grad()
def embed_patches(model, csv_path: Path, class_to_idx: dict, batch_size: int):
    tf = build_transforms(train=False, aug=False)
    ds = LogoPatchDataset(str(csv_path), class_to_idx, transform=tf)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)

    Z = []
    Y = []
    IDS = []
    for x, y, instance_id in dl:
        x = x.to(DEVICE)
        z = model(x)  # (B,D) L2-normalized
        Z.append(z.cpu().numpy())
        Y.append(y.numpy())
        IDS.extend(instance_id)
    return np.vstack(Z), np.concatenate(Y), IDS


def build_class_to_idx_from_csv(csv_path: Path):
    df = pd.read_csv(csv_path)
    classes = sorted(df["class_name"].unique().tolist())
    return {c: i for i, c in enumerate(classes)}


def generate_windows(w, h, scales, stride):
    # returns list of (x1,y1,x2,y2)
    boxes = []
    for s in scales:
        win = int(round(min(w, h) * s))
        win = max(32, win)  # safety
        if win >= w or win >= h:
            continue
        for y1 in range(0, h - win + 1, stride):
            for x1 in range(0, w - win + 1, stride):
                boxes.append((x1, y1, x1 + win, y1 + win))
    return boxes


@torch.no_grad()
def score_windows(model, pil_img, boxes, proto, batch_size=64):
    """
    proto: (D,) numpy L2-normalized
    returns best_box, best_score
    """
    tf = build_transforms(train=False, aug=False)

    best_score = -1e9
    best_box = None

    # batch crops
    batch = []
    batch_boxes = []

    def flush():
        nonlocal best_score, best_box, batch, batch_boxes
        if not batch:
            return
        x = torch.stack(batch, dim=0).to(DEVICE)
        z = model(x).cpu().numpy()  # (B,D)
        scores = z @ proto  # cosine dot
        mi = int(np.argmax(scores))
        if float(scores[mi]) > best_score:
            best_score = float(scores[mi])
            best_box = batch_boxes[mi]
        batch = []
        batch_boxes = []

    for b in boxes:
        crop = pil_img.crop(b).convert("RGB")
        x = tf(crop)
        batch.append(x)
        batch_boxes.append(b)
        if len(batch) >= batch_size:
            flush()

    flush()
    return best_box, best_score


def draw_boxes(img, gt, pred, out_path: Path):
    im = img.copy().convert("RGB")
    d = ImageDraw.Draw(im)
    # GT in green
    d.rectangle(gt, outline=(0, 255, 0), width=4)
    # Pred in red
    if pred is not None:
        d.rectangle(pred, outline=(255, 0, 0), width=4)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    im.save(out_path, quality=95)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--split", type=str, default="novel_test", choices=["novel_val","novel_test"])
    parser.add_argument("--k_shot", type=int, default=1)
    parser.add_argument("--max_instances", type=int, default=300)  # keep runtime reasonable
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--stride", type=int, default=32)
    parser.add_argument("--scales", type=str, default="0.25,0.35,0.45")  # window size ratios
    parser.add_argument("--batch_size", type=int, default=64)

    parser.add_argument("--out_dir", type=str, required=True)
    args = parser.parse_args()

    rng = random.Random(args.seed)

    splits_dir = Path("data/processed/splits")
    query_csv = splits_dir / f"{args.split}_query.csv"

    loc_csv = Path("data/processed/localization") / f"{args.split}_instances.csv"
    assert loc_csv.exists(), f"Missing {loc_csv} (run scripts/04_make_localization_csv.py but for {args.split})"
    assert query_csv.exists(), f"Missing {query_csv}"

    # Build label mapping only for novel split
    class_to_idx = build_class_to_idx_from_csv(query_csv)

    # Load ProtoNet
    ckpt = torch.load(args.ckpt, map_location=DEVICE)
    emb_dim = ckpt["config"]["embedding_dim"]
    model = ProtoNet(embedding_dim=emb_dim, pretrained=False)
    model.load_state_dict(ckpt["model_state"])
    model.to(DEVICE)
    model.eval()

    # Pre-embed query pool patches once (used to sample supports)
    ZQ, yQ, idsQ = embed_patches(model, query_csv, class_to_idx, batch_size=128)

    # group indices per class in query pool
    by_class = {}
    for c in np.unique(yQ):
        idxs = np.where(yQ == c)[0].tolist()
        by_class[int(c)] = idxs

    scales = [float(s) for s in args.scales.split(",") if s.strip()]
    out_dir = Path(args.out_dir)
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)
    (out_dir / "logs").mkdir(parents=True, exist_ok=True)

    df_loc = pd.read_csv(loc_csv)
    # keep only instances whose class exists in query pool mapping
    df_loc = df_loc[df_loc["class_name"].isin(class_to_idx.keys())].copy()

    # sample a subset for runtime
    if len(df_loc) > args.max_instances:
        df_loc = df_loc.sample(n=args.max_instances, random_state=args.seed).reset_index(drop=True)

    ious = []
    succ05 = 0
    succ03 = 0

    for i, row in df_loc.iterrows():
        cls = row["class_name"]
        c = class_to_idx[cls]

        # sample supports from query pool
        idxs = by_class.get(c, [])
        if len(idxs) < args.k_shot:
            continue
        rng.shuffle(idxs)
        support = idxs[:args.k_shot]

        proto = ZQ[support].mean(axis=0)
        proto = proto / (np.linalg.norm(proto) + 1e-12)

        img_path = Path(row["image_path"])
        if not img_path.exists():
            continue

        img = Image.open(img_path).convert("RGB")
        w, h = img.size

        gt = (int(row["x1"]), int(row["y1"]), int(row["x2"]), int(row["y2"]))

        boxes = generate_windows(w, h, scales=scales, stride=args.stride)
        if not boxes:
            continue

        pred, score = score_windows(model, img, boxes, proto, batch_size=args.batch_size)
        if pred is None:
            continue

        j = iou_xyxy(gt, pred)
        ious.append(j)
        succ05 += int(j >= 0.5)
        succ03 += int(j >= 0.3)

        # save a few qualitative examples
        if i < 30:
            out_img = out_dir / "figures" / f"loc_{i:03d}_{cls}.jpg"
            draw_boxes(img, gt, pred, out_img)

    n = len(ious)
    metrics = {
        "split": args.split,
        "k_shot": args.k_shot,
        "max_instances": args.max_instances,
        "stride": args.stride,
        "scales": scales,
        "num_eval": n,
        "mean_iou": float(np.mean(ious)) if ious else 0.0,
        "success@0.5": float(succ05 / n) if n else 0.0,
        "success@0.3": float(succ03 / n) if n else 0.0,
    }

    (out_dir / "logs" / f"localization_{args.split}_{args.k_shot}shot.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()