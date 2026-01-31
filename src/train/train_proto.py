import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR

from src.datasets.logodet3k import LogoPatchDataset
from src.datasets.transforms import build_transforms
from src.datasets.episode_sampler import EpisodeSampler
from src.models.proto_net import ProtoNet, prototypical_logits

DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


def build_class_to_idx_from_csv(csv_path: Path):
    df = pd.read_csv(csv_path)
    classes = sorted(df["class_name"].unique().tolist())
    return {c: i for i, c in enumerate(classes)}


def episode_to_support_query(x, y, n_way, k_shot, q_query):
    """
    x: (B, C, H, W), y: (B,) where B = n_way*(k_shot+q_query)
    Construct episode labels 0..n_way-1 based on class IDs present in y.
    """
    # map global labels -> episode labels 0..N-1
    unique = torch.unique(y).tolist()
    # stable order
    unique = sorted(unique)
    mapping = {lab: i for i, lab in enumerate(unique)}
    y_epi = torch.tensor([mapping[int(v)] for v in y.cpu().tolist()], device=y.device)

    # split per class: first K as support, rest Q as query
    support_idx = []
    query_idx = []
    for c in range(n_way):
        idxs = (y_epi == c).nonzero(as_tuple=False).view(-1)
        support_idx.append(idxs[:k_shot])
        query_idx.append(idxs[k_shot:k_shot + q_query])

    support_idx = torch.cat(support_idx, dim=0)
    query_idx = torch.cat(query_idx, dim=0)

    return x[support_idx], y_epi[support_idx], x[query_idx], y_epi[query_idx]


@torch.no_grad()
def eval_episodic(model, loader, n_way, k_shot, q_query):
    model.eval()
    total = 0
    correct = 0
    loss_fn = nn.CrossEntropyLoss()
    total_loss = 0.0

    for x, y, _ in loader:
        x = x.to(DEVICE)
        y = y.to(DEVICE)

        z = model(x)
        xs, ys, xq, yq = episode_to_support_query(x, y, n_way, k_shot, q_query)
        zs = model(xs)
        zq = model(xq)

        logits = prototypical_logits(zs, ys, zq, n_way)
        loss = loss_fn(logits, yq)

        pred = logits.argmax(dim=1)
        correct += (pred == yq).sum().item()
        total += yq.numel()
        total_loss += loss.item() * yq.numel()

    return total_loss / total, correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--embedding_dim", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)

    parser.add_argument("--n_way", type=int, default=5)
    parser.add_argument("--k_shot", type=int, default=5)
    parser.add_argument("--q_query", type=int, default=10)
    parser.add_argument("--episodes_per_epoch", type=int, default=200)
    parser.add_argument("--val_episodes", type=int, default=100)
    parser.add_argument("--aug", action="store_true")

    args = parser.parse_args()

    exp_dir = Path(args.exp_dir)
    exp_dir.mkdir(parents=True, exist_ok=True)
    (exp_dir / "logs").mkdir(parents=True, exist_ok=True)
    (exp_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

    splits_dir = Path("data/processed/splits")
    train_csv = splits_dir / "base_train.csv"
    val_csv = splits_dir / "base_val.csv"

    class_to_idx = build_class_to_idx_from_csv(train_csv)
    num_classes = len(class_to_idx)

    tf_train = build_transforms(train=True, aug=args.aug)
    tf_eval = build_transforms(train=False, aug=False)

    ds_train = LogoPatchDataset(str(train_csv), class_to_idx, transform=tf_train)
    ds_val = LogoPatchDataset(str(val_csv), class_to_idx, transform=tf_eval)

    # episodic samplers
    df_train = pd.read_csv(train_csv)
    df_val = pd.read_csv(val_csv)

    sampler_train = EpisodeSampler(
        df_train, class_to_idx,
        n_way=args.n_way, k_shot=args.k_shot, q_query=args.q_query,
        episodes_per_epoch=args.episodes_per_epoch, seed=42
    )
    sampler_val = EpisodeSampler(
        df_val, class_to_idx,
        n_way=args.n_way, k_shot=args.k_shot, q_query=args.q_query,
        episodes_per_epoch=args.val_episodes, seed=999
    )

    dl_train = DataLoader(ds_train, batch_sampler=sampler_train, num_workers=0)
    dl_val = DataLoader(ds_val, batch_sampler=sampler_val, num_workers=0)

    model = ProtoNet(embedding_dim=args.embedding_dim, pretrained=True)
    model.to(DEVICE)

    optim = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = StepLR(optim, step_size=15, gamma=0.1)
    loss_fn = nn.CrossEntropyLoss()

    cfg = vars(args) | {"device": DEVICE, "num_base_classes": num_classes}
    (exp_dir / "logs" / "config.json").write_text(json.dumps(cfg, indent=2))

    best_val_acc = -1.0
    history = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        total_loss = 0.0
        total_q = 0
        correct = 0

        for x, y, _ in dl_train:
            x = x.to(DEVICE)
            y = y.to(DEVICE)

            xs, ys, xq, yq = episode_to_support_query(x, y, args.n_way, args.k_shot, args.q_query)

            zs = model(xs)
            zq = model(xq)

            logits = prototypical_logits(zs, ys, zq, args.n_way)
            loss = loss_fn(logits, yq)

            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()

            pred = logits.argmax(dim=1)
            correct += (pred == yq).sum().item()
            total_q += yq.numel()
            total_loss += loss.item() * yq.numel()

        sched.step()
        tr_loss = total_loss / total_q
        tr_acc = correct / total_q

        va_loss, va_acc = eval_episodic(model, dl_val, args.n_way, args.k_shot, args.q_query)
        dt = time.time() - t0

        row = {
            "epoch": epoch,
            "train_loss": tr_loss,
            "train_acc": tr_acc,
            "val_loss": va_loss,
            "val_acc": va_acc,
            "seconds": dt
        }
        history.append(row)
        print(row)

        if va_acc > best_val_acc:
            best_val_acc = va_acc
            ckpt = {
                "model_state": model.state_dict(),
                "config": cfg,
                "best_val_acc": best_val_acc,
            }
            torch.save(ckpt, exp_dir / "checkpoints" / "best.pt")

    (exp_dir / "logs" / "history.json").write_text(json.dumps(history, indent=2))
    summary = {"best_val_acc": best_val_acc}
    (exp_dir / "logs" / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()