import os
import json
from pathlib import Path
import time
import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR

import pandas as pd

from src.datasets.logodet3k import LogoPatchDataset
from src.datasets.transforms import build_transforms
from src.models.embedder import CEModel

DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

def topk_acc(logits, y, k=1):
    with torch.no_grad():
        _, pred = logits.topk(k, dim=1)
        correct = pred.eq(y.view(-1, 1)).any(dim=1).float().mean().item()
    return correct

def make_class_to_idx(base_train_csv: str):
    df = pd.read_csv(base_train_csv)
    classes = sorted(df["class_name"].unique().tolist())
    return {c: i for i, c in enumerate(classes)}

def set_finetune_mode(model: CEModel, mode: str):
    # freeze everything first
    for p in model.parameters():
        p.requires_grad = False

    # always train proj + classifier
    for p in model.embedder.proj.parameters():
        p.requires_grad = True
    for p in model.classifier.parameters():
        p.requires_grad = True

    if mode == "frozen":
        return

    if mode == "last_block":
        # unfreeze the last block (layer4) robustly
        last = list(model.embedder.backbone.children())[-3]  # typically layer4
        for p in last.parameters():
            p.requires_grad = True
        return

    if mode == "full":
        for p in model.embedder.backbone.parameters():
            p.requires_grad = True
        return

def train_one_epoch(model, loader, optim, loss_fn):
    model.train()
    if all(not p.requires_grad for p in model.embedder.backbone.parameters()):
        model.embedder.backbone.eval()
    total_loss = 0.0
    total_top1 = 0.0
    total_top5 = 0.0
    n = 0

    for x, y, _ in loader:
        x = x.to(DEVICE)
        y = y.to(DEVICE)

        optim.zero_grad(set_to_none=True)
        logits, _ = model(x)
        loss = loss_fn(logits, y)
        loss.backward()
        optim.step()

        bs = x.size(0)
        total_loss += loss.item() * bs
        total_top1 += topk_acc(logits, y, k=1) * bs
        total_top5 += topk_acc(logits, y, k=5) * bs
        n += bs

    return total_loss / n, total_top1 / n, total_top5 / n

@torch.no_grad()
def eval_one_epoch(model, loader, loss_fn):
    model.eval()
    total_loss = 0.0
    total_top1 = 0.0
    total_top5 = 0.0
    n = 0

    for x, y, _ in loader:
        x = x.to(DEVICE)
        y = y.to(DEVICE)

        logits, _ = model(x)
        loss = loss_fn(logits, y)

        bs = x.size(0)
        total_loss += loss.item() * bs
        total_top1 += topk_acc(logits, y, k=1) * bs
        total_top5 += topk_acc(logits, y, k=5) * bs
        n += bs

    return total_loss / n, total_top1 / n, total_top5 / n

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_dir", type=str, required=True)
    parser.add_argument("--aug", action="store_true")
    parser.add_argument("--no_aug", action="store_true")
    parser.add_argument("--finetune", type=str, default="frozen",
                        choices=["frozen", "last_block", "full"])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--embedding_dim", type=int, default=256)
    parser.add_argument("--lr_head", type=float, default=1e-3)
    parser.add_argument("--lr_backbone", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    args = parser.parse_args()

    exp_dir = Path(args.exp_dir)
    exp_dir.mkdir(parents=True, exist_ok=True)
    (exp_dir / "logs").mkdir(parents=True, exist_ok=True)
    (exp_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    
    splits_dir = Path("data/processed/splits")

    train_csv = splits_dir / "base_train.csv"
    val_csv   = splits_dir / "base_val.csv"
    test_csv  = splits_dir / "base_test.csv"

    assert train_csv.exists() and val_csv.exists() and test_csv.exists(), "Missing base split CSVs"


    AUG = True if args.aug else False
    if args.no_aug:
        AUG = False

    embedding_dim = args.embedding_dim
    batch_size = args.batch_size
    epochs = args.epochs
    lr_head = args.lr_head
    lr_backbone = args.lr_backbone
    weight_decay = args.weight_decay

    # build label mapping from base classes
    class_to_idx = make_class_to_idx(str(train_csv))
    num_classes = len(class_to_idx)

    # datasets / loaders
    tf_train = build_transforms(train=True, aug=AUG)
    tf_eval  = build_transforms(train=False, aug=False)

    ds_train = LogoPatchDataset(str(train_csv), class_to_idx, transform=tf_train)
    ds_val   = LogoPatchDataset(str(val_csv), class_to_idx, transform=tf_eval)
    ds_test  = LogoPatchDataset(str(test_csv), class_to_idx, transform=tf_eval)

    dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True, num_workers=0)
    dl_val   = DataLoader(ds_val, batch_size=batch_size, shuffle=False, num_workers=0)
    dl_test  = DataLoader(ds_test, batch_size=batch_size, shuffle=False, num_workers=0)

    # model
    model = CEModel(num_classes=num_classes, embedding_dim=embedding_dim, pretrained=True)
    set_finetune_mode(model, args.finetune)  
    model.to(DEVICE)

    # optimize only trainable params (proj + classifier)
    head_params = list(model.embedder.proj.parameters()) + list(model.classifier.parameters())
    backbone_params = [p for p in model.embedder.backbone.parameters() if p.requires_grad]

    param_groups = [
        {"params": head_params, "lr": lr_head},
    ]
    if len(backbone_params) > 0:
        param_groups.append({"params": backbone_params, "lr": lr_backbone})

    optim = AdamW(param_groups, weight_decay=weight_decay)
    sched = StepLR(optim, step_size=10, gamma=0.1)

    loss_fn = nn.CrossEntropyLoss()

    # logging
    cfg = {
        "device": DEVICE,
        "aug": AUG,
        "embedding_dim": embedding_dim,
        "batch_size": batch_size,
        "epochs": epochs,
        "lr_head": lr_head,
        "weight_decay": weight_decay,
        "num_classes": num_classes,
        "finetune": args.finetune,
        "lr_backbone": lr_backbone,
    }
    (exp_dir / "logs" / "config.json").write_text(json.dumps(cfg, indent=2))

    best_val_top1 = -1.0
    history = []

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        tr_loss, tr_top1, tr_top5 = train_one_epoch(model, dl_train, optim, loss_fn)
        va_loss, va_top1, va_top5 = eval_one_epoch(model, dl_val, loss_fn)

        sched.step()
        dt = time.time() - t0

        row = {
            "epoch": epoch,
            "train_loss": tr_loss,
            "train_top1": tr_top1,
            "train_top5": tr_top5,
            "val_loss": va_loss,
            "val_top1": va_top1,
            "val_top5": va_top5,
            "seconds": dt,
        }
        history.append(row)
        print(row)

        # save best
        if va_top1 > best_val_top1:
            best_val_top1 = va_top1
            ckpt = {
                "model_state": model.state_dict(),
                "class_to_idx": class_to_idx,
                "config": cfg,
                "best_val_top1": best_val_top1,
            }
            torch.save(ckpt, exp_dir / "checkpoints" / "best.pt")

    # final test with best model
    ckpt = torch.load(exp_dir / "checkpoints" / "best.pt", map_location=DEVICE)
    model.load_state_dict(ckpt["model_state"])

    te_loss, te_top1, te_top5 = eval_one_epoch(model, dl_test, loss_fn)
    summary = {
        "best_val_top1": best_val_top1,
        "test_loss": te_loss,
        "test_top1": te_top1,
        "test_top5": te_top5,
    }
    (exp_dir / "logs" / "summary.json").write_text(json.dumps(summary, indent=2))
    (exp_dir / "logs" / "history.json").write_text(json.dumps(history, indent=2))

    print("\nFinished.")
    print(json.dumps(summary, indent=2))
    print("Saved:", exp_dir / "checkpoints" / "best.pt")

if __name__ == "__main__":
    main()