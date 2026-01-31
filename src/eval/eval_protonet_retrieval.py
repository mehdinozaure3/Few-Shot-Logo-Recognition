import argparse
from pathlib import Path
import json
import random

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.datasets.logodet3k import LogoPatchDataset
from src.datasets.transforms import build_transforms
from src.models.proto_net import ProtoNet

DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


@torch.no_grad()
def compute_embeddings(model, loader):
    embs = []
    labels = []
    for x, y, _ in loader:
        x = x.to(DEVICE)
        z = model(x)  # (B, D) L2-normalized
        embs.append(z.cpu().numpy())
        labels.append(y.numpy())
    return np.vstack(embs), np.concatenate(labels)


def average_precision(ranked_relevant: np.ndarray) -> float:
    if ranked_relevant.sum() == 0:
        return 0.0
    precisions = []
    hit = 0
    for i, rel in enumerate(ranked_relevant, start=1):
        if rel:
            hit += 1
            precisions.append(hit / i)
    return float(np.mean(precisions))


def build_class_to_idx_from_split(gallery_csv: Path, query_csv: Path):
    df_g = pd.read_csv(gallery_csv)
    df_q = pd.read_csv(query_csv)
    classes = sorted(set(df_g["class_name"].unique()) | set(df_q["class_name"].unique()))
    return {c: i for i, c in enumerate(classes)}


def sample_episode_indices(yQ: np.ndarray, n_way: int, k_shot: int, seed: int):
    """
    From the QUERY POOL, sample an episode:
    - choose N classes
    - take K support per class
    - use remaining samples of those classes as episode queries
      (we will cap the number of queries per class with --q_query if desired)
    Returns:
      classes_epi: list of class ids
      support_idx: list of indices into Q
      query_idx: list of indices into Q
    """
    rng = random.Random(seed)

    by_class = {}
    for cls in np.unique(yQ):
        idxs = np.where(yQ == cls)[0].tolist()
        by_class[int(cls)] = idxs

    # keep only classes with >= K+1 to have at least 1 query
    valid = [c for c, idxs in by_class.items() if len(idxs) >= (k_shot + 1)]
    if len(valid) < n_way:
        raise ValueError(f"Not enough classes with >=(K+1). valid={len(valid)}, need n_way={n_way}")

    classes_epi = rng.sample(valid, n_way)

    support_idx = []
    query_idx = []

    for c in classes_epi:
        idxs = by_class[c][:]
        rng.shuffle(idxs)
        support = idxs[:k_shot]
        query = idxs[k_shot:] 
        support_idx.extend(support)
        query_idx.extend(query)

    return classes_epi, support_idx, query_idx


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--split", type=str, required=True, choices=["novel_val", "novel_test"])
    parser.add_argument("--out_path", type=str, required=True)

    parser.add_argument("--n_way", type=int, default=5)
    parser.add_argument("--k_shot", type=int, default=1)
    parser.add_argument("--q_query", type=int, default=10)  # cap queries/class in episode
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    splits_dir = Path("data/processed/splits")
    gallery_csv = splits_dir / f"{args.split}_gallery.csv"
    query_csv = splits_dir / f"{args.split}_query.csv"

    assert gallery_csv.exists(), f"Missing {gallery_csv}"
    assert query_csv.exists(), f"Missing {query_csv}"

    # build novel label mapping from the novel split itself
    class_to_idx = build_class_to_idx_from_split(gallery_csv, query_csv)

    tf = build_transforms(train=False, aug=False)

    ds_g = LogoPatchDataset(str(gallery_csv), class_to_idx, transform=tf)
    ds_q = LogoPatchDataset(str(query_csv), class_to_idx, transform=tf)
    dl_g = DataLoader(ds_g, batch_size=args.batch_size, shuffle=False, num_workers=0)
    dl_q = DataLoader(ds_q, batch_size=args.batch_size, shuffle=False, num_workers=0)

    ckpt = torch.load(args.ckpt, map_location=DEVICE)

    # Build ProtoNet model
    # Note: training checkpoint contains ProtoNet weights, not CEModel.
    embedding_dim = ckpt["config"]["embedding_dim"]
    model = ProtoNet(embedding_dim=embedding_dim, pretrained=False)
    model.load_state_dict(ckpt["model_state"])
    model.to(DEVICE)
    model.eval()

    # Compute embeddings once
    G, yG = compute_embeddings(model, dl_g)  # gallery embeddings/labels
    Q, yQ = compute_embeddings(model, dl_q)  # query-pool embeddings/labels

    rng = random.Random(args.seed)

    recall1_total = 0
    recall5_total = 0
    aps = []
    episodic_acc = []

    for ep in range(args.episodes):
        # different seed per episode
        ep_seed = rng.randint(0, 10**9)

        classes_epi, support_idx, query_idx = sample_episode_indices(
            yQ=yQ, n_way=args.n_way, k_shot=args.k_shot, seed=ep_seed
        )

        # build prototypes from supports
        prototypes = []
        for c in classes_epi:
            zc = Q[[i for i in support_idx if yQ[i] == c]]
            proto = zc.mean(axis=0)
            # L2 normalize proto for cosine dot product consistency
            proto = proto / (np.linalg.norm(proto) + 1e-12)
            prototypes.append(proto)
        prototypes = np.stack(prototypes, axis=0)  # (N, D)

        # evaluate each episode query against the full galery
        # cap queries per class to q_query
        final_query_idx = []
        for c in classes_epi:
            idxs_c = [i for i in query_idx if yQ[i] == c]
            rng_ep = random.Random(ep_seed + c)
            rng_ep.shuffle(idxs_c)
            final_query_idx.extend(idxs_c[: args.q_query])

        # If too few queries, skip episode
        if len(final_query_idx) == 0:
            continue

        # episode accuracy: classify query by nearest prototype (5-way classification)
        # and retrieval: use nearest prototype to decide target class, then measure retrieval from gallery
        correct_cls = 0
        n_cls = 0

        for qi in final_query_idx:
            zq = Q[qi]
            zq = zq / (np.linalg.norm(zq) + 1e-12)

            # classify by cosine to prototypes
            proto_sims = prototypes @ zq
            pred_pos = int(np.argmax(proto_sims))
            pred_class = classes_epi[pred_pos]
            true_class = int(yQ[qi])

            correct_cls += int(pred_class == true_class)
            n_cls += 1

            # retrieval: rank gallery by cosine similarity to query embedding
            sims = G @ zq 
            ranking = np.argsort(-sims)

            rel = (yG[ranking] == true_class)
            recall1_total += int(rel[:1].any())
            recall5_total += int(rel[:5].any())
            aps.append(average_precision(rel))

        episodic_acc.append(correct_cls / max(1, n_cls))

    nQ = len(aps)
    metrics = {
        "split": args.split,
        "episodes": int(args.episodes),
        "n_way": int(args.n_way),
        "k_shot": int(args.k_shot),
        "q_query_cap_per_class": int(args.q_query),
        "num_gallery": int(len(yG)),
        "num_query_pool": int(len(yQ)),
        "episodic_acc_mean": float(np.mean(episodic_acc)) if episodic_acc else 0.0,
        "episodic_acc_std": float(np.std(episodic_acc)) if episodic_acc else 0.0,
        "recall@1": float(recall1_total / nQ) if nQ > 0 else 0.0,
        "recall@5": float(recall5_total / nQ) if nQ > 0 else 0.0,
        "mAP": float(np.mean(aps)) if aps else 0.0,
        "num_eval_queries_total": int(nQ),
    }

    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()