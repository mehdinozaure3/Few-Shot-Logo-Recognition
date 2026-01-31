import json
import random
from pathlib import Path

import pandas as pd

# =========================
# Config 
# =========================
SEED = 42

PATCHES_CSV = Path("data/processed/patches.csv")
SPLITS_DIR = Path("data/processed/splits")

# Keep the project manageable (proposal: 50–100 classes)
MAX_TOTAL_CLASSES = 100          
MIN_PATCHES_PER_CLASS = 40       

# Class-level splits
BASE_FRACTION = 0.80             
NOVEL_VAL_FRACTION = 0.50        

# Instance-level splits for base classes
BASE_TRAIN_FRACTION = 0.70
BASE_VAL_FRACTION = 0.15
BASE_TEST_FRACTION = 0.15        

# Instance-level splits inside novel_val/novel_test for retrieval protocol
NOVEL_GALLERY_FRACTION = 0.70    

# Safety: minimum per-class required in each split
MIN_BASE_TRAIN_PER_CLASS = 10    
MIN_NOVEL_GALLERY_PER_CLASS = 5
MIN_NOVEL_QUERY_PER_CLASS = 2

# =========================


def set_seed(seed: int):
    random.seed(seed)


def write_list(path: Path, items):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for x in items:
            f.write(str(x) + "\n")


def stratified_instance_split(df_cls: pd.DataFrame, fractions, rng: random.Random):
    """
    Split a single-class dataframe into multiple parts by shuffled indices.
    fractions: list of floats summing to 1.0, e.g., [0.7, 0.15, 0.15]
    returns: list of dataframes in same order as fractions
    """
    idx = df_cls.index.tolist()
    rng.shuffle(idx)
    n = len(idx)

    cuts = []
    acc = 0
    for frac in fractions[:-1]:
        acc += int(round(frac * n))
        cuts.append(acc)

    parts = []
    start = 0
    for cut in cuts:
        parts.append(df_cls.loc[idx[start:cut]])
        start = cut
    parts.append(df_cls.loc[idx[start:]])

    return parts


def main():
    assert PATCHES_CSV.exists(), f"Missing {PATCHES_CSV}. Run scripts/01_build_patches.py first."

    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)

    df = pd.read_csv(PATCHES_CSV)

    # Basic sanity
    required_cols = {"instance_id", "class_name", "patch_path"}
    missing = required_cols - set(df.columns)
    assert not missing, f"patches.csv missing columns: {missing}"

    # Choose eligible classes (enough patches)
    counts = df["class_name"].value_counts()
    eligible_classes = counts[counts >= MIN_PATCHES_PER_CLASS].index.tolist()

    if len(eligible_classes) < MAX_TOTAL_CLASSES:
        print(f"[WARN] Only {len(eligible_classes)} classes meet MIN_PATCHES_PER_CLASS={MIN_PATCHES_PER_CLASS}.")
        selected_classes = eligible_classes
    else:
        selected_classes = rng.sample(eligible_classes, k=MAX_TOTAL_CLASSES)

    selected_classes = sorted(selected_classes)
    df = df[df["class_name"].isin(selected_classes)].copy()

    # Split classes into base vs novel (class-level)
    n_total = len(selected_classes)
    n_base = int(round(BASE_FRACTION * n_total))
    base_classes = selected_classes[:]
    rng.shuffle(base_classes)
    base_classes = sorted(base_classes[:n_base])
    novel_classes = sorted(list(set(selected_classes) - set(base_classes)))

    # Split novel classes into novel_val and novel_test (class-level)
    n_novel = len(novel_classes)
    n_novel_val = int(round(NOVEL_VAL_FRACTION * n_novel))
    novel_shuffled = novel_classes[:]
    rng.shuffle(novel_shuffled)
    novel_val_classes = sorted(novel_shuffled[:n_novel_val])
    novel_test_classes = sorted(novel_shuffled[n_novel_val:])

    # Save class lists
    write_list(SPLITS_DIR / "selected_classes.txt", selected_classes)
    write_list(SPLITS_DIR / "base_classes.txt", base_classes)
    write_list(SPLITS_DIR / "novel_classes.txt", novel_classes)
    write_list(SPLITS_DIR / "novel_val_classes.txt", novel_val_classes)
    write_list(SPLITS_DIR / "novel_test_classes.txt", novel_test_classes)

    # Helper to filter df by class list
    def df_for_classes(classes):
        return df[df["class_name"].isin(classes)].copy()

    df_base = df_for_classes(base_classes)
    df_novel_val = df_for_classes(novel_val_classes)
    df_novel_test = df_for_classes(novel_test_classes)

    # Base: instance-level train/val/test
    base_train_parts = []
    base_val_parts = []
    base_test_parts = []

    for cls in base_classes:
        df_cls = df_base[df_base["class_name"] == cls]
        train_df, val_df, test_df = stratified_instance_split(
            df_cls,
            [BASE_TRAIN_FRACTION, BASE_VAL_FRACTION, BASE_TEST_FRACTION],
            rng
        )

        if len(train_df) < MIN_BASE_TRAIN_PER_CLASS:
            print(f"[WARN] base class '{cls}' has only {len(train_df)} train samples after split.")

        base_train_parts.append(train_df)
        base_val_parts.append(val_df)
        base_test_parts.append(test_df)

    df_base_train = pd.concat(base_train_parts, ignore_index=True)
    df_base_val = pd.concat(base_val_parts, ignore_index=True)
    df_base_test = pd.concat(base_test_parts, ignore_index=True)

    df_base_train.to_csv(SPLITS_DIR / "base_train.csv", index=False)
    df_base_val.to_csv(SPLITS_DIR / "base_val.csv", index=False)
    df_base_test.to_csv(SPLITS_DIR / "base_test.csv", index=False)

    # Novel (val/test): instance-level gallery/query
    def split_novel_gallery_query(df_novel: pd.DataFrame, split_name: str):
        gallery_parts = []
        query_parts = []

        classes = sorted(df_novel["class_name"].unique().tolist())
        for cls in classes:
            df_cls = df_novel[df_novel["class_name"] == cls]
            gallery_df, query_df = stratified_instance_split(
                df_cls,
                [NOVEL_GALLERY_FRACTION, 1.0 - NOVEL_GALLERY_FRACTION],
                rng
            )

            if len(gallery_df) < MIN_NOVEL_GALLERY_PER_CLASS or len(query_df) < MIN_NOVEL_QUERY_PER_CLASS:
                print(
                    f"[WARN] novel {split_name} class '{cls}' too small after gallery/query split: "
                    f"gallery={len(gallery_df)} query={len(query_df)}"
                )

            gallery_parts.append(gallery_df)
            query_parts.append(query_df)

        df_gallery = pd.concat(gallery_parts, ignore_index=True)
        df_query = pd.concat(query_parts, ignore_index=True)

        df_gallery.to_csv(SPLITS_DIR / f"{split_name}_gallery.csv", index=False)
        df_query.to_csv(SPLITS_DIR / f"{split_name}_query.csv", index=False)

        return df_gallery, df_query

    df_novel_val_gallery, df_novel_val_query = split_novel_gallery_query(df_novel_val, "novel_val")
    df_novel_test_gallery, df_novel_test_query = split_novel_gallery_query(df_novel_test, "novel_test")

    # Summary stats (useful for paper)
    summary = {
        "seed": SEED,
        "min_patches_per_class": MIN_PATCHES_PER_CLASS,
        "max_total_classes": MAX_TOTAL_CLASSES,
        "selected": {
            "num_classes": len(selected_classes),
            "num_instances": int(len(df)),
        },
        "base": {
            "num_classes": len(base_classes),
            "train_instances": int(len(df_base_train)),
            "val_instances": int(len(df_base_val)),
            "test_instances": int(len(df_base_test)),
        },
        "novel": {
            "num_classes": len(novel_classes),
            "val": {
                "num_classes": len(novel_val_classes),
                "gallery_instances": int(len(df_novel_val_gallery)),
                "query_instances": int(len(df_novel_val_query)),
            },
            "test": {
                "num_classes": len(novel_test_classes),
                "gallery_instances": int(len(df_novel_test_gallery)),
                "query_instances": int(len(df_novel_test_query)),
            },
        },
    }

    with (SPLITS_DIR / "split_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # ensure no class leakage
    base_set = set(base_classes)
    novel_val_set = set(novel_val_classes)
    novel_test_set = set(novel_test_classes)

    assert base_set.isdisjoint(novel_val_set)
    assert base_set.isdisjoint(novel_test_set)
    assert novel_val_set.isdisjoint(novel_test_set)

    print("\n Done. Wrote splits to:", SPLITS_DIR)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    set_seed(SEED)
    main()