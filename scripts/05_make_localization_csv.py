from pathlib import Path
import pandas as pd

INSTANCES = Path("data/processed/instances.csv")
NOVEL_TEST_CLASSES = Path("data/processed/splits/novel_test_classes.txt")
OUT_DIR = Path("data/processed/localization")
OUT_CSV = OUT_DIR / "novel_test_instances.csv"

def main():
    assert INSTANCES.exists(), f"Missing {INSTANCES}"
    assert NOVEL_TEST_CLASSES.exists(), f"Missing {NOVEL_TEST_CLASSES}"

    classes = [l.strip() for l in NOVEL_TEST_CLASSES.read_text(encoding="utf-8").splitlines() if l.strip()]
    df = pd.read_csv(INSTANCES)

    needed = {"instance_id","image_path","class_name","x1","y1","x2","y2"}
    missing = needed - set(df.columns)
    assert not missing, f"instances.csv missing columns: {missing}"

    df = df[df["class_name"].isin(classes)].copy()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    print("Wrote:", OUT_CSV)
    print("Rows:", len(df))
    print("Unique classes:", df["class_name"].nunique())

if __name__ == "__main__":
    main()