import csv
from pathlib import Path
from PIL import Image
from tqdm import tqdm

INSTANCES_CSV = Path("data/processed/instances.csv")
PATCHES_ROOT = Path("data/processed/patches")
OUT_CSV = Path("data/processed/patches.csv")

MIN_SIZE = 16  # minimum width or height in pixels

def clamp(val, minv, maxv):
    return max(minv, min(val, maxv))

def main():
    assert INSTANCES_CSV.exists(), "instances.csv not found"

    PATCHES_ROOT.mkdir(parents=True, exist_ok=True)

    rows_out = []
    skipped_small = 0
    skipped_error = 0

    with INSTANCES_CSV.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    for r in tqdm(rows, desc="Cropping patches"):
        try:
            img_path = Path(r["image_path"])
            img = Image.open(img_path).convert("RGB")

            w_img, h_img = img.size
            x1 = clamp(int(r["x1"]), 0, w_img - 1)
            y1 = clamp(int(r["y1"]), 0, h_img - 1)
            x2 = clamp(int(r["x2"]), x1 + 1, w_img)
            y2 = clamp(int(r["y2"]), y1 + 1, h_img)

            pw = x2 - x1
            ph = y2 - y1

            if pw < MIN_SIZE or ph < MIN_SIZE:
                skipped_small += 1
                continue

            patch = img.crop((x1, y1, x2, y2))

            class_dir = PATCHES_ROOT / r["class_name"]
            class_dir.mkdir(parents=True, exist_ok=True)

            patch_path = class_dir / f"{r['instance_id']}.jpg"
            patch.save(patch_path, quality=95)

            rows_out.append({
                "instance_id": r["instance_id"],
                "class_name": r["class_name"],
                "patch_path": patch_path.as_posix(),
                "orig_image": r["image_path"],
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "patch_width": pw,
                "patch_height": ph
            })

        except Exception:
            skipped_error += 1

    # write csv
    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(rows_out[0].keys())
        )
        writer.writeheader()
        writer.writerows(rows_out)

    print("\nDone.")
    print("Total patches:", len(rows_out))
    print("Skipped (too small):", skipped_small)
    print("Skipped (errors):", skipped_error)
    print("Wrote:", OUT_CSV)

if __name__ == "__main__":
    main()