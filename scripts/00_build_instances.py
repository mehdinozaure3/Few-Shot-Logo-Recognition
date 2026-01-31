import csv
import hashlib
from pathlib import Path
import xml.etree.ElementTree as ET

DATA_ROOT = Path("data/raw/logodet3k")
OUT_CSV = Path("data/processed/instances.csv")

IMG_EXTS = [".jpg", ".jpeg", ".png", ".bmp"]

def sha1_id(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:16]

def find_image_for_xml(xml_path: Path, filename_in_xml: str | None) -> Path | None:
    if filename_in_xml:
        cand = (xml_path.parent / filename_in_xml)
        if cand.exists():
            return cand

        cand = (DATA_ROOT / filename_in_xml)
        if cand.exists():
            return cand


    for ext in IMG_EXTS:
        cand = xml_path.with_suffix(ext)
        if cand.exists():
            return cand


    stem = xml_path.stem
    for ext in IMG_EXTS:
        hits = list(xml_path.parent.rglob(stem + ext))
        if hits:
            return hits[0]
    return None

def parse_voc_xml(xml_path: Path):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    filename = root.findtext("filename")
    size = root.find("size")
    w = int(size.findtext("width")) if size is not None else None
    h = int(size.findtext("height")) if size is not None else None

    objects = []
    for obj in root.findall("object"):
        name = obj.findtext("name")
        bnd = obj.find("bndbox")
        if bnd is None:
            continue
        x1 = int(float(bnd.findtext("xmin")))
        y1 = int(float(bnd.findtext("ymin")))
        x2 = int(float(bnd.findtext("xmax")))
        y2 = int(float(bnd.findtext("ymax")))

        objects.append((name, x1, y1, x2, y2))
    return filename, w, h, objects

def main():
    assert DATA_ROOT.exists(), f"Dataset root not found: {DATA_ROOT}"
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    xml_paths = list(DATA_ROOT.rglob("*.xml"))
    print(f"Found {len(xml_paths)} xml files")

    rows = []
    bad_xml = 0
    missing_img = 0
    empty_objs = 0

    for xml_path in xml_paths:
        try:
            filename, w, h, objects = parse_voc_xml(xml_path)
        except Exception as e:
            bad_xml += 1
            continue

        if not objects:
            empty_objs += 1
            continue

        img_path = find_image_for_xml(xml_path, filename)
        if img_path is None:
            missing_img += 1
            continue


        try:
            rel = xml_path.relative_to(DATA_ROOT)
            root_category = rel.parts[0] if len(rel.parts) > 0 else ""
        except Exception:
            root_category = ""

        img_rel = img_path.as_posix()
        ann_rel = xml_path.as_posix()

        for (cls, x1, y1, x2, y2) in objects:
            
            if x2 <= x1 or y2 <= y1:
                continue
            inst_str = f"{ann_rel}|{cls}|{x1},{y1},{x2},{y2}"
            instance_id = sha1_id(inst_str)

            rows.append({
                "instance_id": instance_id,
                "image_path": img_rel,
                "ann_path": ann_rel,
                "root_category": root_category,
                "class_name": cls,
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "width": w if w is not None else "",
                "height": h if h is not None else "",
            })

    # write csv
    fieldnames = ["instance_id","image_path","ann_path","root_category","class_name","x1","y1","x2","y2","width","height"]
    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print("Wrote:", OUT_CSV)
    print("Rows (logo instances):", len(rows))
    print("Bad xml:", bad_xml)
    print("Missing image:", missing_img)
    print("Empty objects:", empty_objs)

if __name__ == "__main__":
    main()