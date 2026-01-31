from pathlib import Path

ROOT = Path("data/raw/logodet3k")

def main():
    assert ROOT.exists(), f"Dataset not found: {ROOT}"
    xmls = list(ROOT.rglob("*.xml"))
    jsons = list(ROOT.rglob("*.json"))
    txts = list(ROOT.rglob("*.txt"))
    imgs = []
    for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp"]:
        imgs.extend(ROOT.rglob(ext))

    print("Root:", ROOT)
    print("Images:", len(imgs))
    print("XML:", len(xmls))
    print("JSON:", len(jsons))
    print("TXT:", len(txts))

    if xmls[:5]:
        print("\nSample XML paths:")
        for p in xmls[:5]:
            print(" -", p)

    if jsons[:5]:
        print("\nSample JSON paths:")
        for p in jsons[:5]:
            print(" -", p)

    if txts[:5]:
        print("\nSample TXT paths:")
        for p in txts[:5]:
            print(" -", p)

if __name__ == "__main__":
    main()