from pathlib import Path
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

class LogoPatchDataset(Dataset):
    def __init__(self, csv_path: str, class_to_idx: dict, transform=None):
        self.csv_path = Path(csv_path)
        self.df = pd.read_csv(self.csv_path)
        self.class_to_idx = class_to_idx
        self.transform = transform

        assert "patch_path" in self.df.columns
        assert "class_name" in self.df.columns

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i: int):
        row = self.df.iloc[i]
        img = Image.open(Path(row["patch_path"])).convert("RGB")
        y = self.class_to_idx[row["class_name"]]
        instance_id = row["instance_id"] if "instance_id" in self.df.columns else str(i)

        if self.transform is not None:
            img = self.transform(img)

        return img, y, instance_id