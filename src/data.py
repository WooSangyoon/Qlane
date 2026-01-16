import os
import torch
import torch.utils.data as data
from PIL import Image
import torchvision.transforms as transforms

from src.datasets import iter_records, find_image_path, get_dataset_root
from src.constants import LABEL_FILE
from src.representations import get_table


class LaneDataset(data.Dataset):
    """
    Returns a dict per sample:
      - image: FloatTensor (3, H, W) resized
      - x_table: FloatTensor (K, H_grid) normalized to [0,1] in resized coordinate system
      - mask_table: FloatTensor (K, H_grid) in {0,1}
      - h_samples: list[int] original y coordinates (len=H_grid)
      - orig_hw: (H0, W0) original image size
      - resize_hw: (H, W) resized image size
      - raw_file: str
      - image_path: str
    """
    def __init__(self, *, resize_hw=(360, 640)):
        self.dataset_root = get_dataset_root()
        self.label_path = os.path.join(self.dataset_root, LABEL_FILE)
        self.records = list(iter_records(self.label_path))

        self.resize_hw = resize_hw  # (H, W)

        self.transform = transforms.Compose([
            transforms.Resize(self.resize_hw),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]

        raw_file = rec.get("raw_file")
        if not raw_file:
            raise ValueError(f"Record {idx} has no raw_file.")

        image_path = find_image_path(self.dataset_root, raw_file)
        if image_path is None:
            raise FileNotFoundError(f"Image not found for raw_file={raw_file}")

        # Load original image (to get original size), and resized tensor for model input
        image = Image.open(image_path).convert("RGB")
        orig_w, orig_h = image.size  # PIL: (W, H)
        image_tensor = self.transform(image)  # (3, H, W) where (H,W)=resize_hw

        # Encode label tables (these are in original pixel coordinate system)
        x_table, mask_table = get_table(rec)
        x_table_tensor = torch.tensor(x_table, dtype=torch.float32)        # (K, H_grid), original x pixels
        mask_table_tensor = torch.tensor(mask_table, dtype=torch.float32)  # (K, H_grid), 0/1

        # Convert x to resized coordinate system and normalize to [0,1]
        resize_h, resize_w = self.resize_hw
        sx = float(resize_w) / float(orig_w)  # x scaling: original -> resized

        # Only meaningful where mask=1, but applying uniformly is fine since mask will gate loss
        x_resized = x_table_tensor * sx
        x_norm = x_resized / float(resize_w)  # -> [0,1] scale

        sample = {
            "image": image_tensor,
            "x_table": x_norm,
            "mask_table": mask_table_tensor,
            "h_samples": rec.get("h_samples", []),
            "orig_hw": (orig_h, orig_w),
            "resize_hw": self.resize_hw,
            "raw_file": raw_file,
            "image_path": image_path,
        }
        return sample


def main():
    ds = LaneDataset(resize_hw=(360, 640))
    sample = ds[0]
    print("image:", tuple(sample["image"].shape))
    print("x_table:", tuple(sample["x_table"].shape), "min/max:", float(sample["x_table"].min()), float(sample["x_table"].max()))
    print("mask_table:", tuple(sample["mask_table"].shape), "mask_sum:", float(sample["mask_table"].sum()))
    print("orig_hw:", sample["orig_hw"], "resize_hw:", sample["resize_hw"])
    print("raw_file:", sample["raw_file"])
    print("image_path:", sample["image_path"])


if __name__ == "__main__":
    main()
