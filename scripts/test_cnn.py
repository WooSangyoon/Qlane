import os
import random
from pathlib import Path
import sys
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from src.data import LaneDataset
from src.models.cnn import SimpleCNN

sys.path.append(str(Path(__file__).resolve().parents[1]))


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_checkpoint_strict(model: torch.nn.Module, ckpt_path: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location="cpu")

    if isinstance(ckpt, dict) and all(isinstance(k, str) for k in ckpt.keys()) and any(
        k.endswith("weight") or k.endswith("bias") for k in ckpt.keys()
    ):
        state_dict = ckpt

    elif isinstance(ckpt, dict) and "model" in ckpt and isinstance(ckpt["model"], dict):
        state_dict = ckpt["model"]

    elif isinstance(ckpt, dict) and "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
        state_dict = ckpt["state_dict"]

    elif isinstance(ckpt, dict) and "model_state_dict" in ckpt and isinstance(ckpt["model_state_dict"], dict):
        state_dict = ckpt["model_state_dict"]
    else:
        raise ValueError(f"Unrecognized checkpoint format: {type(ckpt)} keys={getattr(ckpt,'keys',lambda:[])()}")



    if any(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()


def make_dataset(resize_hw=(360, 640)) -> LaneDataset:
    return LaneDataset(resize_hw=resize_hw)


def overlay_points(img_bgr, x_table_norm, mask, y_resized, color, r=3):
    H, W = img_bgr.shape[:2]
    out = img_bgr.copy()
    K, R = x_table_norm.shape

    for k in range(K):
        for j in range(R):
            if mask[k, j] < 0.5:
                continue
            x_px = int(round(float(x_table_norm[k, j]) * W))
            y_px = int(round(float(y_resized[j])))
            if 0 <= x_px < W and 0 <= y_px < H:
                cv2.circle(out, (x_px, y_px), r, color, -1, lineType=cv2.LINE_AA)
    return out


@torch.no_grad()
def visualize_one(model, ds, idx, device):
    sample = ds[idx]

    img_t = sample["image"].unsqueeze(0).to(device)
    pred = model(img_t)[0].detach().cpu().numpy()

    gt = sample["x_table"].detach().cpu().numpy()
    mask = sample["mask_table"].detach().cpu().numpy()

    resize_h, resize_w = sample["resize_hw"]
    orig_h, orig_w = sample["orig_hw"]
    h_samples = np.asarray(sample["h_samples"], dtype=np.float32)
    y_resized = h_samples * (float(resize_h) / float(orig_h))

    img = sample["image"].detach().cpu().numpy()
    img = np.transpose(img, (1, 2, 0))

    img = np.clip(img * 255.0, 0, 255).astype(np.uint8)

    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    vis = overlay_points(img_bgr, gt, mask, y_resized, color=(0, 0, 255), r=3)
    vis = overlay_points(vis, pred, mask, y_resized, color=(255, 0, 0), r=3)

    title = f"idx={idx} | raw_file={sample.get('raw_file', '')}"
    return vis, title


def main():
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    device = get_device()
    print("device:", device)

    ds = make_dataset(resize_hw=(360, 640))

    model = SimpleCNN(num_lanes=4, num_rows=56)
    load_checkpoint_strict(model, "outputs/simplecnn.pt", device)

    while True:
        idx = random.randrange(len(ds))
        vis_bgr, title = visualize_one(model, ds, idx, device)

        vis_rgb = cv2.cvtColor(vis_bgr, cv2.COLOR_BGR2RGB)

        plt.figure(figsize=(10, 5))
        plt.imshow(vis_rgb)
        plt.title(title)
        plt.axis("off")

        plt.show() 


if __name__ == "__main__":
    main()
