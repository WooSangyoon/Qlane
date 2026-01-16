import os
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import cv2

from src.data import LaneDataset
from src.models.cnn import SimpleCNN
from src.losses import masked_mse_loss


def get_device():
    # Prefer CUDA if present, else MPS on macOS, else CPU
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def preview_one_sample(
    model,
    ds,
    device,
    *,
    idx=0,
    num_lanes=4,
    num_rows=56,
):
    """
    Preview one sample: GT (red) vs Prediction (blue).
    Assumes x_table and model outputs are normalized in [0,1] w.r.t resized width.
    Uses true TuSimple h_samples (original y) scaled to resized height.
    """
    model.eval()

    sample = ds[idx]
    image_tensor = sample["image"].unsqueeze(0).to(device)  # (1,3,H,W)
    x_gt = sample["x_table"]          # (K,H_grid), normalized [0,1]
    mask = sample["mask_table"]       # (K,H_grid), 0/1
    h_samples = sample["h_samples"]   # list length H_grid (original y)
    orig_h, orig_w = sample["orig_hw"]
    resize_h, resize_w = sample["resize_hw"]
    image_path = sample["image_path"]

    if not h_samples or len(h_samples) != num_rows:
        raise ValueError(f"h_samples is missing or has wrong length. got={len(h_samples)} expected={num_rows}")

    with torch.no_grad():
        pred = model(image_tensor).squeeze(0).cpu()  # (K,H_grid), normalized [0,1] expected

    # Load and resize image for display
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Failed to read image: {image_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (resize_w, resize_h))

    # Convert y from original coord to resized coord
    sy = float(resize_h) / float(orig_h)
    ys_resized = [y * sy for y in h_samples]

    # Convert normalized x to resized pixel x for plotting
    x_gt_np = x_gt.cpu().numpy() * float(resize_w)
    pred_np = pred.cpu().numpy() * float(resize_w)
    mask_np = mask.cpu().numpy()

    # Optional: clip predictions for nicer plots
    pred_np = pred_np.clip(0.0, float(resize_w))

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(img, alpha=0.95)
    ax.axis("off")
    ax.set_title("GT (red) vs Prediction (blue)")

    # GT points (red)
    for i in range(num_lanes):
        for j in range(num_rows):
            if mask_np[i, j] == 1:
                ax.scatter(x_gt_np[i, j], ys_resized[j], c="red", s=12, alpha=0.9)

    # Prediction points (blue) -- plot only where GT exists for direct comparison
    for i in range(num_lanes):
        for j in range(num_rows):
            if mask_np[i, j] == 1:
                ax.scatter(pred_np[i, j], ys_resized[j], c="blue", s=12, alpha=0.6)

    plt.show()


def main():
    batch_size = 8
    num_epochs = 50
    lr = 1e-3
    num_workers = 0

    device = get_device()
    print("Using device:", device)

    ds = LaneDataset(resize_hw=(360, 640))
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    model = SimpleCNN(num_lanes=4, num_rows=56).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Sanity: one forward
    batch0 = next(iter(loader))
    with torch.no_grad():
        pred0 = model(batch0["image"].to(device))
    print("sanity pred shape:", tuple(pred0.shape), "gt shape:", tuple(batch0["x_table"].shape))

    # Train
    model.train()
    for epoch in range(1, num_epochs + 1):
        running_loss = 0.0
        n_batches = 0

        for batch in loader:
            images = batch["image"].to(device)
            x_table = batch["x_table"].to(device)
            mask_table = batch["mask_table"].to(device)

            pred = model(images)
            loss = masked_mse_loss(pred, x_table, mask_table)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            running_loss += float(loss.item())
            n_batches += 1

        avg_loss = running_loss / max(n_batches, 1)
        print(f"Epoch {epoch}/{num_epochs}, Loss: {avg_loss:.6f}")

    # Save checkpoint
    os.makedirs("outputs", exist_ok=True)
    ckpt_path = "outputs/simplecnn.pt"
    torch.save({"model_state_dict": model.state_dict()}, ckpt_path)
    print("saved:", ckpt_path)

    # Preview one sample
    preview_one_sample(
        model,
        ds,
        device,
        idx=0,
        num_lanes=4,
        num_rows=56,
    )


if __name__ == "__main__":
    main()
