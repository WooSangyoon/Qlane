import os
import json
import cv2
import matplotlib.pyplot as plt
from src.datasets import get_dataset_root

LABEL_FILE = "test_label_new.json"

def iter_records(label_path):
    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)

def find_image_path(root, raw_file):
    candidates = [
        root,
        os.path.join(root, "TUSimple"),
        os.path.join(root, "TUSimple", "train_set"),
        os.path.join(root, "TUSimple", "test_set"),
    ]

    for base in candidates:
        p = os.path.join(base, raw_file)
        if os.path.exists(p):
            return p
    return None



def overlay_gt_points(ax, rec, *, s=5):
    lanes = rec.get("lanes", [])
    h_samples = rec.get("h_samples", [])
    if not lanes or not h_samples: 
        return
    
    laneY = h_samples

    for laneX in lanes:
        for x, y in zip(laneX, laneY):
            if x<0 or x is None:
                continue

            ax.scatter(x, y, s=s, c="red", marker="o")



def main():
    root = get_dataset_root()
    label_path = os.path.join(root, LABEL_FILE)

    for rec in iter_records(label_path):
        raw_file = rec.get("raw_file")
        if not raw_file:
            continue

        img_path = find_image_path(root, raw_file)
        if img_path is None:
            continue

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        plt.figure(figsize=(10, 6))
        plt.imshow(img, alpha=0.7)
        overlay_gt_points(plt, rec, s=5)
        plt.show()

        return
        

if __name__ == "__main__":
    main()
