import os
import cv2
import matplotlib.pyplot as plt
from src.datasets import (get_dataset_root, iter_records, find_image_path, get_lane_points)
from src.constants import LABEL_FILE

def overlay_points(ax, lane_points, *, s=5):

    for lane in lane_points:
        xs = [p[0] for p in lane]
        ys = [p[1] for p in lane]
        ax.scatter(xs, ys, s=s, c="red", marker="o")
        # ax.plot(xs, ys, linewidth=2)


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
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        lane_points = get_lane_points(rec)
        if not lane_points:
            continue

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.imshow(img, alpha=0.85)
        overlay_points(ax, lane_points, s=5)
        ax.axis("off")
        plt.show()
        return


if __name__ == "__main__":
    main()
