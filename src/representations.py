import os
import numpy as np
from src.datasets import get_dataset_root, iter_records, get_lane_points

LABEL_FILE = "test_label_new.json"
MAX_LANES = 4
MISSING_X = -1


def build_tables_from_record(rec, *, max_lanes=MAX_LANES, missing_x=MISSING_X):

    h_samples = rec.get("h_samples", [])
    if not h_samples:
        return None, None

    H = len(h_samples)
    y_to_idx = {y: i for i, y in enumerate(h_samples)}

    lane_points = get_lane_points(rec) 
    if not lane_points:
        return None, None

    lanes = lane_points[:max_lanes]
    K = max_lanes

    x_table = np.full((K, H), missing_x, dtype=int)
    mask_table = np.zeros((K, H), dtype=int)

    for lane_i, pts in enumerate(lanes):
        for x, y in pts:
            j = y_to_idx.get(y)
            if j is None:
                continue
            x_table[lane_i, j] = int(x)
            mask_table[lane_i, j] = 1

    return x_table.tolist(), mask_table.tolist()


def get_table():
    root = get_dataset_root()
    label_path = os.path.join(root, LABEL_FILE)

    for rec in iter_records(label_path):
        x_table, mask_table = build_tables_from_record(rec)
        if x_table is None:
            continue
        return x_table, mask_table

    return None, None


def main():
    x_table, mask_table = get_table()
    # print("X Table:", x_table)
    # print("Mask Table:", mask_table)


if __name__ == "__main__":
    main()
