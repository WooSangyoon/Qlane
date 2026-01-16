import os
import numpy as np
from src.datasets import get_dataset_root, iter_records, get_lane_points
from src.constants import LABEL_FILE, MAX_LANES, MISSING_X

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


def get_table(rec, *, max_lanes=MAX_LANES, missing_x=MISSING_X):
    """
    rec -> (x_table, mask_table)
    x_table: (K, H)  (결측은 missing_x)
    mask_table: (K, H) (유효=1, 결측=0)
    """
    lanes = rec.get("lanes", [])
    h_samples = rec.get("h_samples", [])
    if not lanes or not h_samples:
        K = max_lanes
        H = len(h_samples) if h_samples else 0
        x_table = np.full((K, H), missing_x, dtype=int).tolist()
        mask_table = np.zeros((K, H), dtype=int).tolist()
        return x_table, mask_table

    H = len(h_samples)
    K = max_lanes

    # TuSimple lanes는 lane별 x 리스트이며, 길이는 H와 동일하다고 가정(대부분 동일)
    x_table = np.full((K, H), missing_x, dtype=int)
    mask_table = np.zeros((K, H), dtype=int)

    for i, laneX in enumerate(lanes[:K]):
        for j in range(min(H, len(laneX))):
            x = laneX[j]
            if x is None or x < 0:
                continue
            x_table[i, j] = int(x)
            mask_table[i, j] = 1

    return x_table.tolist(), mask_table.tolist()


def main():
    x_table, mask_table = get_table()
    # print("X Table:", x_table)
    # print("Mask Table:", mask_table)


if __name__ == "__main__":
    main()
