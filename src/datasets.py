import os
import json
import kagglehub

DATASET_HANDLE = "manideep1108/tusimple"

def get_dataset_root():
    dataset_dir = kagglehub.dataset_download(DATASET_HANDLE)
    return dataset_dir

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


def get_lane_points(rec):
    lanes = rec.get("lanes", [])
    h_samples = rec.get("h_samples", [])
    if not lanes or not h_samples: 
        return []
    
    laneY = h_samples

    lane_points = []
    
    for laneX in lanes:
        points = []
        for x, y in zip(laneX, laneY):
            if x is None or x < 0:
                continue
            points.append((x, y))
        if points:
            lane_points.append(points)

    return lane_points