import os
import kagglehub
from kagglehub import KaggleDatasetAdapter

DATASET_HANDLE = "manideep1108/tusimple"
LABEL_FILE = "test_label_new.json"

def get_dataset_root():
    dataset_dir = kagglehub.dataset_download(DATASET_HANDLE)
    return dataset_dir

def load_labels():
    df = kagglehub.load_dataset(
        KaggleDatasetAdapter.PANDAS,
        DATASET_HANDLE,
        LABEL_FILE,
    )
    return df