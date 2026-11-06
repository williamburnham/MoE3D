import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import glob

MAX_LEN = 196
CLASSES = {"Hlth": 0, "Strk": 1}

def parse(path):
    """
    Parse the label, id, action and clip from the file path.
    """
    name = os.path.basename(path)[:-4]
    # Cut first character from mirrored clip names
    if name.startswith("M"):
        name = name[1:]
    dsid, cls, action, pid, clip = name.split("-")
    return CLASSES.get(cls), f"{dsid}-{pid}", action, clip  # label, id, action, clip 

class MotionDataset(Dataset):

    def __init__(self, files, mean, std):
        self.files = files # all file paths
        self.mean = mean
        self.std = std

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        x = np.load(self.files[idx])

        # Normalise
        x = (x - self.mean) / self.std

        # Pad or truncate to MAX_LEN
        T = min(len(x), MAX_LEN)
        out = np.zeros((MAX_LEN, x.shape[1]), np.float32)
        out[:T] = x[:T]
        pad_mask = np.ones(MAX_LEN, bool); pad_mask[:T] = False  # True = padding
        label, _, _, _ = parse(self.files[idx])

        return torch.from_numpy(out), torch.from_numpy(pad_mask), label