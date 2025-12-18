import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from config import Config
from utils.augment import augment_batch

class S3DISDataset(Dataset):
    def __init__(self, root, split="train", area="Area_5", num_points=4096, augment=False):
        self.root = root
        self.split = split
        self.area = area
        self.num_points = num_points
        self.augment = augment
        self.data_list = []
        self.labels_list = []

        area_path = os.path.join(root, area)
        if not os.path.exists(area_path):
            raise ValueError(f"Area {area} not found in {root}")

        for room in os.listdir(area_path):
            room_path = os.path.join(area_path, room)
            if os.path.isdir(room_path):
                for block_file in os.listdir(room_path):
                    if block_file.endswith('.npy') or block_file.endswith('.txt'):
                        path = os.path.join(room_path, block_file)
                        data = self._load_data(path)
                        if data is not None:
                            self.data_list.append(data[:, :6])      # xyzrgb
                            self.labels_list.append(data[:, -1])    # label

    def _load_data(self, path):
        try:
            if path.endswith('.npy'):
                data = np.load(path)
            else:
                data = np.loadtxt(path).astype(np.float32)
            if data.shape[1] < 7:
                return None
            return data
        except Exception as e:
            print(f"Error loading {path}: {e}")
            return None

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        points = self.data_list[idx].copy()  # (N_all, 6)
        labels = self.labels_list[idx].copy().astype(np.int64)  # (N_all,)

        N = points.shape[0]
        if N >= self.num_points:
            choice = np.random.choice(N, self.num_points, replace=False)
        else:
            choice = np.random.choice(N, self.num_points, replace=True)
        points = points[choice]
        labels = labels[choice]

        centroid = np.mean(points[:, :3], axis=0)
        points[:, :3] -= centroid
        #  нормализуем цвет [0,1]
        points[:, 3:] /= 255.0

        if self.augment and self.split == "train":
            points = augment_batch(points[np.newaxis], Config())[0]

        return torch.from_numpy(points).float(), torch.from_numpy(labels).long()

def get_dataloaders(config):
    train_datasets = []
    for area in config.TRAIN_AREAS:
        ds = S3DISDataset(config.DATA_ROOT, split="train", area=area,
                          num_points=config.NUM_POINTS, augment=config.AUGMENT)
        train_datasets.append(ds)
    train_ds = torch.utils.data.ConcatDataset(train_datasets)

    val_ds = S3DISDataset(config.DATA_ROOT, split="val", area=config.VAL_AREA,
                          num_points=config.NUM_POINTS, augment=False)
    test_ds = S3DISDataset(config.DATA_ROOT, split="test", area=config.TEST_AREA,
                           num_points=config.NUM_POINTS, augment=False)

    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE,
                              shuffle=True, num_workers=config.NUM_WORKERS)
    val_loader = DataLoader(val_ds, batch_size=config.BATCH_SIZE,
                            shuffle=False, num_workers=config.NUM_WORKERS)
    test_loader = DataLoader(test_ds, batch_size=config.BATCH_SIZE,
                             shuffle=False, num_workers=config.NUM_WORKERS)

    return train_loader, val_loader, test_loader