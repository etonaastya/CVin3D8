import os
import numpy as np
import open3d as o3d
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import h5py

class ModelNetDataset(Dataset):
    def __init__(self, split='train', cache_path="modelnet10_1024.h5"):
        if not os.path.exists(cache_path):
            raise FileNotFoundError(
                f"Кэш {cache_path} не найден. Сначала запустите dataset.py для его создания."
            )
        with h5py.File(cache_path, 'r') as f:
            self.points = f[f'{split}_points'][:]
            self.labels = f[f'{split}_labels'][:]
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.points[idx], self.labels[idx]


def load_off(path):
    with open(path, 'r') as f:
        first_line = f.readline().strip()
        if first_line.startswith('OFF'):
            line = f.readline().strip()
        else:
            line = first_line
        n_verts, n_faces, _ = map(int, line.split()[:3])
        verts = [list(map(float, f.readline().strip().split())) for _ in range(n_verts)]
        _ = [f.readline() for _ in range(n_faces)]
        return np.array(verts, dtype=np.float32)


def sample_pointcloud(verts, n_points=1024):
    if len(verts) == 0:
        return np.zeros((n_points, 3), dtype=np.float32)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(verts)
    pcd = pcd.voxel_down_sample(voxel_size=0.02)
    points = np.asarray(pcd.points)
    if len(points) < n_points:
        choice = np.random.choice(len(points), n_points, replace=True)
    else:
        choice = np.random.choice(len(points), n_points, replace=False)
    points = points[choice]
    centroid = np.mean(points, axis=0)
    points -= centroid
    dist = np.max(np.sqrt(np.sum(points**2, axis=1)))
    if dist > 1e-8:
        points /= dist
    return points.astype(np.float32)


def preprocess_and_cache():
    dataset = "C:/Users/anast/ml/CVin3D/ModelNet10"
    n_points = 1024
    cache = "modelnet10_1024.h5"
    
    print("Preprocessing ModelNet10 → HDF5 cache...")
    classes = sorted([
        d for d in os.listdir(dataset)
        if os.path.isdir(os.path.join(dataset, d)) and not d.startswith('.')
    ])
    class_to_idx = {cls: i for i, cls in enumerate(classes)}

    def process_dir(split):
        all_points, all_labels = [], []
        print(f"Processing {split} split...")
        for cls in classes:
            cls_path = os.path.join(dataset, cls, split)
            for fname in tqdm(os.listdir(cls_path), desc=cls, leave=False):
                if not fname.endswith('.off'): continue
                verts = load_off(os.path.join(cls_path, fname))
                points = sample_pointcloud(verts, n_points)
                all_points.append(points)
                all_labels.append(class_to_idx[cls])
        return np.stack(all_points), np.array(all_labels)

    train_points, train_labels = process_dir("train")
    test_points, test_labels = process_dir("test")

    with h5py.File(cache, 'w') as f:
        f.create_dataset('train_points', data=train_points)
        f.create_dataset('train_labels', data=train_labels)
        f.create_dataset('test_points', data=test_points)
        f.create_dataset('test_labels', data=test_labels)
        f.create_dataset('classes', data=[cls.encode() for cls in classes])
    print(f"Saved to {cache}")


if __name__ == "__main__":
    preprocess_and_cache()