# evaluate.py
import os
import torch
import numpy as np
import open3d as o3d
from config import Config
from dataset import get_dataloaders
from model import PointNet2Seg
from utils.metrics import calculate_metrics
from utils.visualize import visualize_point_cloud

def evaluate():
    config = Config()
    _, _, test_loader = get_dataloaders(config)

    model = PointNet2Seg(num_classes=config.NUM_CLASSES, in_dim=6)
    model.load_state_dict(torch.load(os.path.join(config.CHECKPOINT_DIR, "best_model.pth")))
    model.eval()
    model.to(config.DEVICE)

    all_preds, all_labels = [], []
    examples = []

    with torch.no_grad():
        for i, (points, labels) in enumerate(test_loader):
            points, labels = points.to(config.DEVICE), labels.to(config.DEVICE)
            outputs = model(points)
            preds = outputs.argmax(dim=2).cpu().numpy()
            all_preds.append(preds.flatten())
            all_labels.append(labels.cpu().numpy().flatten())

            if len(examples) < 3:
                examples.append((points[0].cpu().numpy(), labels[0].cpu().numpy(), preds[0]))

    preds = np.concatenate(all_preds)
    labels = np.concatenate(all_labels)

    oa, mIoU, ious, cm = calculate_metrics(preds, labels, config.NUM_CLASSES)
    print(f"Test OA: {oa:.4f}, mIoU: {mIoU:.4f}")
    print("Class IoUs:")
    for i, name in enumerate(config.CLASS_NAMES):
        print(f"{name:12}: {ious[i]:.4f}")

    for i, (xyzrgb, gt, pred) in enumerate(examples):
        gt_colors = get_color_palette(gt)
        pred_colors = get_color_palette(pred)
        pcd_gt = o3d.geometry.PointCloud()
        pcd_pred = o3d.geometry.PointCloud()
        pcd_gt.points = o3d.utility.Vector3dVector(xyzrgb[:, :3])
        pcd_pred.points = o3d.utility.Vector3dVector(xyzrgb[:, :3])
        pcd_gt.colors = o3d.utility.Vector3dVector(gt_colors)
        pcd_pred.colors = o3d.utility.Vector3dVector(pred_colors)
        o3d.io.write_point_cloud(f"viz_gt_{i}.ply", pcd_gt)
        o3d.io.write_point_cloud(f"viz_pred_{i}.ply", pcd_pred)
        # o3d.visualization.draw_geometries([pcd_gt])
        # o3d.visualization.draw_geometries([pcd_pred])

def get_color_palette(labels, seed=42):
    np.random.seed(seed)
    palette = np.random.rand(13, 3)
    palette[0] = [0.8, 0.8, 0.8]  # ceiling — light gray
    palette[1] = [0.2, 0.8, 0.2]  # floor — green
    palette[2] = [0.8, 0.2, 0.2]  # wall — red
    return palette[labels]

if __name__ == "__main__":
    evaluate()