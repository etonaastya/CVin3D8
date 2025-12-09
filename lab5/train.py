import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
from tqdm import tqdm
import os

from data import ModelNetDataset
from model import PointNet

BATCH_SIZE = 32
EPOCHS = 80
LR = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_PATH = "pointnet_model.pth"

def train():
    train_ds = ModelNetDataset('train')
    test_ds = ModelNetDataset('test')
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    num_classes = len(np.unique(train_ds.labels))

    model = PointNet(num_classes=num_classes).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    train_losses, train_accs = [], []
    test_losses, test_accs = [], []

    print(f"Обучение PointNet на {num_classes} классах (ModelNet10). Устройство: {DEVICE}")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss, train_correct, train_total = 0, 0, 0
        for points, labels in tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}", leave=False):
            points, labels = points.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            logits, t_input, t_feat = model(points)
            loss = criterion(logits, labels)
            if t_input is not None:
                loss += 0.001 * torch.mean(torch.norm(torch.bmm(t_input, t_input.transpose(1, 2)) - torch.eye(3).unsqueeze(0).to(DEVICE), dim=(1, 2)))
            if t_feat is not None:
                loss += 0.001 * torch.mean(torch.norm(torch.bmm(t_feat, t_feat.transpose(1, 2)) - torch.eye(64).unsqueeze(0).to(DEVICE), dim=(1, 2)))
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            pred = logits.argmax(dim=1)
            train_correct += (pred == labels).sum().item()
            train_total += labels.size(0)

        train_loss /= len(train_loader)
        train_acc = train_correct / train_total
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        model.eval()
        test_loss, test_correct, test_total = 0, 0, 0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for points, labels in test_loader:
                points, labels = points.to(DEVICE), labels.to(DEVICE)
                logits, _, _ = model(points)
                loss = criterion(logits, labels)
                test_loss += loss.item()
                pred = logits.argmax(dim=1)
                test_correct += (pred == labels).sum().item()
                test_total += labels.size(0)
                all_preds.extend(pred.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        test_loss /= len(test_loader)
        test_acc = test_correct / test_total
        test_losses.append(test_loss)
        test_accs.append(test_acc)

        scheduler.step()

        print(f"Epoch {epoch:02d} | Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
              f"Val Loss: {test_loss:.4f}, Acc: {test_acc:.4f}")

        if epoch == 1 or test_acc > max(test_accs[:-1]):
            torch.save(model.state_dict(), SAVE_PATH)
            print(f" Saved best model (acc={test_acc:.4f})")

    print("\n Обучение завершено!")
    print(f"Лучшая точность на тесте: {max(test_accs):.4f}")

    with h5py.File("../modelnet10_1024.h5", 'r') as f:
        classes = [cls.decode('utf-8') for cls in f['classes'][:]]

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig("confusion_matrix.png")
    print(" Confusion matrix saved as 'confusion_matrix.png'")

    # Графики
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Train Loss")
    plt.plot(test_losses, label="Val Loss")
    plt.xlabel("Epoch"), plt.ylabel("Loss"), plt.legend()
    plt.title("Loss Curves")

    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label="Train Acc")
    plt.plot(test_accs, label="Val Acc")
    plt.xlabel("Epoch"), plt.ylabel("Accuracy"), plt.legend()
    plt.title("Accuracy Curves")
    plt.tight_layout()
    plt.savefig("training_curves.png")
    print("Training curves saved as 'training_curves.png'")

    model.load_state_dict(torch.load(SAVE_PATH))
    model.eval()
    with torch.no_grad():
        for i in range(5):
            points, label = test_ds[i]
            points_t = torch.tensor(points).unsqueeze(0).to(DEVICE)
            logits, _, _ = model(points_t)
            pred = logits.argmax().item()

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            color = [0, 1, 0] if pred == label else [1, 0, 0]  # green=correct, red=wrong
            pcd.colors = o3d.utility.Vector3dVector([color] * len(points))

            vis = o3d.visualization.Visualizer()
            vis.create_window(window_name=f"True: {classes[label]} | Pred: {classes[pred]}", width=600, height=400)
            vis.add_geometry(pcd)
            vis.run()
            vis.capture_screen_image(f"pred_{i}.png")
            vis.destroy_window()

if __name__ == "__main__":
    train()