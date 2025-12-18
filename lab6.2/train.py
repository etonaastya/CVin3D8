import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
import numpy as np
from config import Config
from dataset import get_dataloaders
from model import PointNet2Seg
from utils.metrics import calculate_metrics

def train():
    config = Config()
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)

    train_loader, val_loader, _ = get_dataloaders(config)
    model = PointNet2Seg(num_classes=config.NUM_CLASSES, in_dim=6).to(config.DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LR)
    scheduler = StepLR(optimizer, step_size=config.LR_STEP, gamma=config.LR_GAMMA)

    train_losses, val_losses = [], []
    best_val_loss = float('inf')

    for epoch in range(config.EPOCHS):
        model.train()
        total_loss = 0.0
        for points, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.EPOCHS}"):
            points, labels = points.to(config.DEVICE), labels.to(config.DEVICE)
            optimizer.zero_grad()
            outputs = model(points)  # (B, N, C)
            loss = criterion(outputs.permute(0, 2, 1), labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        model.eval()
        val_loss = 0.0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for points, labels in val_loader:
                points, labels = points.to(config.DEVICE), labels.to(config.DEVICE)
                outputs = model(points)
                loss = criterion(outputs.permute(0, 2, 1), labels)
                val_loss += loss.item()
                preds = outputs.argmax(dim=2).cpu().numpy().flatten()
                all_preds.append(preds)
                all_labels.append(labels.cpu().numpy().flatten())
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        oa, mIoU, _, _ = calculate_metrics(
            np.concatenate(all_preds), np.concatenate(all_labels), config.NUM_CLASSES
        )

        print(f"Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, OA: {oa:.3f}, mIoU: {mIoU:.3f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(config.CHECKPOINT_DIR, "best_model.pth"))

        scheduler.step()

    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(config.LOG_DIR, "loss_curve.png"))
    plt.close()

if __name__ == "__main__":
    train()