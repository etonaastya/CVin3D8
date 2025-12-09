import torch
import numpy as np
import matplotlib.pyplot as plt
import h5py  # ← теперь импортирован!
from sklearn.metrics import confusion_matrix
import open3d as o3d
from model import PointNet
from data import ModelNetDataset

print("Загружаем тестовый датасет...")
test_ds = ModelNetDataset('test', cache_path="../modelnet10_1024.h5")
num_classes = len(np.unique(test_ds.labels))

print("Загружаем обученную модель...")
model = PointNet(num_classes=num_classes)
model.load_state_dict(torch.load("../pointnet_model.pth", weights_only=True))
model.eval()

print("Делаем предсказания на тесте...")
all_preds, all_labels = [], []
with torch.no_grad():
    for i in range(len(test_ds)):
        points, label = test_ds[i]
        points_t = torch.tensor(points).unsqueeze(0).float()
        logits, _, _ = model(points_t)
        pred = logits.argmax().item()
        all_preds.append(pred)
        all_labels.append(label)

all_preds = np.array(all_preds)
all_labels = np.array(all_labels)
test_acc = (all_preds == all_labels).mean()
print(f" Точность (проверка): {test_acc:.4f}")

with h5py.File("../modelnet10_1024.h5", 'r') as f:
    classes = [cls.decode('utf-8') for cls in f['classes'][:]]

print(" Строим confusion matrix...")
cm = confusion_matrix(all_labels, all_preds, normalize='true')
plt.figure(figsize=(8, 6))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title(f"Confusion Matrix (Test Acc: {test_acc:.2%})")
plt.colorbar()
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=45, ha='right')
plt.yticks(tick_marks, classes)
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150, bbox_inches='tight')
print(" confusion_matrix.png сохранён")

epochs = list(range(1, 81))
val_acc = [
    0.6476, 0.7059, 0.7148, 0.7533, 0.7687, 0.7698, 0.7621, 0.7885,
    0.8117, 0.8128, 0.7952, 0.7941, 0.7775, 0.8359, 0.8040, 0.8293,
    0.8106, 0.8469, 0.8491, 0.8282, 0.8326, 0.8590, 0.8436, 0.8447,
    0.8502, 0.8667, 0.8634, 0.8700, 0.8590, 0.8414, 0.8436, 0.8425,
    0.8436, 0.8403, 0.8601, 0.8436, 0.8447, 0.8645, 0.8458, 0.8073,
    0.8634, 0.8623, 0.8634, 0.8778, 0.8579, 0.8612, 0.8524, 0.8480,
    0.8546, 0.8469, 0.8623, 0.8634, 0.8546, 0.8645, 0.8502, 0.8601,
    0.8722, 0.8744, 0.8491, 0.8689, 0.8722, 0.8744, 0.8722, 0.8789,
    0.8722, 0.8689, 0.8700, 0.8557, 0.8711, 0.8800, 0.8612, 0.8612,
    0.8634, 0.8789, 0.8689, 0.8634, 0.8667, 0.8579, 0.8557, 0.8579
]

plt.figure(figsize=(6, 4))
plt.plot(epochs, val_acc, label='Validation Accuracy', color='tab:blue', linewidth=2)
plt.axhline(y=0.8811, color='r', linestyle='--', label='Best: 88.11%')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Validation Accuracy over Epochs')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("val_accuracy.png", dpi=150)
print(" val_accuracy.png сохранён")

print("Генерируем 3D-визуализации...")
for i in range(5):
    points, label = test_ds[i]
    points_t = torch.tensor(points).unsqueeze(0).float()
    logits, _, _ = model(points_t)
    pred = logits.argmax().item()
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    color = [0, 1, 0] if pred == label else [1, 0, 0]
    pcd.colors = o3d.utility.Vector3dVector([color] * len(points))
    
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=600, height=400, visible=False)
    vis.add_geometry(pcd)
    vis.poll_events()
    vis.update_renderer()
    vis.capture_screen_image(f"pred_{i}.png", do_render=True)
    vis.destroy_window()

print("\n Готово! Все файлы в папке lab5/")
