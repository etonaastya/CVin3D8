import os
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import h5py

CLASS_NAMES = [
    'ceiling', 'floor', 'wall', 'beam', 'column',
    'window', 'door', 'table', 'chair', 'sofa',
    'bookcase', 'board', 'clutter', 'stairs'
]

def process_room(room_path: str) -> np.ndarray:
    try:
        data = np.genfromtxt(room_path, delimiter=' ', dtype=np.float32)
        if data.ndim == 1:
            data = data.reshape(1, -1)
    except Exception as e:
        raise ValueError(f"Ошибка загрузки {room_path}: {e}")

    xyz = data[:, :3]         # X, Y, Z
    rgb = data[:, 3:6]        # R, G, B
    labels = data[:, 6].astype(np.int64)  # label

    rgb = rgb / 255.0
    rgb = np.clip(rgb, 0.0, 1.0)

    xyz -= np.mean(xyz, axis=0)  
    scale = np.max(np.abs(xyz))
    if scale > 0:
        xyz /= scale

    processed = np.hstack([xyz, rgb, labels.reshape(-1, 1)])

    return processed


def process_area(area_dir: str) -> np.ndarray:
    all_points = []
    
    for room_name in os.listdir(area_dir):
        room_path = os.path.join(area_dir, room_name)
        if not os.path.isdir(room_path):
            continue

        ann_path = os.path.join(room_path, 'Annotations')
        if os.path.isdir(ann_path):
            obj_files = [f for f in os.listdir(ann_path) if f.endswith('.txt')]
            room_data = []
            for obj_file in obj_files:
                obj_full = os.path.join(ann_path, obj_file)
                obj_name = os.path.splitext(obj_file)[0].split('_')[0]  # например, "chair"
                try:
                    label_id = CLASS_NAMES.index(obj_name)
                except ValueError:
                    print(f" Неизвестный класс '{obj_name}' в {obj_full} — пропускаем")
                    continue
                obj_pts = np.genfromtxt(obj_full, delimiter=' ', dtype=np.float32)
                if obj_pts.ndim == 1:
                    obj_pts = obj_pts.reshape(1, -1)
                if obj_pts.shape[1] == 6:
                    obj_pts = np.hstack([obj_pts, np.full((obj_pts.shape[0], 1), label_id)])
                elif obj_pts.shape[1] == 7:
                    obj_pts[:, 6] = label_id
                else:
                    raise ValueError(f"Неверное число столбцов в {obj_full}: {obj_pts.shape[1]}")
                room_data.append(obj_pts)
            if room_data:
                room_points = np.vstack(room_data)
                room_processed = process_room_from_array(room_points)
                all_points.append(room_processed)
        else:
            room_txts = [f for f in os.listdir(room_path) if f.startswith('room') and f.endswith('.txt')]
            for txt in room_txts:
                full_txt = os.path.join(room_path, txt)
                try:
                    processed = process_room(full_txt)
                    all_points.append(processed)
                except Exception as e:
                    print(f"❌ Ошибка обработки {full_txt}: {e}")

    if not all_points:
        raise ValueError(f"Нет данных для обработки в {area_dir}")

    return np.vstack(all_points)


def process_room_from_array(data: np.ndarray) -> np.ndarray:
    xyz = data[:, :3]
    rgb = data[:, 3:6]
    labels = data[:, 6].astype(np.int64)

    # RGB → [0,1]
    rgb = np.clip(rgb / 255.0, 0.0, 1.0)

    # Нормализация XYZ
    xyz -= np.mean(xyz, axis=0)
    scale = np.max(np.abs(xyz))
    if scale > 0:
        xyz /= scale

    return np.hstack([xyz, rgb, labels.reshape(-1, 1)])


def visualize_label_distribution(labels: np.ndarray, save_path: str = None):

    counts = Counter(labels)
    classes = sorted(counts.keys())
    names = [CLASS_NAMES[i] if i < len(CLASS_NAMES) else f'unknown_{i}' for i in classes]
    values = [counts[i] for i in classes]

    plt.figure(figsize=(10, 6))
    bars = plt.bar(names, values, color='skyblue')
    plt.title('Распределение меток по классам (S3DIS)')
    plt.xlabel('Класс')
    plt.ylabel('Количество точек')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    for bar, val in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(values)*0.01,
                 f'{val:,}', ha='center', va='bottom', fontsize=9)

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f" Гистограмма сохранена: {save_path}")
    else:
        plt.show()


def main():
    S3DIS_ROOT = "C:/Users/anast/ml/CVin3D/lab4/task2/Stanford3dDataset"  
    OUTPUT_DIR = "./lab4/task2/output"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    area_paths = [
        os.path.join(S3DIS_ROOT, f"Area_{i}") for i in range(1, 7)
    ]

    all_data = []
    for area_p in area_paths:
        if os.path.isdir(area_p):
            print(f"Обрабатываю {os.path.basename(area_p)}...")
            area_data = process_area(area_p)
            all_data.append(area_data)
            print(f"  → {area_data.shape[0]:,} точек")


    dataset = np.vstack(all_data)
    print(f"\n Всего точек: {dataset.shape[0]:,}")
    print(f"Форма итогового массива: {dataset.shape} (столбцы: X, Y, Z, R, G, B, label)")

    print("\nПервые 5 строк:")
    print(dataset[:5])

    npy_path = os.path.join(OUTPUT_DIR, "s3dis_dataset.npy")
    np.save(npy_path, dataset)
    print(f"\n .npy сохранён: {npy_path}")

    txt_path = os.path.join(OUTPUT_DIR, "s3dis_dataset.txt")
    np.savetxt(txt_path, dataset, fmt='%.6f', delimiter=' ', header='X Y Z R G B label')
    print(f" .txt сохранён: {txt_path}")

    h5_path = os.path.join(OUTPUT_DIR, "s3dis_dataset.h5")
    with h5py.File(h5_path, 'w') as f:
        f.create_dataset('data', data=dataset, compression='gzip')
        f.attrs['columns'] = ['X', 'Y', 'Z', 'R', 'G', 'B', 'label']
        f.attrs['class_names'] = CLASS_NAMES
    print(f" .h5 сохранён: {h5_path}")

    labels = dataset[:, -1].astype(int)
    hist_path = os.path.join(OUTPUT_DIR, "label_distribution.png")
    visualize_label_distribution(labels, save_path=hist_path)

if __name__ == "__main__":
    main()