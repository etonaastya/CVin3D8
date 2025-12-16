import numpy as np
import matplotlib.pyplot as plt
import h5py
import os  

def load_and_process_semantic3d(txt_path: str, label_path: str) -> np.ndarray:
    print("Загрузка данных из .txt...")
    features = np.loadtxt(txt_path)

    X, Y, Z, intensity, R, G, B = features[:, :7].T

    print("Загрузка меток из .labels...")
    labels = np.loadtxt(label_path, dtype=np.int32)

    if len(labels) != len(features):
        raise ValueError("Количество точек в .txt и .labels не совпадает!")

    # Нормализация
    R, G, B = R / 255.0, G / 255.0, B / 255.0
    coords = np.stack([X, Y, Z], axis=1)
    coords_norm = (coords - coords.mean(axis=0)) / (coords.std(axis=0) + 1e-8)
    X_norm, Y_norm, Z_norm = coords_norm.T
    intensity_norm = (intensity - intensity.min()) / (intensity.max() - intensity.min() + 1e-8)

    return np.column_stack([X_norm, Y_norm, Z_norm, R, G, B, intensity_norm, labels])


def plot_label_distribution(labels: np.ndarray, title="Распределение меток классов"):
    unique, counts = np.unique(labels, return_counts=True)
    plt.figure(figsize=(10, 6))
    plt.bar(unique, counts, color='skyblue', edgecolor='black')
    plt.xlabel("Класс (label)"); plt.ylabel("Количество"); plt.title(title)
    plt.xticks(unique); plt.grid(axis='y', linestyle='--', alpha=0.7)
    for i, v in enumerate(counts):
        plt.text(unique[i], v + max(counts)*0.01, str(v), ha='center', va='bottom')
    plt.tight_layout()
    plt.savefig("label_distribution.png")
    plt.show()


def save_dataset(dataset: np.ndarray, base_name="semantic3d_dataset"):
    np.save(f"{base_name}.npy", dataset)
    np.savetxt(f"{base_name}_head1000.txt", dataset[:1000],
               fmt='%.6f %.6f %.6f %.4f %.4f %.4f %.4f %d',
               header="X_norm Y_norm Z_norm R G B intensity_norm label", comments='')
    with h5py.File(f"{base_name}.h5", "w") as f:
        f.create_dataset("data", data=dataset, compression="gzip")
    print(f" Сохранено: .npy, _head1000.txt, .h5")


if __name__ == "__main__":
    txt = "./lab4/task1/Semantic3D/bildstein_station1_xyz_intensity_rgb.txt"
    label = "./lab4/task1/Semantic3D/bildstein_station1_xyz_intensity_rgb.labels"

    dataset = load_and_process_semantic3d(txt, label)
    print(f"\n Обработано {dataset.shape[0]} точек. Размер: {dataset.shape}, dtype: {dataset.dtype}")

    save_dataset(dataset)

    print("\n Первые 5 строк:")
    print(dataset[:5])

    plot_label_distribution(dataset[:, -1].astype(int))

