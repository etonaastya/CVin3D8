import numpy as np
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


points = np.random.uniform(0, 100, size=(100000, 3))
np.savetxt('point_cloud.xyz', points)

points = np.loadtxt('point_cloud.xyz')
print(f"Загружено {points.shape[0]} точек.")

def random_subsampling(points, n_samples):
    indices = np.random.choice(points.shape[0], n_samples, replace=False)
    return points[indices]

def voxel_grid_subsampling(points, voxel_size):
    coords = (points / voxel_size).astype(int)
    _, unique_indices = np.unique(coords, axis=0, return_index=True)
    return points[unique_indices]

def farthest_point_sampling(points, n_samples):
    N, D = points.shape
    if n_samples >= N:
        return points

    selected = np.zeros(n_samples, dtype=int)
    distances = np.full(N, np.inf)
    
    first_idx = np.random.randint(N)
    selected[0] = first_idx
    distances = np.linalg.norm(points - points[first_idx], axis=1)

    for i in range(1, n_samples):
        farthest_idx = np.argmax(distances)
        selected[i] = farthest_idx
        
        new_distances = np.linalg.norm(points - points[farthest_idx], axis=1)
        distances = np.minimum(distances, new_distances)

    return points[selected]

n_target = 10000
voxel_size = 1.0  # 1 м × 1 м × 1 м

t0 = time.time()
subsampled_random = random_subsampling(points, n_target)
t1 = time.time()
time_random = t1 - t0

t0 = time.time()
subsampled_voxel = voxel_grid_subsampling(points, voxel_size)
t2 = time.time()
time_voxel = t2 - t0

t0 = time.time()
subsampled_fps = farthest_point_sampling(points, n_target)
t3 = time.time()
time_fps = t3 - t0

print(f"Random subsampling: {len(subsampled_random)} точек, время: {time_random:.4f} с")
print(f"Voxel grid subsampling: {len(subsampled_voxel)} точек, время: {time_voxel:.4f} с")


np.savetxt('./lab1/task1b/subsampled_random.xyz', subsampled_random)
np.savetxt('./lab1/task1b/subsampled_voxel.xyz', subsampled_voxel)


def plot_point_cloud(ax, pts, title, color='b', alpha=0.6):
    if pts.shape[0] > 5000:
        idx = np.random.choice(pts.shape[0], 5000, replace=False)
        pts = pts[idx]
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=1, c=color, alpha=alpha)
    ax.set_title(title)
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    ax.set_box_aspect([1,1,1])

fig = plt.figure(figsize=(15, 5))

ax1 = fig.add_subplot(131, projection='3d')
plot_point_cloud(ax1, subsampled_random, 'Random (10k)', 'red')

ax2 = fig.add_subplot(132, projection='3d')
plot_point_cloud(ax2, subsampled_voxel, f'Voxel (size={voxel_size})\n({len(subsampled_voxel)} pts)', 'green')

plt.tight_layout()
plt.savefig('subsampling_comparison.png', dpi=150)
plt.show()
