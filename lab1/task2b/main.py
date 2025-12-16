import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

np.random.seed(42)  
points = np.random.uniform(0, 100, size=(100000, 3))
np.savetxt("synthetic_cloud.xyz", points)


def show_cloud(points, title="Point Cloud", ax=None, color='b'):
    if ax is None:
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=0.5, alpha=0.6, c=color)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    ax.set_box_aspect([1, 1, 1])  
    return ax


def filter_by_bbox(points, xmin, xmax, ymin, ymax, zmin, zmax):
    mask = (
        (points[:, 0] >= xmin) & (points[:, 0] <= xmax) &
        (points[:, 1] >= ymin) & (points[:, 1] <= ymax) &
        (points[:, 2] >= zmin) & (points[:, 2] <= zmax)
    )
    return points[mask]

bbox_params = (20, 50, 30, 70, 10, 40)
filtered_bbox = filter_by_bbox(points, *bbox_params)
np.savetxt("bbox_filtered.xyz", filtered_bbox)
print(f"BBox-фильтр [x∈{bbox_params[:2]}, y∈{bbox_params[2:4]}, z∈{bbox_params[4:]}]: {len(filtered_bbox)} точек")


high_points = points[points[:, 2] > 80]
np.savetxt("high_points.xyz", high_points)
print(f" Высокие точки (Z > 80): {len(high_points)} точек")


def filter_by_distance(points, center, radius):
    distances = np.linalg.norm(points - center, axis=1)
    return points[distances <= radius]

center = np.array([50, 50, 50])
radius = 20
near_center = filter_by_distance(points, center, radius)
np.savetxt("near_center.xyz", near_center)
print(f" Точки в радиусе {radius} от {center}: {len(near_center)} точек")


print(f"Исходное облако:          {len(points)} точек")
print(f"После BBox-фильтра:       {len(filtered_bbox)} точек ({len(filtered_bbox)/len(points)*100:.2f}%)")
print(f"Высокие точки (Z > 80):   {len(high_points)} точек ({len(high_points)/len(points)*100:.2f}%)")
print(f"В радиусе 20 от центра:   {len(near_center)} точек ({len(near_center)/len(points)*100:.2f}%)")


fig = plt.figure(figsize=(14, 12))

ax1 = fig.add_subplot(2, 2, 1, projection='3d')
show_cloud(points, "Исходное облако (100 000 точек)", ax=ax1, color='lightgray')

ax2 = fig.add_subplot(2, 2, 2, projection='3d')
show_cloud(points[::20], "BBox-фильтр (подмножество)", ax=ax2, color='lightgray')  
show_cloud(filtered_bbox, "", ax=ax2, color='red')
ax2.set_title("BBox-фильтр (красные точки)")

ax3 = fig.add_subplot(2, 2, 3, projection='3d')
show_cloud(points[::20], "Высокие точки (Z > 80)", ax=ax3, color='lightgray')
show_cloud(high_points, "", ax=ax3, color='green')
ax3.set_title("Высокие точки (Z > 80, зелёные)")

ax4 = fig.add_subplot(2, 2, 4, projection='3d')
show_cloud(points[::20], f"Радиус {radius} от центра", ax=ax4, color='lightgray')
show_cloud(near_center, "", ax=ax4, color='blue')
ax4.set_title(f"Радиус {radius} от (50,50,50), синие")

plt.tight_layout()
plt.savefig("segmentation_results.png", dpi=150)
plt.show()
