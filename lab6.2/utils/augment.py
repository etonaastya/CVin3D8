import numpy as np
import torch

def random_dropout(points, keep_ratio=0.95):
    N = points.shape[0]
    keep = np.random.choice(N, int(N * keep_ratio), replace=False)
    return points[keep]

def random_jitter(points, sigma=0.01, clip=0.05):
    jitter = np.clip(sigma * np.random.randn(*points.shape), -clip, clip)
    points[:, :3] += jitter[:, :3]  # только xyz
    return points

def random_rotate_z(points):
    angle = np.random.uniform(0, 2 * np.pi)
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    R = np.array([[cos_a, -sin_a, 0],
                  [sin_a,  cos_a, 0],
                  [0,      0,     1]], dtype=np.float32)
    points[:, :3] = points[:, :3] @ R.T
    return points

def augment_batch(points_batch, config):
    # points_batch: (B, N, C)
    B, N, C = points_batch.shape
    augmented = []
    for i in range(B):
        p = points_batch[i].copy()
        if config.ROTATE_Z:
            p = random_rotate_z(p)
        if config.JITTER:
            p = random_jitter(p, config.JITTER_SIGMA, config.JITTER_CLIP) 
        if config.DROPOUT_POINTS < 1.0:
            p = random_dropout(p, config.DROPOUT_POINTS)
            if len(p) < N:
                idx = np.random.choice(len(p), N - len(p), replace=True)
                p = np.vstack([p, p[idx]])
            elif len(p) > N:
                p = p[:N]
        augmented.append(p)
    return np.stack(augmented, axis=0)