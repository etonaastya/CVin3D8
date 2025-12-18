import torch
import torch.nn as nn
import torch.nn.functional as F

def square_distance(src, dst):
    # src: (B, N, 3), dst: (B, M, 3)
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.transpose(2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist

def index_points(points, idx):
    # points: (B, N, C), idx: (B, npoint) or (B, npoint, K)
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points

def farthest_point_sample(xyz, npoint):
    # xyz: (B, N, 3)
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids

def query_ball_point(radius, nsample, xyz, new_xyz):
    # xyz: (B, N, 3), new_xyz: (B, npoint, 3)
    device = xyz.device
    B, N, C = xyz.shape
    _, npoint, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, npoint, 1])
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, npoint, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx

class PointNetSetAbstraction(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all=False):
        super().__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel
        self.group_all = group_all

    def forward(self, xyz, points):
        # xyz: (B, N, 3), points: (B, N, C), C >= 3
        if self.group_all:
            new_xyz = xyz.mean(dim=1, keepdim=True)  # (B, 1, 3)
            grouped_points = points.unsqueeze(1)      # (B, 1, N, C)
        else:
            fps_idx = farthest_point_sample(xyz, self.npoint)
            new_xyz = index_points(xyz, fps_idx)
            idx = query_ball_point(self.radius, self.nsample, xyz, new_xyz)
            grouped_points = index_points(points, idx)  # (B, npoint, nsample, C)
            grouped_xyz = index_points(xyz, idx)        # (B, npoint, nsample, 3)
            grouped_xyz -= new_xyz.unsqueeze(2)
            grouped_points = torch.cat([grouped_xyz, grouped_points], dim=-1)
        grouped_points = grouped_points.permute(0, 3, 2, 1)  # (B, C+3, nsample, npoint)
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            grouped_points = F.relu(bn(conv(grouped_points)))
        new_points = torch.max(grouped_points, 2)[0].transpose(1, 2)  # (B, npoint, out_channel)
        return new_xyz, new_points

class PointNetFeaturePropagation(nn.Module):
    def __init__(self, in_channel, mlp):
        super().__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel

    def forward(self, xyz1, xyz2, points1, points2):
        # xyz1: (B, N1, 3), xyz2: (B, N2, 3)
        # points1: (B, N1, C1), points2: (B, N2, C2)
        if xyz2 is None:
            # global feature
            net = torch.max(points1, dim=1, keepdim=True)[0].repeat(1, points1.shape[1], 1)
            return torch.cat([points1, net], dim=-1)
        dist, idx = self.three_nn(xyz1, xyz2)
        dist_recip = 1.0 / (dist + 1e-8)
        norm = torch.sum(dist_recip, dim=2, keepdim=True)
        weight = dist_recip / norm
        interpolated_points = self.three_interpolate(points2, idx, weight)
        if points1 is not None:
            new_points = torch.cat([points1, interpolated_points], dim=2)
        else:
            new_points = interpolated_points
        new_points = new_points.permute(0, 2, 1)
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))
        return new_points.permute(0, 2, 1)

    def three_nn(self, xyz1, xyz2):
        # xyz1: (B, N1, 3), xyz2: (B, N2, 3)
        dist = square_distance(xyz1, xyz2)
        _, idx = torch.topk(dist, 3, dim=2, largest=False, sorted=False)
        dist2 = dist.gather(2, idx)
        return dist2, idx

    def three_interpolate(self, features, idx, weight):
        # features: (B, N2, C), idx: (B, N1, 3), weight: (B, N1, 3)
        B, N2, C = features.shape
        _, N1, _ = idx.shape
        device = features.device
        idx = idx.view(B, N1 * 3)
        weight = weight.view(B, N1 * 3, 1).repeat(1, 1, C)
        features = features.contiguous().view(B * N2, C)
        idx = idx.view(B, N1 * 3, 1).expand(B, N1 * 3, C).contiguous().view(B * N1 * 3, C)
        interpolated = torch.gather(features, 0, idx)
        interpolated = interpolated.view(B, N1, 3, C)
        interpolated = torch.sum(interpolated * weight.view(B, N1, 3, C), dim=2)
        return interpolated

class PointNet2Seg(nn.Module):
    def __init__(self, num_classes=13, in_dim=6):
        super().__init__()
        self.sa1 = PointNetSetAbstraction(
            npoint=1024, radius=0.2, nsample=32,
            in_channel=in_dim, mlp=[64, 64, 128]
        )
        self.sa2 = PointNetSetAbstraction(
            npoint=256, radius=0.4, nsample=32,
            in_channel=128 + 3, mlp=[128, 128, 256]
        )
        self.sa3 = PointNetSetAbstraction(
            npoint=64, radius=0.8, nsample=32,
            in_channel=256 + 3, mlp=[256, 256, 512]
        )
        self.sa4 = PointNetSetAbstraction(
            npoint=None, radius=None, nsample=None,
            in_channel=512 + 3, mlp=[512, 512, 1024], group_all=True
        )

        self.fp4 = PointNetFeaturePropagation(in_channel=1024 + 512, mlp=[512, 512])
        self.fp3 = PointNetFeaturePropagation(in_channel=512 + 256, mlp=[256, 256])
        self.fp2 = PointNetFeaturePropagation(in_channel=256 + 128, mlp=[256, 128])
        self.fp1 = PointNetFeaturePropagation(in_channel=128 + in_dim, mlp=[128, 128, 128])

        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(128, num_classes, 1)

    def forward(self, xyz):
        # xyz: (B, N, 6) — [x,y,z,r,g,b]
        B, N, C = xyz.shape
        l0_xyz = xyz[:, :, :3]
        l0_points = xyz

        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)

        l3_points = self.fp4(l3_xyz, l4_xyz, l3_points, l4_points)
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(l0_xyz, l1_xyz, l0_points, l1_points)

        x = l0_points.permute(0, 2, 1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.drop1(x)
        x = self.conv2(x)
        x = x.permute(0, 2, 1)
        return x  # (B, N, num_classes)