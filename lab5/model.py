import torch
import torch.nn as nn
import torch.nn.functional as F

class TNet(nn.Module):
    def __init__(self, k=3):
        super().__init__()
        self.k = k
        self.mlp1 = nn.Sequential(
            nn.Conv1d(k, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, k * k),
        )
        self.register_buffer('eye', torch.eye(k))

    def forward(self, x):
        # x: (B, k, N)
        B, _, _ = x.shape
        x = self.mlp1(x)        # (B, 1024, N)
        x = F.adaptive_max_pool1d(x, 1).squeeze(-1)  # (B, 1024)
        x = self.fc(x)           # (B, k*k)
        x = x.view(B, self.k, self.k) + self.eye
        return x  # transformation matrix

class PointNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.input_transform = TNet(k=3)
        self.feature_transform = TNet(k=64)

        # shared MLPs
        self.mlp1 = nn.Sequential(
            nn.Conv1d(3, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )
        self.mlp2 = nn.Sequential(
            nn.Conv1d(64, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        # x: (B, N, 3) → transpose to (B, 3, N)
        x = x.transpose(1, 2).contiguous()

        # Input transform
        t_input = self.input_transform(x)  # (B, 3, 3)
        x = torch.bmm(t_input, x)          # (B, 3, N)

        # First MLP
        x = self.mlp1(x)                   # (B, 64, N)

        # Feature transform
        t_feat = self.feature_transform(x) # (B, 64, 64)
        x = torch.bmm(t_feat, x)           # (B, 64, N)

        # Second MLP
        x = self.mlp2(x)                   # (B, 1024, N)

        # Max pool
        x = F.adaptive_max_pool1d(x, 1).squeeze(-1)  # (B, 1024)

        # Classifier
        logits = self.classifier(x)
        return logits, t_input, t_feat