"""Reference models. Replace with your exact architectures so results stay comparable with the paper.

Each model takes a (batch, n_features) float tensor and treats the feature vector as a
1-channel sequence, which matches the "70-timestep" input described in Appendix A.1.
"""
from __future__ import annotations

import torch
from torch import nn


class CNN1D(nn.Module):
    """norm='bn' (BatchNorm, default) or 'gn' (GroupNorm with 8 groups, no running statistics)."""

    def __init__(self, n_features: int, n_classes: int, width: int = 32, norm: str = "bn"):
        super().__init__()
        N = (lambda c: nn.BatchNorm1d(c)) if norm == "bn" else (lambda c: nn.GroupNorm(8, c))
        self.net = nn.Sequential(
            nn.Conv1d(1, width, 3, padding=1), N(width), nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(width, 2 * width, 3, padding=1), N(2 * width), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1), nn.Flatten(),
            nn.Linear(2 * width, 64), nn.ReLU(), nn.Linear(64, n_classes))

    def forward(self, x):
        return self.net(x.unsqueeze(1))


class MLP(nn.Module):
    """Plain multilayer perceptron for tabular flow features (no normalisation layers)."""

    def __init__(self, n_features: int, n_classes: int, hidden=(128, 64)):
        super().__init__()
        layers, d = [], n_features
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU()]
            d = h
        self.net = nn.Sequential(*layers, nn.Linear(d, n_classes))

    def forward(self, x):
        return self.net(x)


def split_head(model: nn.Module):
    """(feature extractor, linear classifier) for CNN1D and MLP, used by CCVR."""
    body = nn.Sequential(*list(model.net.children())[:-1])
    if isinstance(model, CNN1D):
        return (lambda x: body(x.unsqueeze(1))), model.net[-1]
    return body, model.net[-1]


class RecurrentNet(nn.Module):
    """LSTM / BiLSTM, optionally with a CNN front end (CNN-LSTM, CNN-BiLSTM)."""

    def __init__(self, n_features: int, n_classes: int, hidden: int = 64,
                 bidirectional: bool = False, cnn_front: bool = False):
        super().__init__()
        self.cnn_front = cnn_front
        in_dim = 1
        if cnn_front:
            self.front = nn.Sequential(nn.Conv1d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool1d(2))
            in_dim = 32
        self.rnn = nn.LSTM(in_dim, hidden, batch_first=True, bidirectional=bidirectional)
        self.head = nn.Linear(hidden * (2 if bidirectional else 1), n_classes)

    def forward(self, x):
        x = x.unsqueeze(1)                       # (B, 1, F)
        if self.cnn_front:
            x = self.front(x)                    # (B, 32, F/2)
        x = x.transpose(1, 2)                    # (B, T, C)
        out, _ = self.rnn(x)
        return self.head(out[:, -1])


def build_model(name: str, n_features: int, n_classes: int) -> nn.Module:
    name = name.lower()
    if name == "cnn":
        return CNN1D(n_features, n_classes)
    if name == "cnn_gn":
        return CNN1D(n_features, n_classes, norm="gn")
    if name == "mlp":
        return MLP(n_features, n_classes)
    if name == "lstm":
        return RecurrentNet(n_features, n_classes)
    if name == "bilstm":
        return RecurrentNet(n_features, n_classes, bidirectional=True)
    if name == "cnn-lstm":
        return RecurrentNet(n_features, n_classes, cnn_front=True)
    if name == "cnn-bilstm":
        return RecurrentNet(n_features, n_classes, bidirectional=True, cnn_front=True)
    raise ValueError(f"unknown model {name}")


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
