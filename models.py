"""Sequence classifiers used by SeqSense. Inputs are [batch, time, channels]."""
from __future__ import annotations

import torch
from torch import nn


class CNN1D(nn.Module):
    def __init__(self, input_channels: int = 9, num_classes: int = 6):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(input_channels, 64, kernel_size=7, padding=3), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=5, padding=2), nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=5, padding=2), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Conv1d(128, 128, kernel_size=3, padding=1), nn.BatchNorm1d(128), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(128, 128, kernel_size=3, padding=1), nn.BatchNorm1d(128), nn.ReLU(),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(128, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x.transpose(1, 2))
        return self.classifier(self.pool(x).squeeze(-1))


class RecurrentClassifier(nn.Module):
    def __init__(self, kind: str, input_channels: int = 9, hidden_size: int = 64, num_classes: int = 6):
        super().__init__()
        key = kind.lower()
        constructors = {"rnn": nn.RNN, "lstm": nn.LSTM, "gru": nn.GRU}
        if key not in constructors:
            raise ValueError(f"Unsupported recurrent model: {kind}")
        self.kind = key
        recurrent_args = dict(input_size=input_channels, hidden_size=hidden_size,
                              num_layers=2, dropout=0.3, batch_first=True)
        if key == "rnn":
            self.recurrent = nn.RNN(**recurrent_args, nonlinearity="tanh")
        else:
            self.recurrent = constructors[key](**recurrent_args)
        self.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(hidden_size, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, _ = self.recurrent(x)
        return self.classifier(output[:, -1, :])


class CNNLSTM(nn.Module):
    def __init__(self, input_channels: int = 9, num_classes: int = 6):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(input_channels, 64, kernel_size=5, padding=2), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=5, padding=2), nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2),
        )
        self.lstm = nn.LSTM(input_size=64, hidden_size=64, batch_first=True)
        self.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(64, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x.transpose(1, 2)).transpose(1, 2)
        output, _ = self.lstm(x)
        return self.classifier(output[:, -1, :])


MODEL_NAMES = ("cnn", "rnn", "lstm", "gru", "cnn_lstm")


def build_model(name: str) -> nn.Module:
    key = name.lower()
    if key == "cnn":
        return CNN1D()
    if key in {"rnn", "lstm", "gru"}:
        return RecurrentClassifier(key)
    if key == "cnn_lstm":
        return CNNLSTM()
    raise ValueError(f"Unknown model {name!r}; choose from {', '.join(MODEL_NAMES)}")
