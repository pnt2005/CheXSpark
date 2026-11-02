"""Nạp Stage-2 DenseNet121 (multi-label 14 bệnh) từ checkpoint .pth."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models

from dl.preprocess import DISEASES


def build_model(n_classes: int = len(DISEASES)) -> nn.Module:
    """DenseNet121 với classifier thay bằng Linear(1024, n_classes) — khớp lúc train."""
    model = models.densenet121(weights=None)  # trọng số lấy từ checkpoint
    model.classifier = nn.Linear(model.classifier.in_features, n_classes)
    return model


def load_stage2_model(ckpt: str | Path, device: str = "cpu") -> nn.Module:
    """Load state_dict Stage-2, trả model ở chế độ eval."""
    model = build_model()
    state = torch.load(ckpt, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model.to(device)
