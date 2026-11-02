"""Grad-CAM (Gradient-weighted Class Activation Mapping) cho Stage-3 Explainability.

Sinh bản đồ nhiệt từ feature map tích chập cuối của DenseNet121 để khoanh vùng
tổn thương nghi ngờ cho từng bệnh — bác sĩ đối chiếu thay vì chỉ nhận con số.

Công thức (Selvaraju et al., 2017):
    α_k = (1/Z) Σ_ij ∂y^c / ∂A^k_ij        (global average pooling của gradient)
    L^c = ReLU( Σ_k α_k · A^k )            (tổ hợp có trọng số + ReLU)

Lớp đích: `model.features.denseblock4` (output 1024×7×7). Không hook
`model.features` vì DenseNet gọi `F.relu(features, inplace=True)` ngay sau đó —
in-place trên output của hook sẽ làm autograd báo lỗi
("Output 0 of BackwardHookFunction is a view and is being modified inplace").
`denseblock4` cũng cho feature map 1024 kênh cuối cùng nên tương đương về mặt CAM.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


class GradCAM:
    """Grad-CAM cho model multi-label: mỗi class = 1 logit độc lập (không softmax)."""

    def __init__(self, model, target_layer=None):
        self.model = model
        self.target = (
            target_layer if target_layer is not None else model.features.denseblock4
        )
        self._acts: torch.Tensor | None = None
        self._grads: torch.Tensor | None = None
        self._handles = [
            self.target.register_forward_hook(self._save_acts),
            self.target.register_full_backward_hook(self._save_grads),
        ]

    def _save_acts(self, module, inputs, output):
        self._acts = output.detach()

    def _save_grads(self, module, grad_input, grad_output):
        self._grads = grad_output[0].detach()

    def remove(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles = []

    def generate(self, x: torch.Tensor, class_idx: int) -> np.ndarray:
        """Trả CAM chuẩn hóa [0,1] kích thước feature map (vd 7×7)."""
        self.model.zero_grad(set_to_none=True)
        logits = self.model(x)
        logits[0, class_idx].backward()

        weights = self._grads.mean(dim=(2, 3), keepdim=True)  # α_k
        cam = F.relu((weights * self._acts).sum(dim=1))        # (1, h, w)
        cam = cam.squeeze(0)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        return cam.cpu().numpy()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.remove()


def upsample_cam(cam: np.ndarray, size: int) -> np.ndarray:
    """Bilinear upsample CAM về kích thước ảnh gốc."""
    t = torch.from_numpy(cam).unsqueeze(0).unsqueeze(0)
    t = F.interpolate(t, size=(size, size), mode="bilinear", align_corners=False)
    return t.squeeze().numpy()


def jet_rgb(v: np.ndarray) -> np.ndarray:
    """Colormap 'jet' thuần numpy (tránh phụ thuộc matplotlib).

    v ∈ [0,1] → RGB uint8. Xấp xỉ chuẩn: R=1.5-|4v-3|, G=1.5-|4v-2|, B=1.5-|4v-1|.
    """
    r = np.clip(1.5 - np.abs(4.0 * v - 3.0), 0, 1)
    g = np.clip(1.5 - np.abs(4.0 * v - 2.0), 0, 1)
    b = np.clip(1.5 - np.abs(4.0 * v - 1.0), 0, 1)
    return (np.stack([r, g, b], axis=-1) * 255).astype(np.uint8)


def overlay_heatmap(gray: np.ndarray, cam: np.ndarray, mask: np.ndarray | None = None,
                    alpha: float = 0.45) -> np.ndarray:
    """Phủ heatmap lên ảnh xám.

    Nếu có `mask` (vùng phổi) thì chỉ giữ CAM trong phổi — tránh model "nhìn" nền
    và tạo heatmap ngoài vùng giải phẫu.
    """
    cam = upsample_cam(cam, gray.shape[0])
    if mask is not None:
        cam = cam * mask
    heat = jet_rgb(cam).astype(np.float32) / 255.0
    base = np.stack([gray] * 3, axis=-1)
    out = (1 - alpha) * base + alpha * heat
    return (np.clip(out, 0, 1) * 255).astype(np.uint8)


def mask_visual(gray: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Ảnh xám + viền/ánh xanh vùng phổi để đối chiếu."""
    base = np.stack([gray] * 3, axis=-1)
    tint = np.zeros_like(base)
    tint[..., 1] = 0.55 * mask
    tint[..., 2] = 0.35 * mask
    return (np.clip(base + tint * 0.6, 0, 1) * 255).astype(np.uint8)


def to_pil(arr: np.ndarray) -> Image.Image:
    return Image.fromarray(arr.astype(np.uint8))
