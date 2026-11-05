"""Stage-1: U-Net phân đoạn trường phổi từ ảnh X-quang ngực thô.

Vai trò trong pipeline 3 giai đoạn:

    ảnh CXR thô ──► [U-Net] ──► mask phổi ──► [DenseNet121] ──► 14 xác suất
                                      └────► [Grad-CAM] ──► heatmap

Lúc huấn luyện, mask "đúng" lấy từ **CheXmask** (HybridGNet sinh sẵn cho toàn bộ
NIH ChestX-ray14). Lúc suy luận, U-Net tự sinh mask — nhờ vậy hệ thống không còn
phụ thuộc CheXmask khi nhận ảnh mới từ bệnh viện.

Quy ước dữ liệu (giống `dl.preprocess` để Stage-2 dùng lại được mask này):
  - ảnh xám float32 [0,1], resize về `IMG_SIZE` (224)
  - mask nhị phân cùng kích thước, 1 = phổi
  - model nhận tensor (N,1,H,W) đã chuẩn hoá `(x-0.5)/0.5`, xuất logits cùng H,W
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from dl.preprocess import IMG_SIZE


# --------------------------------------------------------------------------- #
# Kiến trúc
# --------------------------------------------------------------------------- #
class ConvBlock(nn.Module):
    """2×(conv 3×3 + BN + ReLU) — khối tích chập cơ bản của U-Net."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UNet(nn.Module):
    """U-Net 4 tầng xuống, base=16 → ~1.9M tham số.

    Nhỏ có chủ đích: mục tiêu chỉ là mask phổi (vùng tương phản cao, 1 lớp), không
    cần backbone lớn; 2k ảnh dev + CPU cũng train được. Khi mở rộng lên 112k ảnh
    có thể tăng `base` lên 32/64 mà không đổi interface.
    """

    def __init__(self, in_ch: int = 1, out_ch: int = 1, base: int = 16) -> None:
        super().__init__()
        b = base
        self.enc1 = ConvBlock(in_ch, b)
        self.enc2 = ConvBlock(b, b * 2)
        self.enc3 = ConvBlock(b * 2, b * 4)
        self.enc4 = ConvBlock(b * 4, b * 8)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = ConvBlock(b * 8, b * 16)

        self.up4 = nn.ConvTranspose2d(b * 16, b * 8, 2, stride=2)
        self.dec4 = ConvBlock(b * 16, b * 8)
        self.up3 = nn.ConvTranspose2d(b * 8, b * 4, 2, stride=2)
        self.dec3 = ConvBlock(b * 8, b * 4)
        self.up2 = nn.ConvTranspose2d(b * 4, b * 2, 2, stride=2)
        self.dec2 = ConvBlock(b * 4, b * 2)
        self.up1 = nn.ConvTranspose2d(b * 2, b, 2, stride=2)
        self.dec1 = ConvBlock(b * 2, b)
        self.out = nn.Conv2d(b, out_ch, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))

        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.out(d1)


def build_unet(in_ch: int = 1, out_ch: int = 1, base: int = 16) -> nn.Module:
    return UNet(in_ch=in_ch, out_ch=out_ch, base=base)


def load_stage1_model(ckpt: str | Path, device: str = "cpu") -> nn.Module:
    """Load state_dict U-Net Stage-1 (file `.pth` chỉ chứa state_dict).

    Tự suy ra `base` (số kênh tầng đầu) từ checkpoint — model dev train base=16,
    model full 112k train base=32 → không được hardcode.
    """
    state = torch.load(ckpt, map_location="cpu")
    if isinstance(state, dict) and "model" in state and isinstance(state["model"], dict):
        state = state["model"]   # checkpoint resume (best + optimizer + epoch)
    base = int(state["enc1.net.0.weight"].shape[0])
    model = build_unet(base=base)
    model.load_state_dict(state)
    model.eval()
    return model.to(device)


# --------------------------------------------------------------------------- #
# Tiền xử lý / suy luận
# --------------------------------------------------------------------------- #
def normalize(img: np.ndarray) -> np.ndarray:
    """[0,1] → [-1,1] (chuẩn hoá cố định lúc train Stage-1)."""
    return (img.astype(np.float32) - 0.5) / 0.5


def to_tensor(img: np.ndarray) -> torch.Tensor:
    """ảnh xám (H,W) → tensor (1,1,H,W) đã chuẩn hoá."""
    x = torch.from_numpy(np.ascontiguousarray(normalize(img)))[None, None]
    return x.float()


@torch.no_grad()
def predict_mask(
    model: nn.Module,
    gray,
    device: str = "cpu",
    img_size: int = IMG_SIZE,
    threshold: float = 0.5,
) -> np.ndarray:
    """Sinh mask phổi từ ảnh xám float [0,1] hoặc đường dẫn ảnh.

    Trả về `np.float32 (img_size, img_size)` giá trị 0/1 — đúng định dạng
    `dl.preprocess.lung_mask` để Stage-2 dùng thay mask CheXmask.
    """
    if isinstance(gray, (str, Path)):
        gray = load_gray(gray, img_size)
    elif gray.shape != (img_size, img_size):
        gray = np.array(
            Image.fromarray((gray * 255).astype(np.uint8)).convert("L").resize(
                (img_size, img_size)
            )
        ).astype(np.float32) / 255.0

    logits = model(to_tensor(gray).to(device))
    prob = torch.sigmoid(logits)[0, 0].cpu().numpy()
    return (prob > threshold).astype(np.float32)


def load_gray(path, img_size: int = IMG_SIZE) -> np.ndarray:
    """Ảnh xám float32 [0,1] đã resize (cùng `dl.preprocess.load_gray`)."""
    img = Image.open(path).convert("L").resize((img_size, img_size))
    return np.array(img).astype(np.float32) / 255.0


# --------------------------------------------------------------------------- #
# Đo chất lượng
# --------------------------------------------------------------------------- #
def dice_coef(pred: np.ndarray, target: np.ndarray, eps: float = 1e-6) -> float:
    """Dice cho mask nhị phân. Ảnh không có phổi (hiếm) → 1.0 nếu cùng rỗng."""
    pred = pred.astype(np.float32)
    target = target.astype(np.float32)
    inter = float((pred * target).sum())
    return (2 * inter + eps) / (pred.sum() + target.sum() + eps)


def iou_score(pred: np.ndarray, target: np.ndarray, eps: float = 1e-6) -> float:
    pred = pred.astype(np.float32)
    target = target.astype(np.float32)
    inter = float((pred * target).sum())
    union = float(pred.sum() + target.sum() - inter)
    return (inter + eps) / (union + eps)


class BCEDiceLoss(nn.Module):
    """BCE (ổn định gradient) + Dice (chịu được mất cân bằng nền/phổi).

    Dice tính theo từng ảnh trong batch rồi lấy trung bình — tránh việc một ảnh
    có phổi to lấn át các ảnh còn lại.
    """

    def __init__(self, bce_weight: float = 0.5) -> None:
        super().__init__()
        self.bce_weight = bce_weight

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, target)
        prob = torch.sigmoid(logits)
        dims = (1, 2, 3)
        inter = (prob * target).sum(dims)
        dice = (2 * inter + 1.0) / (prob.sum(dims) + target.sum(dims) + 1.0)
        return self.bce_weight * bce + (1 - self.bce_weight) * (1 - dice.mean())
