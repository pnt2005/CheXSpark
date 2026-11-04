"""Tiền xử lý ảnh CXR + mask CheXmask cho Stage-2/Stage-3 (dùng chung DL & serving).

Giữ **đúng** pipeline lúc train Stage-2: grayscale → resize 224 → nhân lung mask
→ 3 kênh → ImageNet normalize. Mọi nơi (train/inference/Grad-CAM) phải dùng cùng
hàm này để tránh lệch phân phối.
"""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

IMG_SIZE = 224
DISEASES = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass", "Nodule",
    "Pneumonia", "Pneumothorax", "Consolidation", "Edema", "Emphysema",
    "Fibrosis", "Pleural_Thickening", "Hernia",
]
LABEL_COLS = [f"label_{d}" for d in DISEASES]
NORM = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])


def decode_rle(rle, h: int, w: int) -> np.ndarray:
    """CheXmask RLE: chuỗi cặp (start, length), reshape row-major (C order)."""
    vals = [int(v) for v in str(rle).split()]
    flat = np.zeros(h * w, dtype=np.uint8)
    for i in range(0, len(vals), 2):
        flat[vals[i]: vals[i] + vals[i + 1]] = 1
    return flat.reshape((h, w))


def lung_mask(row, img_size: int = IMG_SIZE, include_heart: bool = False) -> np.ndarray:
    """Mask phổi (tuỳ chọn + tim) resize về img_size, dạng float32 0/1."""
    h, w = int(row["mask_height"]), int(row["mask_width"])
    m = decode_rle(row["mask_left_rle"], h, w) | decode_rle(row["mask_right_rle"], h, w)
    if include_heart:
        m = m | decode_rle(row["mask_heart_rle"], h, w)
    m = Image.fromarray((m * 255).astype(np.uint8)).resize(
        (img_size, img_size), Image.NEAREST
    )
    return (np.array(m) > 127).astype(np.float32)


def load_gray(path, img_size: int = IMG_SIZE) -> np.ndarray:
    """Ảnh xám float32 [0,1], đã resize."""
    img = Image.open(path).convert("L").resize((img_size, img_size))
    return np.array(img).astype(np.float32) / 255.0


def build_tensor(gray: np.ndarray, mask: np.ndarray | None = None) -> torch.Tensor:
    """(H,W) xám (+mask) → tensor (1,3,H,W) đã normalize."""
    arr = gray * mask if mask is not None else gray
    x = np.stack([arr] * 3, axis=0)
    return NORM(torch.from_numpy(x)).unsqueeze(0)


def build_input(
    gray: np.ndarray,
    lung: np.ndarray | None = None,
    heart: np.ndarray | None = None,
    mask_input: str = "3ch",
) -> torch.Tensor:
    """Ghép ảnh + mask thành input Stage-2 — **khớp tuyệt đối** notebook train.

    Ba chế độ (xem `CheXSparkDataset.__getitem__` trong notebook Kaggle):

        3ch    ch0 = ảnh gốc (giữ trung thất/tim), ch1 = ảnh×phổi,
               ch2 = ảnh×(phổi∪ tim)   ← cấu hình của model production
        masked ch0=ch1=ch2 = ảnh×phổi   (cách cũ, hại Cardiomegaly)
        none   ch0=ch1=ch2 = ảnh gốc    (không mask)

    Args:
        gray: ảnh xám float32 [0,1] cỡ IMG_SIZE
        lung: mask phổi 0/1 cùng cỡ (U-Net Stage-1 khi suy luận)
        heart: mask tim 0/1 cùng cỡ (CheXmask khi train; proxy khi suy luận)
        mask_input: "3ch" | "masked" | "none" (lấy từ metrics config)
    """
    g = torch.from_numpy(np.ascontiguousarray(gray, dtype=np.float32))[None]  # (1,H,W)

    if mask_input == "none" or lung is None:
        x = torch.cat([g] * 3, dim=0)
    elif mask_input == "masked":
        lm = torch.from_numpy(np.asarray(lung, dtype=np.float32))[None]
        x = torch.cat([g * lm] * 3, dim=0)
    else:  # "3ch"
        lm = torch.from_numpy(np.asarray(lung, dtype=np.float32))[None]
        hm = (
            torch.from_numpy(np.asarray(heart, dtype=np.float32))[None]
            if heart is not None
            else torch.zeros_like(lm)
        )
        x = torch.cat([g, g * lm, g * torch.clamp(lm + hm, 0, 1)], dim=0)

    return NORM(x).unsqueeze(0)
