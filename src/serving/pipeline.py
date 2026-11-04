"""Pipeline suy luận 3 giai đoạn cho serving: U-Net → DenseNet121 → Grad-CAM.

    ảnh CXR ──► [U-Net Stage-1] ──► mask phổi ──┐
                                               ├─► input 3 kênh ──► [DenseNet121]
                                               │        (xem dl.preprocess.build_input)
                                               └──────────────────────► [Grad-CAM] ──► heatmap

Điểm then chốt so với lúc train trên Kaggle:

1. **Mask**: train dùng CheXmask (sinh sẵn cho 112k ảnh NIH); suy luận trên ảnh
   mới từ bệnh viện **không có** CheXmask → dùng U-Net Stage-1. Model production
   huấn luyện với `mask_input=3ch`, kênh 2 = ảnh×(phổi∪ tim) nên serving phải
   tự ước lượng vùng tim (`heart_proxy` — xem `scripts/dev/benchmark_serving_mask.py`
   để đo chênh lệch so với CheXmask thật).
2. **Calibration**: xác suất đi qua temperature per-label + ngưỡng Youden per-label
   lấy từ `stage2_metrics.json` (fit trên val lúc train) — không dùng 0.5 cứng.

Nạp model: ưu tiên **MLflow registry theo alias** (`models:/…@production`, xem
`mlops.registry`) — fallback checkpoint local khi MLflow/MinIO không chạy.

Usage::

    engine = InferenceEngine()                 # registry → local fallback
    out = engine.predict("some.png")           # dict findings + mảng ảnh
"""
from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from dl.gradcam import GradCAM, mask_visual, overlay_heatmap
from dl.model import load_stage2_model
from dl.preprocess import DISEASES, IMG_SIZE, build_input, load_gray
from dl.segmentation import load_stage1_model, predict_mask

# Repo root — cho fallback checkpoint local (container: /app, host: <repo>)
ROOT = Path(os.getenv("CHEXSPARK_ROOT", Path(__file__).resolve().parents[2]))
DEFAULT_STAGE1_CKPT = Path(
    os.getenv("SERVING_STAGE1_CKPT", ROOT / "notebooks/kaggle/best_stage1_unet.pth")
)
DEFAULT_STAGE2_CKPT = Path(
    os.getenv("SERVING_STAGE2_CKPT", ROOT / "notebooks/kaggle/best_stage2_densenet121.pth")
)
DEFAULT_CALIBRATION = Path(
    os.getenv("SERVING_CALIBRATION", ROOT / "notebooks/kaggle/stage2_metrics.json")
)

# Ưu tiên registry (MLOps) → tắt bằng SERVING_USE_REGISTRY=0 nếu muốn load .pth
USE_REGISTRY = os.getenv("SERVING_USE_REGISTRY", "1").strip().lower() not in ("0", "false", "no")

HEART_MODES = ("none", "closing", "scanline")


# --------------------------------------------------------------------------- #
# Ước lượng vùng tim khi suy luận (thay CheXmask heart RLE)
# --------------------------------------------------------------------------- #
def _disk(radius: int) -> np.ndarray:
    y, x = np.ogrid[-radius : radius + 1, -radius : radius + 1]
    return (x * x + y * y) <= radius * radius


def heart_closing(lung: np.ndarray, radius: int = 11) -> np.ndarray:
    """Đóng morphology + lấp lỗ → lấp khoảng trống giữa hai phổi (trung thất/tim).

    `binary_closing` với đĩa bán kính `radius` (pixel@224) nối hai biên trong
    rồi `binary_fill_holes` giữ lại phần lõm — xấp xỉ bóng tim/trung thất.
    """
    from scipy import ndimage

    m = np.asarray(lung) > 0.5
    if not m.any():
        return np.zeros_like(m, dtype=np.float32)
    m = ndimage.binary_closing(m, structure=_disk(int(radius)))
    m = ndimage.binary_fill_holes(m)
    return m.astype(np.float32)


def heart_scanline(lung: np.ndarray) -> np.ndarray:
    """Lấp ngang mỗi hàng: từ cột phổi trái nhất → phải nhất.

    Đáng tin hơn `closing` khi khe trung thất rộng (đóng không với tới): mọi pixel
    nằm giữa hai phổi trên cùng hàng đều thuộc tim/trung thất. Vùng lấp chỉ xuất
    hiện ở hàng có phổi nên không tràn ra nền.
    """
    m = np.asarray(lung) > 0.5
    out = np.zeros_like(m, dtype=np.float32)
    for y in range(m.shape[0]):
        xs = np.flatnonzero(m[y])
        if xs.size:
            out[y, xs[0] : xs[-1] + 1] = 1.0
    return out


HEART_FN = {"none": lambda lung: None, "closing": heart_closing, "scanline": heart_scanline}


# --------------------------------------------------------------------------- #
# Calibration (temperature + ngưỡng per-label từ stage2_metrics.json)
# --------------------------------------------------------------------------- #
def load_calibration(path: str | Path = DEFAULT_CALIBRATION) -> dict:
    """Đọc `stage2_metrics.json` → temperature/threshold/mask_input của model.

    Thiếu file (hoặc thiếu khoá) → mặc định T=1 (không scale), ngưỡng 0.5,
    `mask_input="3ch"` — serving vẫn chạy được nhưng mất calibration.
    """
    path = Path(path)
    raw: dict = {}
    if path.exists():
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            print(f"[serving] không đọc được calibration ({exc}) → mặc định")

    cal = raw.get("calibration") or {}
    cfg = raw.get("config") or {}
    n = len(DISEASES)
    temperature = np.asarray(cal.get("temperature") or [1.0] * n, dtype=np.float32)
    threshold = np.asarray(cal.get("threshold") or [0.5] * n, dtype=np.float32)
    if temperature.shape != (n,):
        temperature = np.full(n, 1.0, dtype=np.float32)
    if threshold.shape != (n,):
        threshold = np.full(n, 0.5, dtype=np.float32)

    return {
        "temperature": temperature,
        "threshold": threshold,
        "mask_input": str(cfg.get("mask_input", "3ch")),
        "mask_mode": str(cfg.get("mask_mode", "lungs_heart")),
        "test_mauroc": raw.get("test_auc"),
        "method": cal.get("method"),
        "source": str(path) if path.exists() else None,
    }


# --------------------------------------------------------------------------- #
# Engine
# --------------------------------------------------------------------------- #
class InferenceEngine:
    """Giữ model trong RAM + serialise mọi lần suy luận bằng lock (an toàn đa luồng)."""

    def __init__(
        self,
        *,
        device: str | None = None,
        stage1_ckpt: str | Path | None = None,
        stage2_ckpt: str | Path | None = None,
        calibration_path: str | Path | None = None,
        heart_mode: str | None = None,
        use_registry: bool = USE_REGISTRY,
    ) -> None:
        self.device = device or os.getenv("SERVING_DEVICE") or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.lock = threading.Lock()

        self.stage2, self.stage2_info = self._load_stage2(
            use_registry, Path(stage2_ckpt or DEFAULT_STAGE2_CKPT), self.device
        )
        self.stage1, self.stage1_info = self._load_stage1(
            use_registry, Path(stage1_ckpt or DEFAULT_STAGE1_CKPT), self.device
        )

        cal = load_calibration(calibration_path or DEFAULT_CALIBRATION)
        self.temperature = torch.from_numpy(cal["temperature"])
        self.threshold = cal["threshold"]
        self.mask_input = cal["mask_input"]
        self.heart_mode = (heart_mode or os.getenv("SERVING_HEART_MODE") or "scanline").lower()
        if self.heart_mode not in HEART_MODES:
            raise ValueError(f"heart_mode phải là {HEART_MODES}, nhận {self.heart_mode!r}")

        self.calibration = cal
        # Hook Grad-CAM gắn một lần, tái sử dụng cho mọi request
        self.gradcam = GradCAM(self.stage2)

        self.info = {
            "stage1": self.stage1_info,
            "stage2": self.stage2_info,
            "device": self.device,
            "mask_input": self.mask_input,
            "mask_mode": cal["mask_mode"],
            "heart_mode": self.heart_mode if self.mask_input != "none" else None,
            "calibration": {
                "method": cal["method"],
                "source": cal["source"],
                "test_mauroc": cal["test_mauroc"],
                "threshold": {d: round(float(t), 3) for d, t in zip(DISEASES, self.threshold)},
            },
            "img_size": IMG_SIZE,
        }

    # ---------------- nạp model ---------------- #
    @staticmethod
    def _load_stage2(use_registry: bool, ckpt: Path, device: str):
        if use_registry:
            try:
                from mlops.registry import load_stage2_from_registry

                model = load_stage2_from_registry(device=device)
                return model, {"source": "registry", "uri": "models:/chexspark-stage2-densenet121@production"}
            except Exception as exc:  # MLflow/MinIO chưa chạy → fallback local
                print(f"[serving] registry stage2 lỗi: {exc!r} → dùng checkpoint local")
        if not ckpt.exists():
            raise FileNotFoundError(
                f"Không nạp được Stage-2: thiếu cả registry lẫn checkpoint {ckpt}"
            )
        return load_stage2_model(ckpt, device), {"source": "local", "ckpt": str(ckpt)}

    @staticmethod
    def _load_stage1(use_registry: bool, ckpt: Path, device: str):
        if use_registry:
            try:
                from mlops.registry import load_stage1_from_registry

                model = load_stage1_from_registry(device=device)
                return model, {"source": "registry", "uri": "models:/chexspark-stage1-unet@production"}
            except Exception as exc:
                print(f"[serving] registry stage1 lỗi: {exc!r} → dùng checkpoint local")
        if not ckpt.exists():
            raise FileNotFoundError(
                f"Không nạp được Stage-1: thiếu cả registry lẫn checkpoint {ckpt}"
            )
        return load_stage1_model(ckpt, device), {"source": "local", "ckpt": str(ckpt)}

    # ---------------- ảnh đầu vào ---------------- #
    @staticmethod
    def _gray(image: bytes | str | Path | np.ndarray | Image.Image) -> np.ndarray:
        """Chuẩn hoá mọi kiểu input về ảnh xám float32 [0,1] cỡ IMG_SIZE."""
        if isinstance(image, (bytes, bytearray)):
            import io

            img = Image.open(io.BytesIO(bytes(image))).convert("L")
            return np.array(img.resize((IMG_SIZE, IMG_SIZE))).astype(np.float32) / 255.0
        if isinstance(image, Image.Image):
            img = image.convert("L").resize((IMG_SIZE, IMG_SIZE))
            return np.array(img).astype(np.float32) / 255.0
        if isinstance(image, np.ndarray):
            arr = image
            if arr.ndim == 3:  # RGB/RGBA → xám
                arr = arr[..., :3].mean(axis=-1)
            arr = arr.astype(np.float32)
            if arr.max() > 1.5:
                arr = arr / 255.0
            if arr.shape != (IMG_SIZE, IMG_SIZE):
                arr = np.array(
                    Image.fromarray((arr * 255).astype(np.uint8)).resize(
                        (IMG_SIZE, IMG_SIZE)
                    )
                ).astype(np.float32) / 255.0
            return arr
        return load_gray(image)  # đường dẫn file

    # ---------------- suy luận ---------------- #
    def predict(
        self,
        image: bytes | str | Path | np.ndarray | Image.Image,
        *,
        disease: str | None = None,
        want_gradcam: bool = True,
    ) -> dict:
        """Chạy đủ 3 giai đoạn. Trả findings + mảng ảnh (numpy) cho layer API encode."""
        with self.lock:
            t0 = time.perf_counter()
            gray = self._gray(image)
            t1 = time.perf_counter()

            lung = predict_mask(self.stage1, gray, device=self.device)
            heart = None
            if self.mask_input != "none":
                heart = HEART_FN[self.heart_mode](lung)
            t2 = time.perf_counter()

            x = build_input(gray, lung, heart, self.mask_input)
            if want_gradcam:
                logits = self.stage2(x)  # cần grad cho backward của Grad-CAM
            else:
                with torch.no_grad():
                    logits = self.stage2(x)
            # .detach(): khi want_gradcam=True logits còn grad → không gọi numpy() trực tiếp
            probs = torch.sigmoid(logits / self.temperature)[0].detach().cpu().numpy()
            t3 = time.perf_counter()

            idx = self._gradcam_index(probs, disease)
            heat = None
            if want_gradcam:
                cam = self.gradcam.generate(x, idx)
                heat = overlay_heatmap(gray, cam, mask=lung)
            t4 = time.perf_counter()

            findings = [
                {
                    "disease_name": name,
                    "probability": round(float(probs[i]), 4),
                    "threshold": round(float(self.threshold[i]), 3),
                    "is_positive": bool(probs[i] >= self.threshold[i]),
                }
                for i, name in enumerate(DISEASES)
            ]

            return {
                "findings": findings,
                "top_disease": DISEASES[int(np.argmax(probs))],
                "gradcam_disease": DISEASES[idx],
                "images": {
                    "original": gray,
                    "mask": mask_visual(gray, lung),
                    "masked": (gray * lung),
                    "heatmap": heat,
                },
                "mask_coverage": round(float(lung.mean()), 4),
                "timings_ms": {
                    "preprocess": round((t1 - t0) * 1e3, 1),
                    "segmentation": round((t2 - t1) * 1e3, 1),
                    "classification": round((t3 - t2) * 1e3, 1),
                    "gradcam": round((t4 - t3) * 1e3, 1),
                    "total": round((t4 - t0) * 1e3, 1),
                },
                "model": {
                    "stage1": self.stage1_info,
                    "stage2": self.stage2_info,
                    "mask_input": self.mask_input,
                    "heart_mode": self.heart_mode if self.mask_input != "none" else None,
                    "calibration_method": self.calibration["method"],
                },
            }

    @staticmethod
    def _gradcam_index(probs: np.ndarray, disease: str | None) -> int:
        """Chọn bệnh vẽ heatmap: chỉ định sẵn > bệnh có xác suất cao nhất."""
        if disease:
            key = disease.strip().lower()
            for i, name in enumerate(DISEASES):
                if name.lower() == key:
                    return i
            raise ValueError(
                f"Không có bệnh {disease!r}. Danh sách: {', '.join(DISEASES)}"
            )
        return int(np.argmax(probs))
