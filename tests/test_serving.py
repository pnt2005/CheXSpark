"""Unit test cho serving: input builder, ước lượng tim, calibration, AUC.

Chạy:  pytest tests/test_serving.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts/dev"))

from dl.preprocess import DISEASES, IMG_SIZE, build_input  # noqa: E402
from serving.pipeline import HEART_FN, heart_closing, heart_scanline, load_calibration  # noqa: E402


# --------------------------------------------------------------------------- #
# build_input — phải khớp CheXSparkDataset.__getitem__ của notebook Kaggle
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("mode", ["3ch", "masked", "none"])
def test_build_input_shape_and_norm(mode: str) -> None:
    gray = np.random.rand(IMG_SIZE, IMG_SIZE).astype(np.float32)
    lung = (np.random.rand(IMG_SIZE, IMG_SIZE) > 0.5).astype(np.float32)
    heart = (np.random.rand(IMG_SIZE, IMG_SIZE) > 0.7).astype(np.float32)

    x = build_input(gray, lung, heart, mode)

    assert x.shape == (1, 3, IMG_SIZE, IMG_SIZE)
    assert x.dtype == torch.float32
    # đã normalize ImageNet → trung bình quanh 0, không còn trong [0,1]
    assert abs(float(x.mean())) < 3.0


def test_build_input_3ch_keeps_original_in_channel0() -> None:
    gray = np.random.rand(IMG_SIZE, IMG_SIZE).astype(np.float32)
    lung = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32)
    lung[:10, :10] = 1.0  # mask chỉ phủ góc

    x = build_input(gray, lung, None, "3ch")
    # De-normalize để so sánh: ch0 = ảnh gốc (không phụ thuộc mask)
    ch0 = x[0, 0] * 0.229 + 0.485
    ch1 = x[0, 1] * 0.224 + 0.456
    assert torch.allclose(ch0, torch.from_numpy(gray), atol=1e-5)
    # ngoài vùng mask thì ch1 = 0
    assert float(ch1[100, 100]) == pytest.approx(0.0, abs=1e-5)


def test_build_input_none_matches_all_gray() -> None:
    gray = np.random.rand(IMG_SIZE, IMG_SIZE).astype(np.float32)
    x = build_input(gray, None, None, "none")
    # Sau ImageNet normalize mỗi kênh có mean/std khác nhau → so sánh bằng cách
    # khử normalize về [0,1] (đúng khâu norm của notebook train)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    raw = x[0] * std + mean
    assert torch.allclose(raw[0], raw[1], atol=1e-5)
    assert torch.allclose(raw[1], raw[2], atol=1e-5)
    assert torch.allclose(raw[0], torch.from_numpy(gray), atol=1e-5)


# --------------------------------------------------------------------------- #
# Ước lượng vùng tim (thay CheXmask khi suy luận)
# --------------------------------------------------------------------------- #
def _two_lungs() -> np.ndarray:
    m = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32)
    m[60:170, 40:100] = 1.0   # phổi trái
    m[60:170, 130:190] = 1.0  # phổi phải (khe trung thất 100..129)
    return m


def test_heart_scanline_fills_gap_between_lungs() -> None:
    lung = _two_lungs()
    heart = heart_scanline(lung)
    # khe trung thất (cùng hàng, giữa 2 phổi) được lấp
    assert heart[100, 115] == 1.0
    # nền (trên/dưới phổi) không bị lấp
    assert heart[10, 115] == 0.0
    # mọi pixel tim + phổi nằm trong vùng lấp theo hàng
    assert (heart[lung > 0.5] == 1.0).all()


def test_heart_closing_is_superset_of_lung() -> None:
    lung = _two_lungs()
    heart = heart_closing(lung, radius=11)
    assert heart.shape == lung.shape
    assert (heart[lung > 0.5] == 1.0).all()  # không mất pixel phổi
    assert heart.sum() >= lung.sum()


def test_heart_modes_registry() -> None:
    lung = _two_lungs()
    assert HEART_FN["none"](lung) is None
    for mode in ("closing", "scanline"):
        out = HEART_FN[mode](lung)
        assert out.shape == lung.shape and set(np.unique(out)) <= {0.0, 1.0}


def test_heart_scanline_empty_mask() -> None:
    empty = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32)
    assert heart_scanline(empty).sum() == 0
    assert heart_closing(empty).sum() == 0


# --------------------------------------------------------------------------- #
# Calibration
# --------------------------------------------------------------------------- #
def test_load_calibration_missing_file_gives_defaults(tmp_path: Path) -> None:
    cal = load_calibration(tmp_path / "nope.json")
    assert cal["mask_input"] == "3ch"
    assert np.allclose(cal["temperature"], 1.0)
    assert np.allclose(cal["threshold"], 0.5)
    assert cal["source"] is None


def test_load_calibration_real_metrics() -> None:
    path = ROOT / "notebooks/kaggle/stage2_metrics.json"
    if not path.exists():
        pytest.skip("chưa có stage2_metrics.json")
    cal = load_calibration(path)
    assert cal["mask_input"] == "3ch"
    assert cal["mask_mode"] == "lungs_heart"
    assert cal["temperature"].shape == (len(DISEASES),)
    assert cal["threshold"].shape == (len(DISEASES),)
    assert (cal["temperature"] > 0).all()
    assert ((cal["threshold"] > 0) & (cal["threshold"] < 1)).all()
    assert cal["test_mauroc"] > 0.7


# --------------------------------------------------------------------------- #
# AUC (dùng trong benchmark mask)
# --------------------------------------------------------------------------- #
def test_auc_rank_perfect_and_random() -> None:
    from benchmark_serving_mask import auc_rank

    y = np.array([0, 0, 1, 1])
    assert auc_rank(y, np.array([0.1, 0.2, 0.8, 0.9])) == 1.0
    assert auc_rank(y, np.array([0.9, 0.8, 0.2, 0.1])) == 0.0
    # điểm bằng nhau hoàn toàn → 0.5
    assert auc_rank(y, np.ones(4)) == pytest.approx(0.5)
    # một hạng duy nhất → undefined
    assert np.isnan(auc_rank(np.zeros(4, dtype=int), np.arange(4.0)))


def test_auc_rank_matches_sklearn_when_available() -> None:
    from benchmark_serving_mask import auc_rank

    try:
        from sklearn.metrics import roc_auc_score
    except ImportError:
        pytest.skip("không có sklearn để đối chiếu")
    rng = np.random.default_rng(0)
    y = (rng.random(500) > 0.7).astype(int)
    s = rng.random(500)
    assert auc_rank(y, s) == pytest.approx(roc_auc_score(y, s), abs=1e-9)


# --------------------------------------------------------------------------- #
# Engine end-to-end (checkpoint local, không cần MLflow)
# --------------------------------------------------------------------------- #
def test_engine_predict_local() -> None:
    stage1 = ROOT / "notebooks/kaggle/best_stage1_unet.pth"
    stage2 = ROOT / "notebooks/kaggle/best_stage2_densenet121.pth"
    if not (stage1.exists() and stage2.exists()):
        pytest.skip("chưa có checkpoint local")

    from serving.pipeline import InferenceEngine

    engine = InferenceEngine(use_registry=False, heart_mode="scanline")
    img = np.random.rand(IMG_SIZE, IMG_SIZE).astype(np.float32)  # ảnh giả
    out = engine.predict(img, want_gradcam=True)

    assert len(out["findings"]) == 14
    assert {f["disease_name"] for f in out["findings"]} == set(DISEASES)
    assert all(0.0 <= f["probability"] <= 1.0 for f in out["findings"])
    assert all(0.0 < f["threshold"] < 1.0 for f in out["findings"])
    assert out["images"]["heatmap"].shape == (IMG_SIZE, IMG_SIZE, 3)
    assert out["images"]["mask"].shape == (IMG_SIZE, IMG_SIZE, 3)
    assert out["timings_ms"]["total"] > 0
    assert out["model"]["stage2"]["source"] == "local"

    # tên bệnh sai → ValueError
    with pytest.raises(ValueError, match="Không có bệnh"):
        engine.predict(img, disease="Foo", want_gradcam=False)
