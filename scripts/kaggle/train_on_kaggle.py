"""Train Stage-1/Stage-2 tự động trên Kaggle: upload dataset → push notebook → chờ → tải output.

Máy local không có GPU, nên train thật diễn ra trên Kaggle (T4 ×2). Script này gói
toàn bộ vòng đó thành 1 lệnh, thay vì thao tác tay trên web:

    1. (tuỳ chọn) cập nhật dataset `chexspark-stage2-nih-dev` (CSV + RLE mask)
    2. (tuỳ chọn) cập nhật dataset `chexspark-masks-224` (mask precompute, bỏ decode RLE)
    3. Sinh `kernel-metadata.json` + notebook đã patch config (epochs/base/lr…)
    4. `kaggle kernels push` → Kaggle chạy notebook trên GPU
    5. Poll `kaggle kernels status` tới khi complete/error (in log nếu lỗi)
    6. `kaggle kernels output` → tải checkpoint + metrics json (cả khi lỗi/timeout)
    7. (tuỳ chọn) copy vào `notebooks/kaggle/` và đăng ký MLflow Registry

Dùng **Kaggle CLI** (subprocess) chứ không dùng SDK Python: CLI giữ nguyên
subcommand qua các phiên bản (SDK 2.x đã bỏ `dataset_create_new`), và tự xử lý
credential (kaggle.json legacy, `KAGGLE_API_TOKEN`, hoặc `kaggle auth login`).

Chuẩn bị (một lần):
    pip install kaggle
    kaggle auth login                       # hoặc tạo ~/.kaggle/kaggle.json
    # GPU cần xác thực số điện thoại trong Kaggle Settings

Usage:
    # Kiểm tra credential + cấu hình trước khi đẩy
    python scripts/kaggle/train_on_kaggle.py --stage 1 --check

    # Train Stage-1 trên GPU, tải kết quả về và đăng ký MLflow
    python scripts/kaggle/train_on_kaggle.py --stage 1 --upload-dataset --register

    # Train Stage-2 với cấu hình cải tiến (mask 3 kênh + aug + TTA + calibration)
    python scripts/kaggle/train_on_kaggle.py --stage 2 --upload-masks --epochs 30 --register

    # Ablation: tắt aug/TTA để so sánh, hoặc bỏ mask hoàn toàn (đối chứng)
    python scripts/kaggle/train_on_kaggle.py --stage 2 --masks-dataset pnt2005/chexspark-masks-224 \
        --no-aug --no-tta --mask-input masked
    python scripts/kaggle/train_on_kaggle.py --stage 2 --mask-input none --no-aug

    # Kaggle cắt session 12h -> chạy tiếp từ checkpoint của lần trước
    python scripts/kaggle/train_on_kaggle.py --stage 2 --epochs 30 \\
        --resume-from pnt2005/chexspark-stage2-densenet121

    # Chỉ build + in lệnh, không gọi Kaggle
    python scripts/kaggle/train_on_kaggle.py --stage 1 --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))


def load_dotenv(path: Path) -> None:
    """Nạp `.env` (KEY=VALUE) vào os.environ nếu chưa có — khỏi phải export tay."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        key, val = key.strip(), val.strip().strip('"').strip("'")
        if key and val:
            os.environ.setdefault(key, val)


load_dotenv(ROOT / ".env")
# Kaggle CLI >= 2.0 xác thực bằng KAGGLE_API_TOKEN (token dạng KGAT_...);
# .env cũ để KAGGLE_USERNAME/KAGGLE_KEY nên map sang tên mới.
if not os.getenv("KAGGLE_API_TOKEN") and os.getenv("KAGGLE_KEY"):
    os.environ["KAGGLE_API_TOKEN"] = os.environ["KAGGLE_KEY"]

# Dataset ảnh 224×224 của Kaggle (dùng chung cho cả 2 stage)
IMAGE_DATASET = "khanfashee/nih-chest-x-ray-14-224x224-resized"
DEFAULT_DATASET_SLUG = "chexspark-stage2-nih-dev"
DEFAULT_MASKS_SLUG = "chexspark-masks-224"

STAGES = {
    1: {
        "notebook": ROOT / "notebooks/kaggle/stage1_unet_segmentation.ipynb",
        "kernel_slug": "chexspark-stage1-unet",
        "title": "CheXSpark Stage1 UNet",
        "metrics": "stage1_metrics.json",
        "checkpoint": "best_stage1_unet.pth",
        "last_checkpoint": "last_stage1_unet.pth",
        "patch": {"BASE": "32", "EPOCHS": "30", "BATCH_SIZE": "16", "LR": "1e-3"},
        "register": "register_stage1_model",
        "primary": "test_dice_mean",
    },
    2: {
        "notebook": ROOT / "notebooks/kaggle/stage2_densenet121.ipynb",
        "kernel_slug": "chexspark-stage2-densenet121",
        "title": "CheXSpark Stage2 DenseNet121",
        "metrics": "stage2_metrics.json",
        "checkpoint": "best_stage2_densenet121.pth",
        "last_checkpoint": "last_stage2_densenet121.pth",
        "patch": {"EPOCHS": "8", "BATCH_SIZE": "32", "LR": "1e-4", "NUM_WORKERS": "4",
                  "MASK_INPUT": "'3ch'", "MASK_MODE": "'lungs_heart'",
                  "AUG": "True", "TTA": "True", "CALIBRATE": "True"},
        "register": "register_stage2_model",
        "primary": "test_mauroc",
    },
}


# --------------------------------------------------------------------------- #
# Kaggle CLI wrapper
# --------------------------------------------------------------------------- #
def kaggle_bin() -> str:
    """Tìm lệnh `kaggle` (env KAGGLE_BIN → PATH → venv đang chạy script)."""
    explicit = os.getenv("KAGGLE_BIN")
    if explicit and Path(explicit).exists():
        return explicit
    found = shutil.which("kaggle")
    if found:
        return found
    candidate = Path(sys.executable).with_name("kaggle.exe" if os.name == "nt" else "kaggle")
    if candidate.exists():
        return str(candidate)
    raise SystemExit(
        "Không tìm thấy Kaggle CLI. Cài bằng:\n"
        f"  {sys.executable} -m pip install kaggle\n"
        "rồi xác thực: kaggle auth login (hoặc đặt ~/.kaggle/kaggle.json)"
    )


def run(cmd: list[str], *, check: bool = True, timeout: int | None = None) -> subprocess.CompletedProcess:
    """Chạy CLI, in lệnh (che credential) và trả output."""
    print(f"$ {' '.join(str(c) for c in cmd)}")
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                          encoding="utf-8", errors="replace")
    if proc.stdout.strip():
        print(proc.stdout.strip())
    if proc.returncode != 0:
        if proc.stderr.strip():
            print(proc.stderr.strip(), file=sys.stderr)
        if check:
            raise SystemExit(f"Lệnh thất bại (exit {proc.returncode}): {' '.join(cmd[:3])} …")
    return proc


def detect_username(kaggle: str) -> str | None:
    """Lấy username từ env / kaggle.json / `kaggle config view`."""
    user = os.getenv("KAGGLE_USERNAME")
    if user:
        return user
    cfg = Path.home() / ".kaggle" / "kaggle.json"
    if cfg.exists():
        try:
            return json.loads(cfg.read_text(encoding="utf-8")).get("username")
        except json.JSONDecodeError:
            pass
    proc = run([kaggle, "config", "view"], check=False)
    match = re.search(r"^username:\s*(\S+)", proc.stdout or "", re.MULTILINE)
    return match.group(1) if match else None


# --------------------------------------------------------------------------- #
# Build kernel folder
# --------------------------------------------------------------------------- #
def patch_notebook(nb_path: Path, values: dict[str, str], out_path: Path) -> list[str]:
    """Ghi bản notebook đã thay giá trị các hằng CONFIG (EPOCHS, BASE, ...)."""
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    changed: list[str] = []

    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        src = "".join(cell["source"])
        if "CONFIG" not in src:
            continue
        for key, value in values.items():
            # giữ comment cuối dòng khi thay giá trị
            pattern = re.compile(rf"^(\s*{key}\s*=\s*)([^#\n]*)(#[^\n]*)?$", re.MULTILINE)

            def _sub(m: re.Match, v: str = value) -> str:
                comment = f" {m.group(3)}" if m.group(3) else ""
                return f"{m.group(1)}{v}{comment}"

            src, n = pattern.subn(_sub, src)
            if n:
                changed.append(f"{key}={value}")
        cell["source"] = src.splitlines(True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    return changed


def slugify_title(text: str) -> str:
    """Kaggle tự sinh slug kernel từ title (giống Clean URL): lowercase, gạch nối."""
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def build_kernel_dir(stage: int, cfg: dict, username: str, dataset_slug: str,
                     overrides: dict[str, str]) -> Path:
    """Tạo `data/kaggle/kernels/stageN/` gồm notebook (đã patch) + kernel-metadata.json."""
    slug = cfg["kernel_slug"] if not overrides.get("kernel_slug") else overrides["kernel_slug"]
    title = cfg["title"]
    # Kaggle từ chối push (409 Conflict) nếu slug(title) != id trong metadata,
    # nên luôn chỉnh title cho khớp slug mong muốn.
    if slugify_title(title) != slug:
        fixed = slug.replace("-", " ")
        print(f"[WARN] title {title!r} -> slug {slugify_title(title)!r} != id {slug!r}; "
              f"dùng title {fixed!r}")
        title = fixed
    img_ref = overrides.get("image_dataset") or IMAGE_DATASET
    meta_ref = f"{username}/{dataset_slug.split('/')[-1]}"
    masks_ref = overrides.get("masks_dataset")
    resume_ref = overrides.get("resume_from")

    kernel_id = f"{username}/{slug}"
    build_dir = ROOT / "data/kaggle/kernels" / f"stage{stage}"
    nb_out = build_dir / cfg["notebook"].name

    values = dict(cfg["patch"])
    # Kaggle 2025+ mount dataset tại /kaggle/input/datasets/<owner>/<slug> -> notebook
    # tự dò bằng resolve_input() nên chỉ cần truyền ref, không phải path tuyệt đối.
    values["IMG_DS"] = f"'{img_ref}'"
    values["META_DS"] = f"'{meta_ref}'"
    values["MASKS_DS"] = f"'{masks_ref}'" if masks_ref else "''"
    if resume_ref:
        values["RESUME_FROM"] = (
            f"'/kaggle/input/{resume_ref.split('/')[-1]}/{cfg['last_checkpoint']}'"
        )
    for key in ("epochs", "base", "batch_size", "lr", "mask_mode", "img_size",
                "mask_input", "num_workers"):
        if overrides.get(key) is not None:
            value = str(overrides[key])
            if key in ("mask_mode", "mask_input") and not value.startswith(("'", '"')):
                value = f"'{value}'"   # MASK_MODE / MASK_INPUT là chuỗi
            values[key.upper()] = value
    for key in ("aug", "tta", "calibrate"):
        if overrides.get(key) is not None:
            values[key.upper()] = str(bool(overrides[key]))
    changed = patch_notebook(cfg["notebook"], values, nb_out)

    dataset_sources = [img_ref, meta_ref] + ([masks_ref] if masks_ref else [])
    metadata = {
        "id": kernel_id,
        "title": title,
        "code_file": nb_out.name,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": bool(overrides.get("gpu", True)),
        "enable_tpu": False,
        "enable_internet": True,
        "dataset_sources": dataset_sources,
        "competition_sources": [],
        # kernel_sources: mount output của lần chạy trước vào /kaggle/input/<slug>
        "kernel_sources": [resume_ref] if resume_ref else [],
        "model_sources": [],
    }
    (build_dir / "kernel-metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nKernel folder: {build_dir}")
    print(f"  id           : {kernel_id}")
    print(f"  notebook     : {nb_out.name}")
    print(f"  datasets     : {', '.join(dataset_sources)}")
    print(f"  kernel source: {resume_ref or '(không — train từ đầu)'}")
    print(f"  GPU          : {metadata['enable_gpu']}")
    if changed:
        print(f"  config patch : {', '.join(changed)}")
    return build_dir


# --------------------------------------------------------------------------- #
# Dataset upload
# --------------------------------------------------------------------------- #
def upload_dataset(kaggle: str, username: str, dataset_slug: str, dataset_dir: Path,
                   notes: str, wait: bool = True) -> None:
    """Tạo dataset mới hoặc version mới cho CSV + RLE mask."""
    if not dataset_dir.exists():
        raise SystemExit(f"Không thấy thư mục dataset: {dataset_dir}")

    meta_path = dataset_dir / "dataset-metadata.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8-sig")) if meta_path.exists() else {}
    meta["id"] = f"{username}/{dataset_slug.split('/')[-1]}"
    meta.setdefault("title", "CheXSpark NIH dev (labels + CheXmask RLE)")
    meta.setdefault("licenses", [{"name": "CC0-1.0"}])
    # Kaggle yêu cầu title 6..50 ký tự — cắt bớt thay vì để `datasets create` fail.
    title = str(meta.get("title", "")).strip()
    if not (6 <= len(title) <= 50):
        meta["title"] = (title[:50].strip() or f"CheXSpark {meta['id'].split('/')[-1]}")[:50]
        print(f"[WARN] title {title!r} không hợp lệ (6..50 ký tự) -> dùng {meta['title']!r}")
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    ref = meta["id"]
    exists = run([kaggle, "datasets", "status", ref], check=False).returncode == 0
    if exists:
        print(f"\nDataset {ref} đã có → tạo version mới")
        run([kaggle, "datasets", "version", "-p", str(dataset_dir), "-m", notes,
             "--dir-mode", "skip"])
    else:
        print(f"\nDataset {ref} chưa có → tạo mới")
        run([kaggle, "datasets", "create", "-p", str(dataset_dir), "--dir-mode", "skip"])

    if not wait:
        return
    print("Chờ Kaggle xử lý dataset …")
    for _ in range(60):
        time.sleep(15)
        status = (run([kaggle, "datasets", "status", ref], check=False).stdout or "").lower()
        if "ready" in status or "ok" in status:
            print("Dataset sẵn sàng.")
            return
        if "error" in status or "fail" in status:
            raise SystemExit(f"Upload dataset lỗi: {status.strip()[:300]}")
    print("[WARN] Hết thời gian chờ dataset — vẫn tiếp tục push kernel.")


# --------------------------------------------------------------------------- #
# Kernel run
# --------------------------------------------------------------------------- #
def wait_for_kernel(kaggle: str, kernel_id: str, *, poll: int,
                    wait_timeout: int) -> tuple[str, str]:
    """Poll status của kernel trên Kaggle tới khi xong. Trả (status, kernel_id)."""
    print(f"\nChờ kernel {kernel_id} (poll {poll}s, tối đa {wait_timeout // 60} phút) …")
    start = time.time()
    while time.time() - start < wait_timeout:
        out = (run([kaggle, "kernels", "status", kernel_id], check=False).stdout or "").lower()
        line = next((ln for ln in out.splitlines() if "kernelworkerstatus" in ln.lower()
                     or "has status" in ln.lower()), "")
        print(f"  [{int(time.time() - start):>4}s] {line.strip() or out.strip()[:120]}")

        if "error" in out or "cancel" in out:
            print("\nKernel lỗi — log cuối:")
            logs = run([kaggle, "kernels", "logs", kernel_id], check=False).stdout or ""
            print("\n".join(logs.splitlines()[-40:]))
            return "error", kernel_id
        if "complete" in out:
            print("\nKernel chạy xong.")
            return "complete", kernel_id
        time.sleep(poll)

    print("[WARN] Hết thời gian chờ — kernel có thể vẫn đang chạy.")
    return "timeout", kernel_id


def push_and_wait(kaggle: str, kernel_id: str, build_dir: Path, *,
                  accelerator: str | None, run_timeout: int, poll: int,
                  wait_timeout: int, no_wait: bool) -> tuple[str, str]:
    """Push notebook và poll tới khi xong.

    Trả (status, kernel_id) — kernel_id có thể KHÁC slug khai báo: khi kernel chưa
    tồn tại, Kaggle tự sinh slug từ `title` (vd thêm hậu tố), nên phải đọc URL
    trong output của `kernels push` rồi dùng slug thật để status/logs/output.
    """
    cmd = [kaggle, "kernels", "push", "-p", str(build_dir), "--timeout", str(run_timeout)]
    if accelerator:
        cmd += ["--accelerator", accelerator]
    proc = run(cmd)

    match = re.search(r"kaggle\.com/code/([\w.-]+/[\w.-]+)", proc.stdout or "")
    if match:
        real_id = match.group(1)
        if real_id != kernel_id:
            print(f"[INFO] Kaggle đặt slug '{real_id}' (khai báo '{kernel_id}') -> dùng slug thật")
        kernel_id = real_id

    if no_wait:
        print(f"\nĐã push (không chờ). Theo dõi: kaggle kernels status {kernel_id}")
        return "queued", kernel_id

    return wait_for_kernel(kaggle, kernel_id, poll=poll, wait_timeout=wait_timeout)


def fetch_output(kaggle: str, kernel_id: str, out_dir: Path) -> Path:
    """Tải output của lần chạy mới nhất."""
    out_dir.mkdir(parents=True, exist_ok=True)
    run([kaggle, "kernels", "output", kernel_id, "-p", str(out_dir), "--force"])
    files = [p for p in out_dir.rglob("*") if p.is_file()]
    print(f"\nĐã tải {len(files)} file vào {out_dir}")
    for f in files[:12]:
        print(f"  {f.relative_to(out_dir)}  ({f.stat().st_size / 1024:.0f} KB)")
    return out_dir


def install_and_register(stage: int, cfg: dict, out_dir: Path, *,
                         install: bool, register: bool, alias: str) -> None:
    """Copy checkpoint/metrics vào `notebooks/kaggle/` rồi đăng ký MLflow Registry."""
    metrics_files = list(out_dir.rglob(cfg["metrics"]))
    ckpt_files = list(out_dir.rglob(cfg["checkpoint"]))
    last_files = list(out_dir.rglob(cfg["last_checkpoint"]))
    if not metrics_files and not ckpt_files:
        print("\n[WARN] Không thấy checkpoint/metrics trong output — bỏ qua bước cài đặt.")
        return

    if install:
        for f in ckpt_files + last_files + metrics_files:
            dest = ROOT / "notebooks/kaggle" / f.name
            shutil.copy2(f, dest)
            print(f"  đã copy {f.name} -> notebooks/kaggle/")

    if not metrics_files:
        print("\n[WARN] Thiếu metrics json → không đăng ký MLflow (cần metrics để so sánh version).")
        return
    if not ckpt_files:
        print("\n[WARN] Thiếu checkpoint .pth → không đăng ký MLflow.")
        return

    metrics = json.loads(metrics_files[0].read_text(encoding="utf-8"))
    primary = cfg["primary"]
    if primary.startswith("test_") and isinstance(metrics.get("test"), dict):
        value = metrics["test"].get(primary[len("test_"):])
    else:
        value = metrics.get(primary)
    print(f"\nMetrics từ Kaggle: {primary} = {value}")

    if register:
        from mlops import registry

        fn = getattr(registry, cfg["register"])
        result = fn(ckpt=ckpt_files[0], metrics_json=metrics_files[0], alias=alias)
        print(f"Đã đăng ký {result['model_name']} version {result['version']} "
              f"(alias {result['alias']}) — run {result['run_id']}")
    else:
        print("Bỏ qua đăng ký registry (thêm --register để bật).")


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", type=int, choices=(1, 2), default=1)
    ap.add_argument("--check", action="store_true", help="chỉ kiểm tra credential/cấu hình")
    ap.add_argument("--dry-run", action="store_true", help="chỉ build kernel folder, không push")
    ap.add_argument("--username", default=None, help="username Kaggle (mặc định tự dò)")
    ap.add_argument("--dataset-slug", default=DEFAULT_DATASET_SLUG,
                    help="dataset chứa CSV + RLE mask (dùng chung 2 stage)")
    ap.add_argument("--dataset-dir", type=Path, default=ROOT / "data/kaggle/stage2-nih-dev")
    ap.add_argument("--upload-dataset", action="store_true",
                    help="tạo/ cập nhật version dataset trước khi train")
    ap.add_argument("--dataset-notes", default="CheXSpark: cap nhat CSV + RLE mask")
    ap.add_argument("--masks-dataset", default=None,
                    help="dataset mask precompute (mặc định chexspark-masks-224) — "
                         "bỏ decode RLE mỗi epoch")
    ap.add_argument("--masks-dir", type=Path,
                    default=ROOT / "data/kaggle/stage2-masks-224")
    ap.add_argument("--upload-masks", action="store_true",
                    help="tạo/ cập nhật version dataset mask precompute trước khi train")
    ap.add_argument("--resume-from", default=None,
                    help="kernel chạy trước (vd pnt2005/chexspark-stage2-densenet121) — "
                         "mount output làm kernel_sources và resume từ last_*.pth")
    ap.add_argument("--image-dataset", default=IMAGE_DATASET,
                    help="dataset ảnh 224×224 (mặc định mirror NIH của khanfashee)")
    ap.add_argument("--kernel-slug", default=None, help="đổi slug kernel (mặc định theo stage)")
    ap.add_argument("--epochs", default=None)
    ap.add_argument("--base", default=None, help="số kênh tầng đầu U-Net (stage 1)")
    ap.add_argument("--batch-size", default=None)
    ap.add_argument("--lr", default=None)
    ap.add_argument("--mask-mode", default=None, help="stage 2: 'lungs' | 'lungs_heart' | 'none'")
    ap.add_argument("--mask-input", choices=("3ch", "masked", "none"), default=None,
                    help="stage 2: cách đưa mask vào ảnh. '3ch' = [ảnh, ảnh*phổi, "
                         "ảnh*(phổi|tim)] — giữ ngữ cảnh tim/trung thất (mặc định)")
    ap.add_argument("--no-aug", dest="aug", action="store_false", default=None,
                    help="tắt augmentation (RRC + hflip + rotate) để so sánh")
    ap.add_argument("--no-tta", dest="tta", action="store_false", default=None,
                    help="tắt TTA hflip lúc eval")
    ap.add_argument("--no-calibrate", dest="calibrate", action="store_false", default=None,
                    help="tắt temperature scaling + threshold theo nhãn")
    ap.add_argument("--workers", default=None, help="NUM_WORKERS của DataLoader")
    ap.add_argument("--img-size", default=None)
    ap.add_argument("--no-gpu", dest="gpu", action="store_false", help="chạy CPU (debug)")
    ap.add_argument("--accelerator", default=None,
                    help="vd NvidiaTeslaT4 (mặc định: để Kaggle tự chọn theo enable_gpu)")
    ap.add_argument("--run-timeout", type=int, default=7200, help="giới hạn thời gian chạy kernel (giây)")
    ap.add_argument("--poll", type=int, default=60, help="chu kỳ poll status (giây)")
    ap.add_argument("--wait-timeout", type=int, default=14400, help="thời gian chờ tối đa (giây)")
    ap.add_argument("--no-wait", action="store_true", help="push xong thoát, không chờ")
    ap.add_argument("--wait-only", action="store_true",
                    help="không upload/push — chỉ chờ kernel ĐÃ chạy trên Kaggle rồi "
                         "tải output + đăng ký (dùng lại run bị gián đoạn lúc chờ)")
    ap.add_argument("--output-dir", type=Path, default=None)
    ap.add_argument("--install", action="store_true",
                    help="copy checkpoint/metrics tải về vào notebooks/kaggle/")
    ap.add_argument("--register", action="store_true", help="đăng ký MLflow Registry sau khi tải")
    ap.add_argument("--alias", default="production")
    args = ap.parse_args()

    cfg = STAGES[args.stage]
    kaggle = kaggle_bin()

    if args.check:
        print(f"Kaggle CLI: {kaggle}")
        proc = run([kaggle, "datasets", "list", "-m", "--page-size", "1"], check=False)
        if proc.returncode != 0:
            raise SystemExit(
                "\nChưa xác thực Kaggle. Chọn 1 cách:\n"
                "  kaggle auth login                (OAuth, khuyến nghị)\n"
                "  hoặc đặt ~/.kaggle/kaggle.json   (Account → Create New API Token)\n"
                "  hoặc set KAGGLE_API_TOKEN        (token ở trang Settings → API)"
            )
        username = args.username or detect_username(kaggle)
        print(f"\nXác thực OK | username: {username or '(không dò được — truyền --username)'}")
        print(f"Dataset ảnh : {args.image_dataset}")
        print(f"Dataset meta: {args.dataset_slug} (local: {args.dataset_dir})")
        print(f"Dataset mask: {args.masks_dataset or DEFAULT_MASKS_SLUG} "
              f"(local: {args.masks_dir})")
        print(f"Notebook    : {cfg['notebook'].relative_to(ROOT)}")
        return

    username = args.username or detect_username(kaggle)
    if not username:
        raise SystemExit("Không dò được username Kaggle — truyền --username <tên>.")

    # --resume-from: nhận 'user/kernel' hoặc chỉ 'kernel'
    resume_ref = None
    if args.resume_from:
        resume_ref = args.resume_from.strip("/")
        if "/" not in resume_ref:
            resume_ref = f"{username}/{resume_ref}"

    # Mask precompute chỉ có ý nghĩa với Stage-2 (Stage-1 decode RLE làm GT).
    # Chỉ bật khi người dùng chủ động chọn (dataset phải tồn tại trên Kaggle rồi),
    # tránh push fail vì thiếu dataset ở lần chạy đầu.
    masks_ref = None
    if args.stage == 2 and (args.masks_dataset or args.upload_masks):
        masks_ref = args.masks_dataset or f"{username}/{DEFAULT_MASKS_SLUG}"
    if args.stage == 2 and not masks_ref and args.mask_mode != "none":
        print("[WARN] Không dùng mask precompute -> notebook decode RLE mỗi epoch "
              "(~8-12 ms/ảnh). Thêm --upload-masks (lần đầu) hoặc --masks-dataset.")

    overrides = {
        "epochs": args.epochs, "base": args.base, "batch_size": args.batch_size,
        "lr": args.lr, "mask_mode": args.mask_mode, "img_size": args.img_size,
        "mask_input": args.mask_input, "num_workers": args.workers,
        "aug": args.aug, "tta": args.tta, "calibrate": args.calibrate,
        "image_dataset": args.image_dataset, "kernel_slug": args.kernel_slug,
        "gpu": args.gpu, "resume_from": resume_ref, "masks_dataset": masks_ref,
    }
    build_dir = build_kernel_dir(args.stage, cfg, username, args.dataset_slug, overrides)
    kernel_id = f"{username}/{(args.kernel_slug or cfg['kernel_slug'])}"

    if args.dry_run:
        print("\n[dry-run] Lệnh sẽ chạy:")
        if args.upload_dataset:
            print(f"  {kaggle} datasets create|version -p {args.dataset_dir} --dir-mode skip")
        if args.upload_masks:
            print(f"  {kaggle} datasets create|version -p {args.masks_dir} --dir-mode skip")
        print(f"  {kaggle} kernels push -p {build_dir} --timeout {args.run_timeout}")
        print(f"  {kaggle} kernels output {kernel_id} -p <output-dir>")
        return

    if args.wait_only:
        # Không upload/push lại: kernel đang chạy trên Kaggle rồi (lần push trước
        # bị gián đoạn ở khâu chờ) -> chỉ poll rồi tải output + đăng ký như bình thường.
        print(f"[wait-only] bỏ qua upload/push, chờ kernel {kernel_id}")
        status, kernel_id = wait_for_kernel(kaggle, kernel_id, poll=args.poll,
                                            wait_timeout=args.wait_timeout)
    else:
        if args.upload_dataset:
            upload_dataset(kaggle, username, args.dataset_slug, args.dataset_dir,
                           args.dataset_notes)
        if args.upload_masks:
            # Đẩy lên slug do --masks-dataset chỉ định (vd ...-112k), không phải
            # mặc định chexspark-masks-224 — tránh vô tình ghi đè dataset dev.
            masks_slug = args.masks_dataset or f"{username}/{DEFAULT_MASKS_SLUG}"
            upload_dataset(kaggle, username, masks_slug, args.masks_dir,
                           "CheXSpark: mask precompute 224 (packed .npy + heart)")

        status, kernel_id = push_and_wait(kaggle, kernel_id, build_dir,
                                         accelerator=args.accelerator, run_timeout=args.run_timeout,
                                         poll=args.poll, wait_timeout=args.wait_timeout,
                                         no_wait=args.no_wait)

    out_dir = args.output_dir or (ROOT / "data/kaggle/output" / f"stage{args.stage}")
    if status == "queued":
        # --no-wait: đã push và chủ động không chờ -> thoát mã 0 để task
        # orchestration coi là xong; khâu chờ/tải/đăng ký do task khác (--wait-only).
        return
    if status != "complete":
        # Kaggle vẫn lưu output của lần chạy lỗi/timeout -> tải về để resume
        print(f"\nKernel kết thúc với trạng thái '{status}'. Thử tải output giữa chừng ...")
        try:
            fetch_output(kaggle, kernel_id, out_dir)
        except SystemExit:
            pass
        print(f"\nChạy tiếp (giữ nguyên --epochs):\n"
              f"  python scripts/kaggle/train_on_kaggle.py --stage {args.stage} "
              f"--resume-from {kernel_id} --epochs {args.epochs or cfg['patch']['EPOCHS']} ...")
        raise SystemExit(f"Xem log: {kaggle} kernels logs {kernel_id}")

    fetch_output(kaggle, kernel_id, out_dir)
    install_and_register(args.stage, cfg, out_dir,
                         install=args.install or args.register,
                         register=args.register, alias=args.alias)


if __name__ == "__main__":
    main()
