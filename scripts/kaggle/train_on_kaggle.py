"""Train Stage-1/Stage-2 tự động trên Kaggle: upload dataset → push notebook → chờ → tải output.

Máy local không có GPU, nên train thật diễn ra trên Kaggle (T4 ×2). Script này gói
toàn bộ vòng đó thành 1 lệnh, thay vì thao tác tay trên web:

    1. (tuỳ chọn) cập nhật dataset `chexspark-stage2-nih-dev` (CSV + RLE mask)
    2. Sinh `kernel-metadata.json` + notebook đã patch config (epochs/base/lr…)
    3. `kaggle kernels push` → Kaggle chạy notebook trên GPU
    4. Poll `kaggle kernels status` tới khi complete/error (in log nếu lỗi)
    5. `kaggle kernels output` → tải checkpoint + metrics json
    6. (tuỳ chọn) copy vào `notebooks/kaggle/` và đăng ký MLflow Registry

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

    # Train Stage-2 với 30 epoch, rồi tự promote lên production
    python scripts/kaggle/train_on_kaggle.py --stage 2 --epochs 30 --register

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

# Dataset ảnh 224×224 của Kaggle (dùng chung cho cả 2 stage)
IMAGE_DATASET = "khanfashee/nih-chest-x-ray-14-224x224-resized"
DEFAULT_DATASET_SLUG = "chexspark-stage2-nih-dev"

STAGES = {
    1: {
        "notebook": ROOT / "notebooks/kaggle/stage1_unet_segmentation.ipynb",
        "kernel_slug": "chexspark-stage1-unet",
        "title": "CheXSpark Stage1 U-Net lung segmentation",
        "metrics": "stage1_metrics.json",
        "checkpoint": "best_stage1_unet.pth",
        "patch": {"BASE": "32", "EPOCHS": "30", "BATCH_SIZE": "16", "LR": "1e-3"},
        "register": "register_stage1_model",
        "primary": "test_dice_mean",
    },
    2: {
        "notebook": ROOT / "notebooks/kaggle/stage2_densenet121.ipynb",
        "kernel_slug": "chexspark-stage2-densenet121",
        "title": "CheXSpark Stage2 DenseNet121 multi-label",
        "metrics": "stage2_metrics.json",
        "checkpoint": "best_stage2_densenet121.pth",
        "patch": {"EPOCHS": "8", "BATCH_SIZE": "32", "LR": "1e-4", "MASK_MODE": "'lungs'"},
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


def build_kernel_dir(stage: int, cfg: dict, username: str, dataset_slug: str,
                     overrides: dict[str, str]) -> Path:
    """Tạo `data/kaggle/kernels/stageN/` gồm notebook (đã patch) + kernel-metadata.json."""
    slug = cfg["kernel_slug"] if not overrides.get("kernel_slug") else overrides["kernel_slug"]
    img_ref = overrides.get("image_dataset") or IMAGE_DATASET
    meta_ref = f"{username}/{dataset_slug.split('/')[-1]}"

    kernel_id = f"{username}/{slug}"
    build_dir = ROOT / "data/kaggle/kernels" / f"stage{stage}"
    nb_out = build_dir / cfg["notebook"].name

    values = dict(cfg["patch"])
    values["IMG_ROOT"] = f"'/kaggle/input/{img_ref.split('/')[-1]}'"
    values["META_ROOT"] = f"'/kaggle/input/{meta_ref.split('/')[-1]}'"
    for key in ("epochs", "base", "batch_size", "lr", "mask_mode", "img_size"):
        if overrides.get(key):
            value = str(overrides[key])
            if key == "mask_mode" and not value.startswith(("'", '"')):
                value = f"'{value}'"   # MASK_MODE là chuỗi
            values[key.upper()] = value
    changed = patch_notebook(cfg["notebook"], values, nb_out)

    metadata = {
        "id": kernel_id,
        "title": cfg["title"],
        "code_file": nb_out.name,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": bool(overrides.get("gpu", True)),
        "enable_tpu": False,
        "enable_internet": True,
        "dataset_sources": [img_ref, meta_ref],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }
    (build_dir / "kernel-metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nKernel folder: {build_dir}")
    print(f"  id           : {kernel_id}")
    print(f"  notebook     : {nb_out.name}")
    print(f"  datasets     : {', '.join(metadata['dataset_sources'])}")
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
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    meta["id"] = f"{username}/{dataset_slug.split('/')[-1]}"
    meta.setdefault("title", "CheXSpark NIH dev (labels + CheXmask RLE)")
    meta.setdefault("licenses", [{"name": "CC0-1.0"}])
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
def push_and_wait(kaggle: str, kernel_id: str, build_dir: Path, *,
                  accelerator: str | None, run_timeout: int, poll: int,
                  wait_timeout: int, no_wait: bool) -> str:
    """Push notebook và poll tới khi xong. Trả status cuối ('complete'|'error'|'queued')."""
    cmd = [kaggle, "kernels", "push", "-p", str(build_dir), "--timeout", str(run_timeout)]
    if accelerator:
        cmd += ["--accelerator", accelerator]
    run(cmd)

    if no_wait:
        print(f"\nĐã push (không chờ). Theo dõi: kaggle kernels status {kernel_id}")
        return "queued"

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
            return "error"
        if "complete" in out:
            print("\nKernel chạy xong.")
            return "complete"
        time.sleep(poll)

    print("[WARN] Hết thời gian chờ — kernel có thể vẫn đang chạy.")
    return "timeout"


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
    if not metrics_files and not ckpt_files:
        print("\n[WARN] Không thấy checkpoint/metrics trong output — bỏ qua bước cài đặt.")
        return

    if install:
        for f in ckpt_files + metrics_files:
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
    ap.add_argument("--image-dataset", default=IMAGE_DATASET,
                    help="dataset ảnh 224×224 (mặc định mirror NIH của khanfashee)")
    ap.add_argument("--kernel-slug", default=None, help="đổi slug kernel (mặc định theo stage)")
    ap.add_argument("--epochs", default=None)
    ap.add_argument("--base", default=None, help="số kênh tầng đầu U-Net (stage 1)")
    ap.add_argument("--batch-size", default=None)
    ap.add_argument("--lr", default=None)
    ap.add_argument("--mask-mode", default=None, help="stage 2: 'lungs' | 'lungs_heart' | 'none'")
    ap.add_argument("--img-size", default=None)
    ap.add_argument("--no-gpu", dest="gpu", action="store_false", help="chạy CPU (debug)")
    ap.add_argument("--accelerator", default=None,
                    help="vd NvidiaTeslaT4 (mặc định: để Kaggle tự chọn theo enable_gpu)")
    ap.add_argument("--run-timeout", type=int, default=7200, help="giới hạn thời gian chạy kernel (giây)")
    ap.add_argument("--poll", type=int, default=60, help="chu kỳ poll status (giây)")
    ap.add_argument("--wait-timeout", type=int, default=14400, help="thời gian chờ tối đa (giây)")
    ap.add_argument("--no-wait", action="store_true", help="push xong thoát, không chờ")
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
        print(f"Notebook    : {cfg['notebook'].relative_to(ROOT)}")
        return

    username = args.username or detect_username(kaggle)
    if not username:
        raise SystemExit("Không dò được username Kaggle — truyền --username <tên>.")

    overrides = {
        "epochs": args.epochs, "base": args.base, "batch_size": args.batch_size,
        "lr": args.lr, "mask_mode": args.mask_mode, "img_size": args.img_size,
        "image_dataset": args.image_dataset, "kernel_slug": args.kernel_slug,
        "gpu": args.gpu,
    }
    build_dir = build_kernel_dir(args.stage, cfg, username, args.dataset_slug, overrides)
    kernel_id = f"{username}/{(args.kernel_slug or cfg['kernel_slug'])}"

    if args.dry_run:
        print("\n[dry-run] Lệnh sẽ chạy:")
        if args.upload_dataset:
            print(f"  {kaggle} datasets create|version -p {args.dataset_dir} --dir-mode skip")
        print(f"  {kaggle} kernels push -p {build_dir} --timeout {args.run_timeout}")
        print(f"  {kaggle} kernels output {kernel_id} -p <output-dir>")
        return

    if args.upload_dataset:
        upload_dataset(kaggle, username, args.dataset_slug, args.dataset_dir,
                       args.dataset_notes)

    status = push_and_wait(kaggle, kernel_id, build_dir,
                          accelerator=args.accelerator, run_timeout=args.run_timeout,
                          poll=args.poll, wait_timeout=args.wait_timeout,
                          no_wait=args.no_wait)
    if status != "complete":
        raise SystemExit(f"Kernel kết thúc với trạng thái '{status}' — xem log: "
                         f"{kaggle} kernels logs {kernel_id}")

    out_dir = args.output_dir or (ROOT / "data/kaggle/output" / f"stage{args.stage}")
    fetch_output(kaggle, kernel_id, out_dir)
    install_and_register(args.stage, cfg, out_dir,
                         install=args.install or args.register,
                         register=args.register, alias=args.alias)


if __name__ == "__main__":
    main()
