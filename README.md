# CheXSpark

Xây dựng hệ thống phát hiện và trực quan hóa đa bệnh lý lồng ngực từ ảnh X-quang ngực (CXR) bằng Deep Learning và Apache Spark.

**Pipeline 3 giai đoạn**: Lung Segmentation (U-Net) → Multi-label Classification (DenseNet121) → Explainability (Grad-CAM).

## Kiến trúc hạ tầng (docker-compose)

| Service | Image | Port(s) | Vai trò |
|---|---|---|---|
| **minio** | bitnamilegacy/minio | 9000 (S3), 9001 (console) | Object storage: ảnh CXR, mask CheXmask, Silver/Gold datasets |
| **minio-init** | bitnamilegacy/minio-client | — | Tạo bucket `mlflow` tự động khi khởi động |
| **postgres** | postgres:16-alpine | 5432 | Metadata catalog + backend store cho MLflow |
| **spark-master** | **chexspark-spark:3.5.1** (build local) | 7077, 8080 | Spark standalone master (S3A jars + mlflow) |
| **spark-worker-1** | **chexspark-spark:3.5.1** | 8081 | Worker (cores/memory cấu hình qua `.env`) |
| **spark-worker-2** | **chexspark-spark:3.5.1** | 8082 | Worker |
| **mlflow** | ghcr.io/mlflow/mlflow | 5000 | Experiment tracking + model registry |
| **airflow-init** | **chexspark-airflow:2.9.3** (build local) | — | `airflow db migrate` + tạo user admin (one-shot) |
| **airflow-webserver** | **chexspark-airflow:2.9.3** | 8085 | Airflow UI |
| **airflow-scheduler** | **chexspark-airflow:2.9.3** | — | Chạy task (LocalExecutor) + Spark driver |

> **Lưu ý image**: `apache/spark` mặc định **không kèm S3A jars** (không đọc được MinIO) và **không có mlflow**.
> Vì vậy ta build image riêng `chexspark-spark:3.5.1` từ `infra/spark/Dockerfile`, thêm
> `hadoop-aws` + `aws-java-sdk-bundle` (đọc/ghi MinIO) và cài `mlflow==2.9.2` (khớp server).
> Image MinIO dùng bản `bitnamilegacy` do `minio/minio` trên Docker Hub đã bị restrict.

## Cấu trúc repo

```
CheXSpark/
├── docker-compose.yml            # MinIO + PostgreSQL + Spark + MLflow + Airflow
├── .env.example                  # Biến môi trường (copy → .env trước khi chạy)
├── dags/                         # Airflow DAGs (Bronze→Silver→Gold)
├── .github/workflows/            # kaggle-train.yml — bấm nút để train trên Kaggle GPU
├── infra/
│   ├── postgres/init/            # SQL bootstrap (mlflow, chexspark_metadata, airflow)
│   ├── airflow/Dockerfile        # Build image chexspark-airflow (PySpark + Spark provider + mlflow)
│   ├── mlflow/Dockerfile         # Image MLflow + psycopg2 (Postgres) + boto3 (MinIO)
│   └── spark/
│       ├── Dockerfile            # Build image chexspark-spark (S3A jars + mlflow)
│       ├── jars/                 # hadoop-aws + aws-java-sdk-bundle (cho S3A/MinIO)
│       └── conf/                 # spark-defaults.conf (S3/MinIO, MLflow URI)
├── src/
│   ├── common/spark_session.py   # SparkSession factory (dùng chung mọi job)
│   ├── ingestion/                # Spark Core: Bronze → Silver (ingest, mask join, verify)
│   ├── silver/                   # precompute_masks.py — decode RLE 1 lần → mask 224 (bits+PNG)
│   ├── gold/                     # co-occurrence, disease network, MLlib PCA/K-Means/LR/RF
│   ├── dl/                       # U-Net (Stage-1), DenseNet121 + Grad-CAM (Stage-2/3), preprocess
│   ├── mlops/                    # MLflow Model Registry (register/promote/load theo alias)
│   └── serving/                  # FastAPI inference + Streamlit UI
├── notebooks/
│   └── kaggle/                   # stage1_unet_segmentation.ipynb (U-Net) + stage2_densenet121.ipynb
├── tests/                        # Unit & integration tests
└── scripts/
    ├── dev/                      # extract_subset, export Silver→Kaggle, export mask pack, train/eval Stage-1, features, Grad-CAM, registry
    └── kaggle/                   # train_on_kaggle.py — push dataset+notebook, chờ train, resume, tải output
```

## Cách chạy

```powershell
# 1. Tạo file .env từ template
Copy-Item .env.example .env

# 2. Khởi động toàn bộ hạ tầng
docker compose up -d

# 3. Kiểm tra trạng thái
docker compose ps
```

### Truy cập các UI

- **MinIO Console**: http://localhost:9001 (user/pass từ `.env`)
- **Spark Master UI**: http://localhost:8080
- **Spark Worker UIs**: http://localhost:8081, http://localhost:8082
- **MLflow UI**: http://localhost:5000
- **Airflow UI**: http://localhost:8085 (`admin` / `admin` — chỉ dùng cho dev)

### Gửi một Spark job thử

```powershell
docker compose exec spark-master /opt/spark/bin/spark-submit `
  --master spark://spark-master:7077 `
  --deploy-mode client `
  /opt/spark/user-src/ingestion/ingest_bronze_to_silver.py
```

### Dọn dẹp

```powershell
docker compose down          # dừng services, giữ data volumes
docker compose down -v       # xóa hoàn toàn (kể cả MinIO/Postgres data)
```

---

## Giai đoạn DEV — chạy end-to-end trên 2k ảnh

Mục tiêu: kiểm chứng trọn pipeline **Bronze → Silver → (sẵn sàng cho DL)** trên quy mô nhỏ,
đảm bảo Spark cluster đọc/ghi MinIO (S3A) và log MLflow hoạt động trước khi mở rộng.

### 1. Tải dữ liệu nguồn (NIH ChestX-ray14)

Dataset đầy đủ ~42GB (112k ảnh). Giai đoạn DEV chỉ cần **1 part zip (~1.9GB, 10k ảnh)** +
metadata CSV (~9MB). Dùng mirror HuggingFace `alkzar90/NIH-Chest-X-ray-dataset` (direct download, resumable):

```powershell
# Metadata CSV (112k rows, nhãn 14 bệnh + Patient ID)
curl.exe -L -o data/raw/metadata/Data_Entry_2017_v2020.csv `
  "https://huggingface.co/datasets/alkzar90/NIH-Chest-X-ray-dataset/resolve/main/data/Data_Entry_2017_v2020.csv"

# 1 part ảnh (images_001.zip ~1.9GB, ~10k ảnh PNG 1024x1024)
curl.exe -L --continue-at - -o data/raw/images/images_001.zip `
  "https://huggingface.co/datasets/alkzar90/NIH-Chest-X-ray-dataset/resolve/main/data/images/images_001.zip"
```

### 2. Trích 2k ảnh + dựng Bronze trong MinIO

```powershell
# Trích 2k ảnh PNG thật (lọc file AppleDouble '._' trong zip)
python scripts/dev/extract_subset.py --zip data/raw/images/images_001.zip --out data/bronze/images --limit 2000

# Tạo bucket + upload ảnh, metadata, manifest vào MinIO (xem scripts/dev/ + mc)
# (chi tiết các lệnh mc cp nằm trong section phát triển)
```

Layer **Bronze** sau bước này:
- `bronze/images/` — 2000 PNG ảnh X-quang
- `bronze/metadata/Data_Entry_2017_v2020.csv` — metadata 112k dòng
- `bronze/images_manifest/images_manifest.csv` — danh sách 2k ảnh đã materialize

### 3. Chạy ingestion Bronze → Silver (Spark + MLflow)

```powershell
# Job Spark: đọc 112k metadata từ MinIO, join manifest 2k, patient-level split,
# ghi Silver Parquet partition theo split, log thống kê lên MLflow
docker compose exec -T spark-master /opt/spark/bin/spark-submit `
  --master spark://spark-master:7077 --deploy-mode client `
  /opt/spark/user-src/ingestion/ingest_bronze_to_silver.py

# Verify Silver: đọc lại Parquet, in schema + đếm theo split
docker compose exec -T spark-master /opt/spark/bin/spark-submit `
  --master spark://spark-master:7077 --deploy-mode client `
  /opt/spark/user-src/ingestion/verify_silver.py
```

**Kết quả đã ghi nhận** (thống kê trên full 112k metadata — khớp số liệu NIH công bố):
- `full_total_images: 112,120`, `full_total_patients: 30,805`, comorbidity (≥2 bệnh): 20,796
- Per-label: Atelectasis 11559, Infiltration 19894, Effusion 13317, Cardiomegaly 2776, Hernia 227…
- Dev subset: 2000 ảnh / 648 bệnh nhân → train 1335 · val 321 · test 344 (patient-level, không rò rỉ)
- Silver: `s3a://silver/cxr14/split={train,val,test}/*.parquet` (schema: 14 cột `label_*` binary + `image_uri` + `split`)
- MLflow: experiment `chexspark-silver-ingestion`, 23 metrics/run → http://localhost:5000

### 4. Mask ETL — CheXmask join vào Silver (Spark)

Mask phổi CheXmask (RLE, file ~2GB / 112k rows) được đưa vào **Bronze** rồi join
với Silver bằng **Spark** — đúng tinh thần AGENTS.md: *"Silver: map chuẩn hóa
Image ID giữa ảnh và mask"*.

```powershell
# 4a. Tải CheXmask (mirror HuggingFace — PhysioNet throttle ~5.6 KB/s, HF ~9 MB/s)
curl.exe -L --continue-at - -o data/raw/chexmask/ChestX-Ray8.csv `
  "https://huggingface.co/datasets/ming0100/chexmask/resolve/main/OriginalResolution/ChestX-Ray8.csv"

# 4b. Upload vào Bronze (2GB, ~20s qua mạng local Docker)
docker run --rm --network chexspark_chexspark-net `
  -v "D:\BigData\CheXSpark\data\raw\chexmask:/data:ro" --entrypoint sh `
  bitnamilegacy/minio-client:latest -c `
  "mc alias set local http://minio:9000 $env:MINIO_ROOT_USER $env:MINIO_ROOT_PASSWORD; mc cp /data/ChestX-Ray8.csv local/bronze/chexmask/ChestX-Ray8.csv"

# 4c. Job Spark join mask (đọc 112k rows phân tán → filter broadcast → 2k → ghi Silver mới)
docker compose exec -T spark-master /opt/spark/bin/spark-submit `
  --master spark://spark-master:7077 --deploy-mode client `
  /opt/spark/user-src/ingestion/ingest_chexmask_to_silver.py

# 4d. Verify Silver masked (path truyền qua argv[1])
docker compose exec -T spark-master /opt/spark/bin/spark-submit `
  --master spark://spark-master:7077 --deploy-mode client `
  /opt/spark/user-src/ingestion/verify_silver.py s3a://silver/cxr14_masked
```

**Kết quả mask join:**
- Silver mới `s3a://silver/cxr14_masked/` — schema **33 cột** = 25 cũ + 7 cột mask
  (`dice_rca_mean/max`, `mask_left/right/heart_rle`, `mask_height/width`) + `chexmask_ingested_at`
- Join **2,000/2,000 (thiếu mask: 0)**, split giữ nguyên 1335/321/344
- Chất lượng: mean Dice RCA **0.841**, 1,979/2,000 (99%) đạt ngưỡng ≥ 0.7
- MLflow: experiment `chexspark-silver-mask-join`
- **Lưu ý kỹ thuật**: RLE strings ~780MB qua shuffle gây OOM cluster 8GB nếu giữ
  112k rows → job filter về 2k rows **ngay lúc đọc** (broadcast join) trước mọi cache/agg;
  submit với `--driver-memory 2g --conf spark.executor.memory=2g`
- RLE CheXmask dạng **cặp `(start, length)`**, reshape row-major (order=`'C'`) —
  đã validate bằng bounding-box so với cột Landmarks

### 5. Export dữ liệu Stage-2 cho Kaggle

```powershell
# Copy Parquet masked từ MinIO về local
docker run --rm --network chexspark_chexspark-net `
  -v "D:\BigData\CheXSpark\data\tmp\silver_masked:/out" --entrypoint sh `
  bitnamilegacy/minio-client:latest -c `
  "mc alias set local http://minio:9000 $env:MINIO_ROOT_USER $env:MINIO_ROOT_PASSWORD; mc cp --recursive local/silver/cxr14_masked /out/"

# Export → stage2_train.csv (labels + RLE mask trong 1 file) + Kaggle metadata
python scripts/dev/export_silver_masked_to_kaggle.py
```

Output `data/kaggle/stage2-nih-dev/` (upload lên Kaggle làm dataset) — gắn kèm
`notebooks/kaggle/stage2_densenet121.ipynb` + dataset ảnh
`khanfashee/nih-chest-x-ray-14-224x224-resized` để train Stage-2.

### 5b. Precompute mask 224×224 (Spark) — bỏ decode RLE mỗi epoch

`cxr14_masked` giữ mask dạng RLE 1024×1024; decode + resize tốn **~8–12 ms/ảnh**.
Với 112k ảnh × 20 epoch ≈ **6–8 giờ CPU** chỉ để dựng lại đúng một mask không đổi
→ job Spark này decode **một lần** rồi lưu 2 dạng:

- `mask_lung_bits` / `mask_heart_bits` — **packbits** (6.272 byte/ảnh; giải nén ~0,1 ms)
- `png_lung` / `png_lungs_heart` — PNG xám 224×224 (~1 KB) để xem bằng mắt / serving

```powershell
# 5b-1. Job Spark (2000 mask dev ~40s; chạy tự động trong DAG chexspark_chexmask_join)
docker compose exec -T spark-master /opt/spark/bin/spark-submit `
  --master spark://spark-master:7077 --deploy-mode client `
  /opt/spark/user-src/silver/precompute_masks.py

# 5b-2. Copy Parquet về local
docker run --rm --network chexspark_chexspark-net `
  -e "MC_HOST_local=http://$env:MINIO_ROOT_USER:$env:MINIO_ROOT_PASSWORD@minio:9000" `
  -v "D:\BigData\CheXSpark\data\tmp:/out" --entrypoint /opt/bitnami/minio-client/bin/mc `
  bitnamilegacy/minio-client:latest cp --recursive local/silver/cxr14_masks /out/silver_masks

# 5b-3. Export dataset Kaggle (masks_{split}.npy + masks_heart_{split}.npy + index)
python scripts/dev/export_masks_to_kaggle.py --silver-dir data/tmp/silver_masks
```

Kết quả dev 2k: **2000/2000 mask**, 0 mask rỗng, diện tích phổi trung bình
**24,5%**, tim **8,9%**, PNG ~1 KB/ảnh; `masks_{split}.npy` 12 MB (112k ảnh ≈ 0,7 GB,
vẫn dưới hạn 20 GB/file của Kaggle). Đã verify **bit-for-bit** với
`dl.preprocess.lung_mask` (8/8 ảnh) — không lệch một pixel.

Notebook Stage-2 tự dùng pack khi có `MASK_PACK_DIR` (dataset `chexspark-masks-224`):
`np.load(..., mmap_mode='r')` + `np.unpackbits` → **~0,1 ms/ảnh**, không tốn RAM,
không decode RLE; vẫn fallback RLE nếu thiếu dataset. `MASK_MODE='lungs_heart'`
OR thêm `masks_heart_{split}.npy`.

### 5c. Scale toàn bộ 112k ảnh (Silver full + mask pack 1,34 GB)

Dev 2k chỉ để chạy đúng luồng — số liệu chốt phải train trên **112.120 ảnh /
30.805 bệnh nhân**. Các bước chuẩn bị dưới đây chạy hoàn toàn bằng Spark, không tốn GPU:

```powershell
# 5c-1. Silver full: bỏ join manifest 2k -> 112.120 rows. Split vẫn hash(patient_id)
#        cố định nên dev 2k và full 112k gán cùng một split cho cùng bệnh nhân.
docker exec chexspark-spark-master /opt/spark/bin/spark-submit `
  --master spark://spark-master:7077 --deploy-mode client `
  /opt/spark/user-src/ingestion/ingest_bronze_to_silver.py --full
#    -> s3a://silver/cxr14_full: train 77.728 / val 17.662 / test 16.730 (~30s)

# 5c-2. Precompute mask 112k: đọc thẳng CheXmask RLE (2 GB) từ Bronze rồi
#        broadcast join split -> không shuffle 2 GB, không repartition (cluster 8 GB)
docker exec chexspark-spark-master /opt/spark/bin/spark-submit `
  --master spark://spark-master:7077 --deploy-mode client `
  --conf spark.executor.memory=1g --conf spark.executor.cores=1 `
  --conf spark.cores.max=4 --conf spark.sql.files.maxPartitionBytes=33554432 `
  /opt/spark/user-src/silver/precompute_masks.py --source bronze `
  --out s3a://silver/cxr14_masks_full
#    -> 112.120/112.120 mask, 0 mask rỗng, phổi 24,4% | tim 8,9% | 36 partitions

# 5c-3. Copy 2 Parquet về local rồi export pack npy + labels CSV (không kèm RLE)
docker run --rm --network chexspark_chexspark-net `
  -e "MC_HOST_local=http://$env:MINIO_ROOT_USER:$env:MINIO_ROOT_PASSWORD@minio:9000" `
  -v "D:\BigData\CheXSpark\data\tmp:/out" --entrypoint /opt/bitnami/minio-client/bin/mc `
  bitnamilegacy/minio-client:latest cp --recursive local/silver/cxr14_full /out/silver_full/
#   (tương tự cho cxr14_masks_full -> data/tmp/silver_masks_full/)

python scripts/dev/export_masks_to_kaggle.py `
  --silver-dir data/tmp/silver_masks_full/cxr14_masks_full `
  --out-dir data/kaggle/stage2-masks-224-112k `
  --dataset-id pnt2005/chexspark-masks-224-112k `
  --title "CheXSpark Masks 224 112k (packed)"

python scripts/dev/export_silver_labels.py `
  --silver-dir data/tmp/silver_full/cxr14_full `
  --out data/kaggle/stage2-nih-112k/stage2_train.csv `
  --drop image_uri silver_ingested_at
```

Kết quả 112k:

| Artefact | Nội dung | Dung lượng |
|---|---|---|
| `stage2-masks-224-112k/` | 6 file `masks{,_heart}_{train,val,test}.npy` (packbits 6.272 B/ảnh) + `mask_index.csv` 112.120 dòng | **1.341 MB** |
| `stage2-nih-112k/stage2_train.csv` | 14 nhãn + split, **không RLE** (mask pack đã cover 112.120/112.120) | **8,9 MB** |

Đã verify: `mask_index.csv` ↔ `stage2_train.csv` join tuyệt đối (0 dòng lệch, 0 split
sai), `mask_row` phủ đúng `0..n-1` mỗi split khớp shape npy, và fg decode từ pack
khớp `n_fg_lung`/`n_fg_heart` trong index.

**Đo scaling** (cùng job, cùng config, `write_sec` của 112k):

| Phép đo | Cấu hình | Thời gian ghi | Throughput |
|---|---|---|---|
| Data scaling | 2.000 ảnh (dev smoke) | 37,5 s | 53 mask/s |
| | **112.120 ảnh** (56× dữ liệu) | **207,5 s** (5,5× thời gian) | **540 mask/s** (10×) |
| Worker scaling | 1 worker × 2 core | 245,4 s | 457 mask/s |
| | 2 worker × 4 core | **207,5 s** | 540 mask/s → **speedup 1,18×** |

Hai phép đo cho thấy: (1) overhead cố định (Spark job setup, broadcast, log MLflow)
chiếm phần lớn ở quy mô nhỏ nên throughput tăng ~10× khi scale dữ liệu 56×;
(2) scale-out **phụ tuyến tính** (2× core chỉ được 1,18×) vì bottleneck là I/O single-node
— CheXmask 2 GB và MinIO đều nằm trên cùng VM 8 GB, cộng thêm stage ghi Parquet
36 partition chạy trên một nút. Muốn speedup tuyến tính thì phải tách MinIO ra nút riêng
hoặc chạy cluster đa node; đây cũng chính là lý do job dùng `maxPartitionBytes=32 MB`
để tách file lớn thành ~66 partition thay vì đọc tuần tự.

### 6. Stage-1 — U-Net phân đoạn phổi (Lung Segmentation)

Giai đoạn 1 của pipeline học sâu: sinh mask phổi từ **ảnh X-quang thô**, thay cho
CheXmask ở thời điểm suy luận (CheXmask chỉ tồn tại cho NIH, không có khi bệnh viện
gửi ảnh mới).

- Kiến trúc: U-Net 4 tầng xuống (`src/dl/segmentation.py`), base=16 → **1.94M tham số**
  (GPU/Kaggle dùng base=32)
- Ground truth: CheXmask (`mask_left_rle | mask_right_rle`, **không** gồm tim)
- Loss: `BCE + Dice` (Dice tính từng ảnh để ảnh phổi to không lấn át ảnh nhỏ)
- Dữ liệu: dùng lại **cùng** `stage2_train.csv` đã export ở mục 5 — không cần export mới

```powershell
# Train trên máy local (CPU) — dev subset
python scripts/dev/train_stage1_unet.py --limit 500 --epochs 12

# Train đầy đủ trên GPU: dùng notebook Kaggle
#   notebooks/kaggle/stage1_unet_segmentation.ipynb  (base=32, 30 epoch)
```

Đánh giá so với mask CheXmask trên tập test (Dice/IoU từng ảnh + 4 ca tệ nhất):

```powershell
python scripts/dev/evaluate_stage1_segmentation.py --split test --viz 4
# -> data/outputs/stage1/stage1_eval_test.csv + stage1_worst_test.png
```

Output: `notebooks/kaggle/best_stage1_unet.pth` + `stage1_metrics.json`.

**Kết quả trên 500 ảnh train / 321 val / 344 test (CPU, 12 epoch, base=16):**

| Chỉ số | Val | Test |
|---|---|---|
| Dice mean | 0.9442 | **0.9461** |
| Dice median | 0.9696 | 0.9639 |
| Dice p10 | 0.8608 | 0.8886 |
| IoU mean | 0.8999 | 0.9015 |
| % ảnh Dice ≥ 0.7 | 98.8% | **99.4%** |
| % ảnh Dice ≥ 0.9 | – | 86.1% |

- Dice tăng 0.857 (epoch 1) → 0.944 (epoch 12); chỉ 500 ảnh đã đủ vì phổi là cấu trúc
  tương phản cao, 1 lớp — khác hẳn Stage-2 (phân loại bệnh, mAUROC 0.627) vốn cần
  nhiều dữ liệu hơn.
- Ca tệ nhất (Dice 0.635) là ảnh mà **CheXmask GT mở rộng quá tay** sang trung thất/
  góc sườn hoành; U-Net cắt sát phổi hơn nên lệch chủ yếu ở **biên**, không ca nào
  mất hẳn một bên phổi → xem `data/outputs/stage1/stage1_worst_test.png`
- Model đã đăng ký MLflow Registry: `chexspark-stage1-unet` version 1, alias
  `production` (verify logits khớp file local, lệch 0.0)

**Ablation — thay CheXmask bằng mask U-Net thì Stage-2 có tụt không?**
(`scripts/dev/compare_mask_sources.py`, chạy DenseNet121 2 lượt trên cùng tập test)

| Chỉ số | CheXmask (GT) | Mask U-Net | Δ |
|---|---|---|---|
| mAUROC 14 bệnh | 0.6279 | 0.6033 | **−0.0247** |
| mean \|Δp\| | – | – | 0.061 |

- Giảm mạnh nhất: **Pleural_Thickening −0.152**, **Mass −0.106**, Emphysema −0.066,
  Cardiomegaly −0.032 (mask mất vùng tim — khớp nhận xét ở Grad-CAM)
- Tăng: Edema +0.078, Nodule +0.047, Pneumothorax +0.035
- Đọc kết quả: mask U-Net "sát phổi" hơn CheXmask (vốn rộng tay ở trung thất) nên
  thay đổi xác suất trung bình 0.061 và mAUROC mất ~0.025 — **chất lượng mask không
  phải nút thắt** của Stage-2; nút thắt là dữ liệu/độ phân giải tổn thương. Muốn cải
  thiện Cardiomegaly thì phải dùng `MASK_MODE='lungs_heart'` chứ không phải mask nét hơn.

**Nối Stage-1 vào Stage-3** (chạy đủ chuỗi U-Net → mask → DenseNet121 → Grad-CAM):

```powershell
python scripts/dev/run_gradcam_batch.py --diseases Effusion --mask-source unet
# hoặc lấy cả 2 model từ registry
python scripts/dev/run_gradcam_batch.py --from-registry --mask-source unet --stage1-from-registry
```

### 7. Gold layer — phân tích dịch tễ & mạng bệnh lý (Spark)

Gold layer chạy trên **full metadata 112,120 ảnh** (không chỉ 2k dev) nên các con
số thống kê/đồ thị dùng trực tiếp cho báo cáo.

```powershell
# 6a. Co-occurrence 14x14 + thống kê dịch tễ (1 pass trên 112k)
docker exec chexspark-spark-master /opt/spark/bin/spark-submit `
  --master spark://spark-master:7077 --deploy-mode client `
  --driver-memory 2g --conf spark.executor.memory=2g `
  /opt/spark/user-src/gold/disease_cooccurrence.py

# 6b. Disease co-occurrence network (GraphX-style: degree, centrality, community)
docker exec chexspark-spark-master /opt/spark/bin/spark-submit `
  --master spark://spark-master:7077 --deploy-mode client `
  --driver-memory 2g --conf spark.executor.memory=2g `
  /opt/spark/user-src/gold/disease_graph.py

# 6c. Verify toàn bộ output Gold
docker exec chexspark-spark-master /opt/spark/bin/spark-submit `
  --master spark://spark-master:7077 --deploy-mode client `
  /opt/spark/user-src/gold/verify_gold.py
```

**Kết quả co-occurrence (112,120 ảnh):**

| Bệnh | Ca | Tỷ lệ |
|---|---|---|
| Infiltration | 19,894 | 17.7% |
| Effusion | 13,317 | 11.9% |
| Atelectasis | 11,559 | 10.3% |
| Nodule | 6,331 | 5.6% |
| Mass | 5,782 | 5.2% |
| Pneumothorax | 5,302 | 4.7% |
| … | | |
| Hernia | 227 | 0.2% |

- **Comorbidity**: 60,361 ảnh không bệnh (53.8%), 30,963 ảnh 1 bệnh, 14,306 ảnh
  2 bệnh, 4,856 ảnh 3 bệnh; tối đa 9 bệnh/ảnh
- **Top cặp đồng mắc**: Effusion+Infiltration (4,000), Atelectasis+Effusion (3,275),
  Atelectasis+Infiltration (3,264)
- **Lưu ý phương pháp**: co-occurrence thô KHÔNG đủ để kết luận liên kết bệnh —
  trên 112k ảnh *mọi* cặp 14 bệnh đều xuất hiện cùng ít nhất 1 lần (91/91 cạnh →
  degree centrality bằng nhau, community sụp thành 1 nhóm). Vì vậy Gold dùng thêm
  **lift** = P(d1,d2)/(P(d1)·P(d2)) và chỉ giữ cạnh có `cooccurrence ≥ 50` và
  `lift ≥ 2.0`.

**Mạng bệnh lý (17 cạnh ý nghĩa, modularity Q = 0.472 — ngưỡng Newman ≥ 0.3):**

| Cộng đồng | Bệnh | Diễn giải |
|---|---|---|
| 1 | Atelectasis, Cardiomegaly, Effusion, Consolidation | Nhóm tràn dịch/tim to/xẹp phổi |
| 2 | Edema, Infiltration, Pneumonia | Nhóm phù phổi/nhiễm trùng |
| 3 | Emphysema, Pneumothorax | Nhóm COPD/tràn khí màng phổi |
| 4 | Fibrosis, Mass, Nodule, Pleural_Thickening | Nhóm tổn thương mạn/tổ chức kẽ |
| 5 | Hernia | Bệnh hiếm, tách biệt |

- **Liên kết mạnh nhất (lift)**: Edema+Pneumonia (11.6), Emphysema+Pneumothorax (6.3),
  Fibrosis+Pleural_Thickening (3.5), Cardiomegaly+Effusion (3.2), Mass+Nodule (2.8)
- **Trung tâm nhất (weighted degree)**: Effusion, Atelectasis, Consolidation
- Ngưỡng lift được chọn tự động: quét `1.0 → 3.0`, lấy ngưỡng **thấp nhất** (đồ thị
  dày nhất, giữ nhiều bệnh) mà modularity ≥ 0.3 → kết quả ghi ở
  `gold/graph/threshold_sweep`

**Lưu ý GraphX**: PySpark không expose API GraphX (chỉ có trong Scala). Đồ thị chỉ
14 node / 91 cạnh nên degree/centrality tính bằng Spark DataFrame, community
detection dùng **greedy modularity (Clauset-Newman-Moore)** chạy trên driver sau khi
collect — tương đương GraphX cho use-case này và tất định (seed cố định).

### 8. Gold layer — MLlib trên feature vectors DenseNet121

```powershell
# 7a. Trích 1024-d features từ checkpoint Stage-2 (preprocessing giống lúc train:
#     grayscale → 224 → nhân lung mask → ImageNet normalize)
python scripts/dev/extract_densenet_features.py

# 7b. Upload Parquet lên Gold
docker run --rm --network chexspark_chexspark-net `
  -v "D:\BigData\CheXSpark\data\tmp:/data:ro" --entrypoint sh `
  bitnamilegacy/minio-client:latest -c `
  "mc alias set local http://minio:9000 $env:MINIO_ROOT_USER $env:MINIO_ROOT_PASSWORD; mc cp /data/densenet_features.parquet local/gold/features/densenet121/features.parquet"

# 7c. PCA + K-Means + baseline LR/RF
docker exec chexspark-spark-master /opt/spark/bin/spark-submit `
  --master spark://spark-master:7077 --deploy-mode client `
  --driver-memory 2g --conf spark.executor.memory=2g `
  /opt/spark/user-src/gold/mllib_analysis.py
```

Output: `gold/mllib/pca_variance`, `pca_points` (scatter 2D + nhãn cụm),
`kmeans_sweep` (silhouette theo k), `kmeans_clusters` (thành phần bệnh mỗi cụm),
`baseline_metrics` (AUC LR vs RF từng bệnh) — MLflow experiment
`chexspark-gold-mllib`.

**Kết quả trên 2,000 ảnh dev (2000 × 1024-d):**

| Phân tích | Kết quả |
|---|---|
| PCA | 2 PC đầu giải thích **18.7%** phương sai, 50 PC = **67.2%** |
| K-Means | silhouette cao nhất chỉ **0.156** (k=2) → không có cụm tách bạch rõ trong không gian 1024-d |
| Cụm 0 (858 ảnh) | giàu Edema ×2.2, Cardiomegaly ×2.1, Hernia ×2.1; 32.6% không bệnh |
| Cụm 1 (1142 ảnh) | 72.9% không bệnh — nhóm "bình thường/nhẹ" |
| **Baseline LR** | **mean AUC 0.631** (cao nhất: Cardiomegaly 0.872, Atelectasis 0.735, Fibrosis 0.697) |
| **Baseline RF** | mean AUC 0.595 (cao nhất: Cardiomegaly 0.762, Atelectasis 0.744) |
| DenseNet121 end-to-end | test mAU 0.627 (Stage-2, 2k ảnh / 8 epochs) |

> **Nhận xét cho báo cáo**: Logistic Regression trên **feature đóng băng** của
> DenseNet121 đạt mean AUC 0.631 — **tương đương** fine-tune end-to-end (0.627).
> Với chỉ 2k ảnh, việc train lại toàn bộ 7M tham số không hiệu quả bằng dùng
> feature trích sẵn; đây là luận cứ để mở rộng dữ liệu (5.6k → 112k) thay vì
> tăng epoch. Silhouette thấp (0.156) cho thấy feature space chưa phân tách bệnh
> lý rõ — nhất quán với mAU còn thấp; cần model/dữ liệu tốt hơn trước khi kỳ vọng
> cụm bệnh lý tách bạch.

### 9. Stage-3 — Grad-CAM (Explainability)

Sinh bản đồ nhiệt từ feature map tích chập cuối của DenseNet121 để khoanh vùng
tổn thương nghi ngờ cho từng bệnh (output thứ 3 của hệ thống, cạnh mask phổi và
danh sách xác suất).

```powershell
# Toàn bộ 14 bệnh, top-4 ca dương tính rõ nhất trong test
python scripts/dev/run_gradcam_batch.py --top-k 4

# Chỉ vài bệnh, hoặc đổi lớp đích (denseblock3 = 14x14 nét hơn, denseblock4 = 7x7)
python scripts/dev/run_gradcam_batch.py --diseases Cardiomegaly,Effusion --target denseblock4

# Nạp model theo alias từ MLflow Model Registry thay vì file .pth
python scripts/dev/run_gradcam_batch.py --from-registry --alias production
```

**Module tái sử dụng** (dùng lại cho FastAPI/Streamlit):
- `src/dl/preprocess.py` — decode RLE mask, load ảnh xám, build tensor (đúng
  pipeline lúc train: 224 → nhân lung mask → ImageNet normalize)
- `src/dl/model.py` — build/load DenseNet121 từ checkpoint Stage-2
- `src/dl/gradcam.py` — `GradCAM` class (forward/backward hook), colormap `jet`
  thuần numpy, `overlay_heatmap` (chỉ giữ CAM trong vùng phổi)
- `src/mlops/registry.py` — đăng ký/promote/nạp model Stage-2 qua MLflow Model Registry

**Output** `data/outputs/gradcam/`:
- `gradcam_<Disease>.png` — contact sheet top-K: `[ảnh gốc | mask phổi | Grad-CAM | overlay]`
- `cam_<image>_<Disease>.png` — ảnh overlay đơn lẻ
- `gradcam_index.csv` — (image_index, disease, prob, label, file)

**Kết quả & nhận xét:**
- Sinh được heatmap cho **13/14 bệnh** (Hernia không có ca dương tính trong test)
- **Effusion** → CAM tập trung ở **đáy phổi**; **Emphysema** → **vùng phổi trên**
  — khớp giải phẫu, cho thấy model học được tín hiệu vị trí dù mAU mới 0.63
- **Cardiomegaly** → CAM không chỉ vào tim mà vào **phổi**: vì `MASK_MODE='lungs'`
  **mask mất vùng tim** khỏi input, model chỉ còn thông tin hình dạng phổi để suy
  ra tim to. Đây là luận cứ để thử `MASK_MODE='lungs_heart'` (đã hỗ trợ sẵn trong
  notebook + `lung_mask(include_heart=True)`)
- Ảnh portable/AP có cột chụp mờ → CAM đôi khi bám vào **biên ảnh/viền collimation**
  thay vì phổi: hạn chế của model yếu (2k ảnh, 8 epoch) chứ không phải lỗi Grad-CAM
- Lớp đích mặc định `denseblock3` (1024×14×14) cho heatmap nét gấp 4 lần so với
  `denseblock4` (1024×7×7); không hook `model.features` vì DenseNet dùng
  `F.relu(features, inplace=True)` làm autograd báo lỗi in-place trên output hook
- Grad-CAM ở đây là **class-specific** trên 14 logit độc lập (multi-label, không
  softmax) — mỗi bệnh có heatmap riêng, không phải CAM của lớp dự đoán cao nhất

## Train trên Kaggle (GPU) — tự động hoá

Máy local không có GPU nên train thật chạy trên **Kaggle T4 ×2**. Thay vì thao tác
tay trên web, `scripts/kaggle/train_on_kaggle.py` gói cả vòng đời thành **1 lệnh**:

```powershell
# 0. Cài + xác thực (một lần). GPU cần xác thực số điện thoại trong Kaggle Settings.
D:\BigData\venv-mlops\Scripts\python -m pip install kaggle
# Cách nhanh: đặt trong .env ở repo root (script tự nạp, không cần export tay):
#   KAGGLE_USERNAME=pnt2005
#   KAGGLE_KEY=KGAT_xxxxxxxx          (Kaggle → Settings → API → Generate New Token)
# Script tự map KAGGLE_KEY -> KAGGLE_API_TOKEN cho kaggle CLI >= 2.0.
# Hoặc cách OAuth: D:\BigData\venv-mlops\Scripts\kaggle auth login

# 1. Kiểm tra credential + cấu hình
D:\BigData\venv-mlops\Scripts\python scripts/kaggle/train_on_kaggle.py --stage 1 --check

# 2. Xem trước những gì sẽ chạy (không gọi Kaggle)
D:\BigData\venv-mlops\Scripts\python scripts/kaggle/train_on_kaggle.py --stage 1 --dry-run

# 3. Train + tải kết quả + đăng ký MLflow (Docker phải đang chạy)
D:\BigData\venv-mlops\Scripts\python scripts/kaggle/train_on_kaggle.py --stage 1 --upload-dataset --register

# 4. Stage-2 full: đẩy luôn dataset mask precompute (bỏ decode RLE) rồi train
D:\BigData\venv-mlops\Scripts\python scripts/kaggle/train_on_kaggle.py --stage 2 `
  --upload-masks --epochs 30 --mask-mode lungs_heart --register

# 5. Kaggle cắt session 12h → chạy tiếp từ checkpoint của lần trước
D:\BigData\venv-mlops\Scripts\python scripts/kaggle/train_on_kaggle.py --stage 2 `
  --epochs 30 --resume-from pnt2005/chexspark-stage2-densenet121 --register

# 6. Train FULL 112k (model promote production): upload 2 dataset + train 30 epoch.
#    Chuẩn bị dữ liệu đã xong ở mục 5c — ở đây chỉ đẩy lên Kaggle rồi train.
D:\BigData\venv-mlops\Scripts\python scripts/kaggle/train_on_kaggle.py --stage 2 `
  --dataset-slug chexspark-stage2-nih-112k --dataset-dir data\kaggle\stage2-nih-112k `
  --upload-dataset `
  --masks-dataset pnt2005/chexspark-masks-224-112k --masks-dir data\kaggle\stage2-masks-224-112k `
  --upload-masks `
  --epochs 30 --register --install `
  --run-timeout 43200 --wait-timeout 46800 `
  --output-dir data\kaggle\output\stage2-112k
```

Lệnh 6 dùng `--dataset-slug` / `--masks-dataset` để **tách dataset 112k khỏi bản dev 2k**
(`chexspark-stage2-nih-dev` / `chexspark-masks-224`) — run dev và run 112k không đụng
vào nhau, và chỉ run 112k mới `--register` lên alias `production`.

Script làm 6 việc:

1. Cập nhật dataset `chexspark-stage2-nih-dev` (nếu `--upload-dataset`) — dùng chung cho cả 2 stage
2. Cập nhật dataset `chexspark-masks-224` (nếu `--upload-masks`) — mask precompute, xem mục 5b
3. Sinh `kernel-metadata.json` + **patch config trong notebook** (`IMG_ROOT`, `META_ROOT`,
   `MASK_PACK_DIR`, `RESUME_FROM`, `EPOCHS`, `BASE`, `BATCH_SIZE`, `LR`, `NUM_WORKERS`,
   `MASK_INPUT`, `MASK_MODE`, `AUG`, `TTA`, `CALIBRATE`) vào `data/kaggle/kernels/stageN/`
4. `kaggle kernels push` → Kaggle chạy notebook trên GPU (`enable_gpu: true`)
5. Poll `kaggle kernels status` tới khi xong; nếu lỗi **vẫn tải output** (checkpoint
   giữa chừng) và in lệnh `--resume-from` để chạy tiếp
6. `kaggle kernels output` → tải checkpoint + metrics (`best_*`, `last_*`), `--install`
   copy vào `notebooks/kaggle/`, `--register` đăng ký version mới + alias `production`

**Resume trên Kaggle** — mỗi epoch notebook ghi `last_*.pth` (model + optimizer +
scheduler + epoch + best + history) vào `/kaggle/working`. Lần chạy sau, `--resume-from`
khai báo kernel đó trong `kernel_sources` → output mount ở `/kaggle/input/<slug>/`
và notebook tự `load_state_dict` rồi chạy tiếp từ `epoch+1` (LR schedule giữ nguyên
vì `EPOCHS` là **tổng** số epoch). Checkpoint `best_*` của lần trước cũng được copy
sang nên không mất kết quả tốt nhất. Local cũng resume được:
`python scripts/dev/train_stage1_unet.py --epochs 30 --resume`.

**Bấm nút trên GitHub** (`.github/workflows/kaggle-train.yml`): thêm 2 secrets
`KAGGLE_USERNAME` / `KAGGLE_KEY` rồi **Actions → Train on Kaggle (GPU) → Run workflow**
(chọn stage, epoch, có cập nhật dataset hay không) — checkpoint tải về làm artifact
14 ngày, không cần bật máy local.

**Ghi chú**
- Dùng **Kaggle CLI** (subprocess) chứ không dùng SDK Python: SDK 2.x đã bỏ
  `dataset_create_new`/`dataset_create_version`, còn subcommand CLI thì ổn định.
- Notebook Kaggle tự chứa (không import `src/`), nên upload lên là chạy được; kiến
  trúc model đã được kiểm tra khớp `state_dict` với `src/dl/*` → checkpoint dùng lẫn được.
- `--check` dò username từ `KAGGLE_USERNAME` → `~/.kaggle/kaggle.json` → `kaggle config view`;
  không dò được thì truyền `--username`.
- `--no-wait` để push rồi thoát; `--accelerator NvidiaTeslaT4` ép loại GPU.
- `--mask-input none` bỏ hẳn mask (đối chứng); `masked` = cách cũ `ảnh*phổi`;
  `3ch` (mặc định) = `[ảnh, ảnh*phổi, ảnh*(phổi∪tim)]`.
- `--no-aug` / `--no-tta` / `--no-calibrate` để chạy ablation.
- `--wait-only` để **không push lại**: chỉ poll kernel đang chạy trên Kaggle rồi tải
  output + đăng ký — dùng khi phiên chờ bị gián đoạn (máy restart) mà run vẫn sống.

### 6b. Cải tiến thuật toán Stage-2 (v1.1)

Toàn bộ nằm trong `notebooks/kaggle/stage2_densenet121.ipynb`, **không đổi kiến trúc và
không đổi `state_dict`** (DenseNet121 vẫn 3 kênh) → checkpoint cũ vẫn load được, nhưng
phải train lại mới thấy hiệu quả.

| Thay đổi | Config | Vì sao |
|---|---|---|
| **Mask 3 kênh ngữ nghĩa** `[ảnh, ảnh*phổi, ảnh*(phổi∪tim)]` | `MASK_INPUT='3ch'` | Cách cũ `ảnh*phổi` **xoá sạch bóng tim + trung thất** → Cardiomegaly (định nghĩa là tỉ lệ tim/ngực), Effusion, Mass sát trung thất mất bằng chứng |
| **Augmentation đồng bộ ảnh + mask**: RRC scale ≥ 0.85, hflip, xoay ±7° | `AUG=True`, `AUG_SCALE`, `AUG_ROT` | Trước đây **không có augmentation nào**. Lấy tham số hình học **một lần** rồi áp cho cả ảnh và mask → không bị lệch mask |
| **TTA hflip** | `TTA=True` | Trung bình logits ảnh gốc và ảnh lật; ~+0.2–0.4 mAUROC, gần như miễn phí |
| **Temperature scaling + threshold từng nhãn** | `CALIBRATE=True` | Fit 1 tham số/nhãn trên **val** (grid search BCE), chọn ngưỡng tối đa Youden J, báo cáo sensitivity/specificity trên test — để xác suất dùng được cho bác sĩ |
| **Gradient clipping 5.0** | — | Ổn định khi chạy AMP |

Ablation (mỗi lần đổi **một** biến):

```powershell
# Đối chứng đúng nghĩa "trước đây": mask nhân cứng, không aug, không TTA
... --stage 2 --mask-input masked --no-aug --no-tta --no-calibrate

# Mask có giúp gì không?
... --stage 2 --mask-input none --no-aug --no-tta

# Chỉ thêm augmentation (giữ cách mask cũ) -> đo riêng tác dụng của aug
... --stage 2 --mask-input masked --no-tta
```

**Kết quả ablation Stage-2 trên dev** (8 epoch, cùng seed/data, mUAROC 14 bệnh
trên test 344 ảnh — chỉ mang giá trị **định hướng** vì tập test quá nhỏ):

| Cấu hình | val mAUROC | test mAUROC | Δ vs v1.1 |
|---|---|---|---|
| **v1.1 đầy đủ**: `3ch` + aug + TTA + calibration | 0.7052 | **0.6873** | — |
| Đối chứng cũ: `masked` + không aug + không TTA | 0.7097 | 0.6685 | −0.0188 |
| Bỏ hẳn mask: `none` + không aug + không TTA | 0.7043 | 0.6375 | −0.0498 |

- **Mask có giúp thật**: cùng tắt aug/TTA, `masked` hơn `none` **+0.0310
  mAUROC** (0.6685 vs 0.6375) — mask phổi/tim mang bằng chứng, không phải trang trí.
- **Gói v1.1 (3ch + aug + TTA) hơn cách cũ +0.0188** (0.6873 vs 0.6685).
- Val của ba run gần như giống nhau (0.704–0.710) trong khi test dao động
  0.638–0.687 → đúng dự đoán: với 321 val / 344 test, con số chỉ để so sánh
  **xu hướng**, không phải hiệu năng tuyệt đối — số liệu chốt là run 112k (**mục 6c**,
  test 0.8331).

Notebook ghi `calibration.temperature` và `calibration.threshold` (mỗi nhãn) vào
`stage2_metrics.json`; serving (FastAPI/Streamlit) đọc lại để biến logit thành xác suất
đã hiệu chỉnh thay vì dùng sigmoid thô.

### 6c. Kết quả train full 112k → model production

Chạy **lệnh 6** ở mục *Train trên Kaggle*: 30 epoch trên T4, ~450 s/epoch ≈ **3,9 giờ**,
dữ liệu 112.120 ảnh / 30.805 bệnh nhân (train 77.728 / val 17.662 / test 16.730),
mask pack **thiếu 0/112.120**, `mask_input=3ch + aug + TTA + calibration`.

| Chỉ số | Dev 2k (test 344 ảnh) | **Full 112k (test 16.730 ảnh)** |
|---|---|---|
| val mAUROC | 0.7052 | 0.8196 |
| **test mAUROC** | 0.6873 | **0.8331** |

- **Best val ở epoch 5** (0.8168); loss train tiếp tục tụt 0.82 → 0.20 trong khi val
  rơi về 0.776 ở epoch 30 → **overfit sau epoch ~5**, 30 epoch là dư — lần sau rút
  còn 10–15 epoch (hoặc tăng weight decay / dropout) sẽ tiết kiệm ~2/3 GPU.
- Test 0.8331 xấp xỉ mức CheXNet báo cáo trên ChestX-ray14 (0.8414) — **có caveat**:
  split của ta là patient-level 112k/16.730 ảnh, không phải split 24.429 ảnh chính
  thức của NIH nên chỉ so sánh được mặt ý nghĩa, không phải bảng so sánh trực tiếp.

Từng nhãn trên test (đã TTA + hiệu chỉnh temperature + ngưỡng Youden trên val):

| Nhãn | AUC | Ngưỡng | Sens | Spec |
|---|---|---|---|---|
| Hernia | 0.929 | 0,07 | 0,652 | 0,988 |
| Emphysema | 0,906 | 0,37 | 0,789 | 0,868 |
| Edema | 0,905 | 0,19 | 0,866 | 0,792 |
| Cardiomegaly | 0,893 | 0,26 | 0,789 | 0,830 |
| Effusion | 0,881 | 0,50 | 0,827 | 0,778 |
| Pneumothorax | 0,877 | 0,49 | 0,797 | 0,820 |
| Mass | 0,839 | 0,38 | 0,701 | 0,821 |
| Atelectasis | 0,811 | 0,50 | 0,776 | 0,690 |
| Pleural_Thickening | 0,801 | 0,22 | 0,741 | 0,705 |
| Consolidation | 0,789 | 0,45 | 0,766 | 0,674 |
| Fibrosis | 0,785 | 0,05 | 0,676 | 0,730 |
| Pneumonia | 0,774 | 0,05 | 0,634 | 0,760 |
| Nodule | 0,764 | 0,39 | 0,637 | 0,747 |
| Infiltration | 0,710 | 0,52 | 0,552 | 0,754 |
| **mean** | **0,833** | | | |

- **Đăng ký**: `chexspark-stage2-densenet121` **version 3, alias `production`**
  (MLflow run `3dd0f14b7a044dd9aa04ad25ad17bd34`) — model này là model được
  FastAPI/Streamlit phục vụ.
- Output tải về `data/kaggle/output/stage2-112k/` (`best_*.pth` 27 MB, `last_*.pth`
  82 MB, `stage2_densenet121_final.pth`, `stage2_metrics.json`) và cài vào
  `notebooks/kaggle/`.
- Nếu phiên chờ bị gián đoạn (máy restart) mà kernel vẫn chạy trên Kaggle: chạy lại
  với **`--wait-only`** — chỉ poll kernel đang chạy rồi tải + đăng ký, không push lại.

## Orchestration với Apache Airflow

Airflow điều phối toàn bộ pipeline Medallion: mỗi bước là một DAG riêng (chạy lại
độc lập được khi lỗi), cộng một DAG tổng điều phối thứ tự.

| DAG | Task | Việc |
|---|---|---|
| `chexspark_bronze_to_silver` | `ingest_bronze_to_silver` | 112k metadata → Silver `cxr14` (patient-level split) |
| `chexspark_chexmask_join` | `ingest_chexmask_to_silver` → `precompute_masks_224` | CheXmask RLE → Silver `cxr14_masked` (broadcast join) → mask 224 packed+PNG |
| `chexspark_gold_analytics` | `disease_cooccurrence` → `disease_graph` → `verify_gold` | Gold: co-occurrence + mạng bệnh lý |
| `chexspark_pipeline` | 3 × `TriggerDagRunOperator` | Chạy tuần tự 3 DAG trên và **chờ** từng cái xong |
| `chexspark_retrain_112k` | 7 task xếp dọc (Spark → export → Kaggle → MLflow) | **Vòng retrain 112k**: chuẩn bị dữ liệu, train trên Kaggle GPU, đăng ký `production` |

```powershell
# Bật hạ tầng (lần đầu sẽ build image chexspark-airflow ~1GB do kèm PySpark)
docker compose up -d --build

# Xem DAG đã được parse
docker compose exec airflow-scheduler airflow dags list

# Chạy toàn pipeline (trigger thủ công)
docker compose exec airflow-scheduler airflow dags trigger chexspark_pipeline

# Hoặc chạy lẻ từng bước
docker compose exec airflow-scheduler airflow dags trigger chexspark_gold_analytics

# Xem trạng thái các run
docker compose exec airflow-scheduler airflow dags list-runs -d chexspark_pipeline
```

UI: http://localhost:8085 (`admin`/`admin`), DAGs bật sẵn (`DAGS_ARE_PAUSED_AT_CREATION=false`).

**Cách Airflow nộp job Spark** (`dags/spark_utils.py`):
- `SparkSubmitOperator` gọi `spark-submit` (PySpark 3.5.1 cài trong image Airflow —
  **khớp đúng** phiên bản cluster `chexspark-spark:3.5.1`)
- **client mode**: driver chạy trong container `airflow-scheduler`, executor chạy
  trên `spark-worker-1/2`; `spark.driver.host=airflow-scheduler` để executor kết
  nối ngược lại driver qua mạng `chexspark-net`
- **Python 3.8** ở cả hai phía: image Airflow dùng `apache/airflow:2.9.3-python3.8`
  vì PySpark bắt buộc driver và worker **cùng minor version** (`apache/spark:3.5.1`
  dùng Python 3.8.10). Lệch version sẽ báo `PYTHON_VERSION_MISMATCH`
- S3A jars (`hadoop-aws`, `aws-java-sdk-bundle`) mount từ `infra/spark/jars` và nạp
  **chỉ cho driver** qua `spark.driver.extraClassPath` — **không** dùng `--jars` vì
  Spark sẽ đẩy thêm một bản jar xuống executor (executor đã có sẵn trong image),
  nạp trùng làm commit file trên S3A hỏng (`No such file or directory` khi rename)
- Cấu hình S3A/MinIO truyền qua `--conf` (driver không đọc `spark-defaults.conf`
  của cluster); dùng `mapreduce.fileoutputcommitter.algorithm.version=2` để job
  ghi lại nhiều lần vẫn ổn định
- Credentials MinIO lấy từ env (`AWS_ACCESS_KEY_ID`/`AWS_SECRET_ACCESS_KEY`) —
  **không** hard-code trong DAG
- `PYTHONPATH=/opt/airflow/src` để job import được `common.spark_session`

**Giới hạn tài nguyên** (cluster dev 8GB): `AIRFLOW__CORE__PARALLELISM=1`,
`DAG_CONCURRENCY=1`, driver `1g`, executor `1g`, `spark.cores.max=2` — tránh tranh
RAM với 2 Spark worker.

> **Lưu ý**: `chexspark_pipeline` chỉ chạy phần **data pipeline** (Bronze→Gold).
> Train Stage-2 (DenseNet121) và Stage-3 (Grad-CAM) chạy trên Kaggle GPU / local
> vì cần GPU. Riêng DAG `chexspark_retrain_112k` dưới đây **đẩy** job train lên
> Kaggle GPU rồi Airflow chờ xong và tự đăng ký model.

### DAG retrain 112k (`chexspark_retrain_112k`)

Vòng lặp MLOps chạy **một lệnh** thay cho các bước tay ở mục 5c + mục Train trên
Kaggle. **Trigger thủ công** (`schedule=None`) vì mỗi lần đầy đủ tốn ~4 GPU-giờ.
Không đụng bản dev 2k: ghi `silver/cxr14_full`, `silver/cxr14_masks_full`, dataset
Kaggle `chexspark-stage2-nih-112k` / `chexspark-masks-224-112k`.

| # | Task | Việc | Timeout |
|---|---|---|---|
| 1 | `ingest_silver_full` | Bronze → Silver full 112k (`ingest_bronze_to_silver.py --full`) | 45' |
| 2 | `precompute_masks_full` | Mask 224 packed từ Bronze CheXmask (`precompute_masks.py --source bronze`) | 90' |
| 3 | `stage_parquet_local` | Stage Parquet từ MinIO về `data/tmp/retrain_stage` (boto3) | 45' |
| 4 | `export_kaggle_datasets` | Export labels CSV + mask pack `.npy` vào `data/kaggle/*` | 120' |
| 5 | `push_training` | Upload 2 dataset + push kernel `--no-wait` | 3' |
| 6 | `wait_and_register` | Chờ kernel xong → tải output → đăng ký MLflow | 14' |
| 7 | `verify_production` | Xác nhận alias `production` trỏ version mới | 15' |

**Chạy thử không tốn GPU/quota** (`push_training=false` → task 5–7 in `[skip]`,
chỉ chạy 4 task dữ liệu, ~7 phút):

```powershell
# PowerShell cắt mất dấu " trong JSON --conf → ghi ra file rồi docker cp
Set-Content -Path $env:TEMP\retrain_conf.json -Value '{"push_training":false}' -NoNewline
docker cp $env:TEMP\retrain_conf.json chexspark-airflow-scheduler:/tmp/retrain_conf.json
docker exec chexspark-airflow-scheduler sh -c 'airflow dags trigger chexspark_retrain_112k --conf $(cat /tmp/retrain_conf.json)'

# Xem trạng thái (task 1–4 success, 5–7 success với log [skip])
docker exec chexspark-airflow-scheduler airflow dags list-runs -d chexspark_retrain_112k --no-backfill
docker exec chexspark-airflow-scheduler airflow tasks states-for-dag-run chexspark_retrain_112k <run_id>
```

**Chạy đầy đủ** (train 30 epoch trên T4 ~4h rồi tự đăng ký `production`): trigger
không cần `--conf` (param `push_training` mặc định `True`), hoặc bấm nút **Trigger
DAG** trên UI http://localhost:8085 và bật toggle `push_training`.

**Chống chạy trùng & khôi phục sau gián đoạn**:
- Task 5 `retries=0` **cố ý**: push lại sẽ khởi động lần train thứ hai trên GPU →
  không retry, chỉ chạy tay sau khi xác nhận kernel cũ đã dừng.
- Task 6 `--wait-only` **chạy lại được**: poll kernel đang chạy rồi tải + đăng ký,
  không push lại. Nếu run rơi khi đang chờ thì **clear riêng task 6** rồi chạy tiếp.
- Task 3 stage vào `data/tmp/retrain_stage` do chính task tạo (thuộc uid airflow);
  không dùng `data/tmp/silver_full` cũ (755 thuộc uid khác → `PermissionError` khi
  xóa lần chạy trước).

## MLOps: MLflow Model Registry

Checkpoint không trỏ bằng đường dẫn file mà **đăng ký thành model version** trong
MLflow. Mỗi lần train (kể cả trên Kaggle) đăng ký một version bất biến, gắn đúng
run + params + metrics của lần train đó; serving luôn nạp theo **alias** nên
promote/rollback không cần sửa code.

| Stage | Registered model | Experiment | Artifact |
|---|---|---|---|
| 1 — U-Net | `chexspark-stage1-unet` | `chexspark-stage1-registry` | MinIO `s3://mlflow/…` (~8 MB) |
| 2 — DenseNet121 | `chexspark-stage2-densenet121` | `chexspark-stage2-registry` | MinIO `s3://mlflow/…` (~27 MB) |

Alias chung: `production` (URI `models:/<tên>@production`); metadata ở PostgreSQL
(`mlflow` DB).

**Đăng ký + promote + kiểm chứng**

```powershell
$env:MLFLOW_TRACKING_URI="http://localhost:5000"   # chỉ cần tracking URI

D:\BigData\venv-mlops\Scripts\python scripts/dev/register_model.py --stage 1   # U-Net
D:\BigData\venv-mlops\Scripts\python scripts/dev/register_model.py --stage 2   # DenseNet121
```

Server chạy ở chế độ **artifact proxy** (`--serve-artifacts --artifacts-destination s3://mlflow
--default-artifact-root mlflow-artifacts:/`): client đẩy artifact qua endpoint
`/api/2.0/mlflow-artifacts/…` của server, server mới ghi lên MinIO. Nhờ vậy máy
train/serving **không cần** `boto3` hay credential MinIO.

Script làm 4 việc: log run (params + metric chính của stage), log model kèm
**signature** (`(1,3,224,224) → (1,14)` cho Stage-2, `(1,1,224,224) → (1,1,224,224)`
cho Stage-1), gán alias `production`, rồi **nạp lại từ registry** và so sánh logits
với checkpoint local (lệch phải < 1e-5).

**Promote / rollback version khác**

```powershell
D:\BigData\venv-mlops\Scripts\python scripts/dev/register_model.py --stage 2 --list
D:\BigData\venv-mlops\Scripts\python scripts/dev/register_model.py --stage 2 --promote 1
```

**Serving nạp model theo alias** (`src/mlops/registry.py`)

```python
from mlops.registry import load_stage1_from_registry, load_stage2_from_registry

unet = load_stage1_from_registry()      # models:/chexspark-stage1-unet@production
model = load_stage2_from_registry()     # models:/chexspark-stage2-densenet121@production
```

**Trạng thái registry sau khi chạy**

| version | checkpoint | alias | val mAUROC | test mAUROC |
|---|---|---|---|---|
| 1 | `best_stage2_densenet121.pth` | **production** | 0.6752 | 0.6269 |
| 2 | `stage2_densenet121_final.pth` | – | – | – |

Version 2 (epoch cuối) được đăng ký để thử promote rồi **rollback** về version 1
(best theo val) bằng `--promote 1`. Kiểm chứng nạp từ registry cho
`max |logit_local − logit_registry| = 0.000e+00`, output `(1, 14)` sau sigmoid
nằm trong `[0.057, 0.687]`.

**Ghi chú thiết kế**
- `venv-mlops` tách riêng vì `mlflow==2.9.2` (khớp server) pin `numpy<2`/`pandas<3`,
  không nên trộn vào venv train/Grad-CAM đang dùng numpy 2.x.
- Alias là cơ chế promote hiện đại; *stages* (`Staging`/`Production`) đã deprecated
  từ MLflow 2.9 nên không dùng.
- Signature + `pip_requirements` (`torch`, `torchvision`, `numpy`) được log kèm để
  model tái lập được ở môi trường khác, không phụ thuộc máy train.
- Image `chexspark-mlflow:2.9.2` (`infra/mlflow/Dockerfile`) = image chính thức +
  `psycopg2-binary` (backend PostgreSQL) + `boto3` (artifact MinIO). Image gốc
  thiếu cả hai, khi đó MLflow **âm thầm** rơi về sqlite + filesystem trong container
  và mất sạch metadata/artifact mỗi lần recreate.
- Image Airflow cài thêm `mlflow==2.9.2` vì driver Spark (client mode) chạy *trong*
  container Airflow: thiếu nó job vẫn báo success nhưng in
  `[WARN] MLflow logging failed` và bỏ qua toàn bộ tracking.
- Train Stage-2 chạy trên GPU Kaggle nên **đăng ký model là bước riêng** (thủ công
  hoặc CI) sau khi tải checkpoint về, không nằm trong DAG Spark — DAG chỉ lo dữ
  liệu Bronze→Gold.

## Ghi chú

- Bucket `mlflow` được tạo tự động bởi service `minio-init`; các bucket dữ liệu (`bronze`, `silver`, `gold`, `checx`) tạo qua MinIO console hoặc `mc`.
- Credential MinIO được truyền vào Spark/MLflow qua biến môi trường `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` — **không** hard-code trong code hay `spark-defaults.conf`.
- Đảm cấp Docker Desktop có ≥ 8 GB RAM để chạy đủ 2 Spark workers (mỗi worker mặc định 4G, chỉnh trong `.env`).
