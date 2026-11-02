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

> **Lưu ý image**: `apache/spark` mặc định **không kèm S3A jars** (không đọc được MinIO) và **không có mlflow**.
> Vì vậy ta build image riêng `chexspark-spark:3.5.1` từ `infra/spark/Dockerfile`, thêm
> `hadoop-aws` + `aws-java-sdk-bundle` (đọc/ghi MinIO) và cài `mlflow==2.9.2` (khớp server).
> Image MinIO dùng bản `bitnamilegacy` do `minio/minio` trên Docker Hub đã bị restrict.

## Cấu trúc repo

```
CheXSpark/
├── docker-compose.yml            # MinIO + PostgreSQL + Spark cluster + MLflow
├── .env.example                  # Biến môi trường (copy → .env trước khi chạy)
├── infra/
│   ├── postgres/init/            # SQL bootstrap (tạo DB mlflow, chexspark_metadata)
│   └── spark/
│       ├── Dockerfile            # Build image chexspark-spark (S3A jars + mlflow)
│       ├── jars/                 # hadoop-aws + aws-java-sdk-bundle (cho S3A/MinIO)
│       └── conf/                 # spark-defaults.conf (S3/MinIO, MLflow URI)
├── src/
│   ├── common/spark_session.py   # SparkSession factory (dùng chung mọi job)
│   ├── ingestion/                # Spark Core: Bronze → Silver (ingest, mask join, verify)
│   ├── silver/                   # (sẵn sàng) cleaning, stratified split
│   ├── gold/                     # co-occurrence, disease network, MLlib PCA/K-Means/LR/RF
│   ├── dl/                       # preprocess, DenseNet121 loader, Grad-CAM (PyTorch)
│   └── serving/                  # FastAPI inference + Streamlit UI
├── notebooks/
│   └── kaggle/stage2_densenet121.ipynb   # Train Stage-2 trên Kaggle GPU
├── tests/                        # Unit & integration tests
└── scripts/
    └── dev/                      # extract_subset, export Silver→Kaggle, features, Grad-CAM, smoke tests
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

### 6. Gold layer — phân tích dịch tễ & mạng bệnh lý (Spark)

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

### 7. Gold layer — MLlib trên feature vectors DenseNet121

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

### 8. Stage-3 — Grad-CAM (Explainability)

Sinh bản đồ nhiệt từ feature map tích chập cuối của DenseNet121 để khoanh vùng
tổn thương nghi ngờ cho từng bệnh (output thứ 3 của hệ thống, cạnh mask phổi và
danh sách xác suất).

```powershell
# Toàn bộ 14 bệnh, top-4 ca dương tính rõ nhất trong test
python scripts/dev/run_gradcam_batch.py --top-k 4

# Chỉ vài bệnh, hoặc đổi lớp đích (denseblock3 = 14x14 nét hơn, denseblock4 = 7x7)
python scripts/dev/run_gradcam_batch.py --diseases Cardiomegaly,Effusion --target denseblock4
```

**Module tái sử dụng** (dùng lại cho FastAPI/Streamlit):
- `src/dl/preprocess.py` — decode RLE mask, load ảnh xám, build tensor (đúng
  pipeline lúc train: 224 → nhân lung mask → ImageNet normalize)
- `src/dl/model.py` — build/load DenseNet121 từ checkpoint Stage-2
- `src/dl/gradcam.py` — `GradCAM` class (forward/backward hook), colormap `jet`
  thuần numpy, `overlay_heatmap` (chỉ giữ CAM trong vùng phổi)

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

## Ghi chú

- Bucket `mlflow` được tạo tự động bởi service `minio-init`; các bucket dữ liệu (`bronze`, `silver`, `gold`, `checx`) tạo qua MinIO console hoặc `mc`.
- Credential MinIO được truyền vào Spark/MLflow qua biến môi trường `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` — **không** hard-code trong code hay `spark-defaults.conf`.
- Đảm cấp Docker Desktop có ≥ 8 GB RAM để chạy đủ 2 Spark workers (mỗi worker mặc định 4G, chỉnh trong `.env`).
