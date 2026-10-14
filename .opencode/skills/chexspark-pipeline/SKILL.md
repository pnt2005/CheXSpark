---
name: chexspark-pipeline
description: Comprehensive architecture, data lakehouse, Apache Spark, 3-stage Deep Learning pipeline, and MLOps guide for CheXSpark.
---

# CheXSpark Pipeline — Big Data & Medical Deep Learning Skill

## 1. Project Overview & Scope
CheXSpark is an end-to-end Big Data & Deep Learning system for detecting, classifying, and visualizing multi-label thoracic pathologies from chest X-ray radiographs (CXR) utilizing **Apache Spark** and **Deep Learning**.

- **Dataset**: NIH ChestX-ray14 (112,120 frontal CXR images, 14 thoracic disease labels, ~42GB) + **CheXmask** (anatomical lung segmentation masks generated via HybridGNet, mapped 1-to-1 by Image ID).
- **Core Paradigm**: Data Lakehouse (Bronze -> Silver -> Gold) on **MinIO** (Object Storage) & **PostgreSQL** (Metadata catalog).
- **Processing Engine**: Apache Spark (Core, SQL, GraphX, MLlib, Distributed DL).
- **Deep Learning Pipeline**: U-Net (Segmentation) -> DenseNet121/ResNet50 (Multi-label Classification) -> Grad-CAM (Heatmap Localization).
- **MLOps & Serving**: Docker Compose, Apache Airflow, MLflow, FastAPI backend, Streamlit diagnostic dashboard.

---

## 2. System Architecture & Data Lakehouse

```
[Raw Sources: NIH14 PNG/DICOM + CheXmask Masks + CSV Metadata]
                         │
                         ▼
        ┌───────────────────────────────────┐
        │       BRONZE LAYER (MinIO)        │
        │ Raw CXR images, raw masks, CSVs   │
        └─────────────────┬─────────────────┘
                          │ Apache Spark Core & SQL ETL
                          ▼
        ┌───────────────────────────────────┐
        │       SILVER LAYER (MinIO/Parquet)│
        │ - Cleaned metadata, no missing    │
        │ - Validated image-mask pairs      │
        │ - Stratified Train/Val/Test split │
        │ - Balanced class distribution     │
        └─────────────────┬─────────────────┘
                          │ Feature extraction & Graph analytics
                          ▼
        ┌───────────────────────────────────┐
        │       GOLD LAYER (MinIO/Postgres) │
        │ - DenseNet121 1024-d embeddings   │
        │ - Disease co-occurrence graphs    │
        │ - Aggregated clinical statistics  │
        │ - MLlib models & benchmark logs   │
        └───────────────────────────────────┘
```

---

## 3. Apache Spark Subsystems Specification

### A. Spark Core (Distributed Ingestion & ETL)
- Ingest and process metadata of 112,120 images.
- Handle missing values, filter corrupted image headers or unreadable formats.
- Generate patient-aware stratified splits (Train/Val/Test) to prevent patient leakage across sets.
- Statistical computation of label distribution across 14 pathology classes; address heavy class imbalance (e.g. Hernia rare vs Infiltration frequent) via positive class weighting (`pos_weight`).
- Store processed metadata in snappy-compressed, partitioned **Parquet** format.

### B. Spark SQL (Clinical Analytics & Cohort Queries)
- Query disease frequency rankings: `SELECT disease, COUNT(*) FROM ... GROUP BY disease ORDER BY count DESC`.
- Specific condition metrics: counts and demographic distributions for Pneumonia, Infiltration, Effusion, etc.
- Multi-morbidity analysis: identify images containing $\ge 2$ simultaneous pathologies, calculate correlation matrix between diseases.

### C. Spark GraphX (Disease Co-occurrence Network)
- Model diseases as a network graph:
  - **Vertices (Nodes)**: 14 thoracic pathologies (Atelectasis, Cardiomegaly, Effusion, Infiltration, Mass, Nodule, Pneumonia, Pneumothorax, Consolidation, Edema, Emphysema, Fibrosis, Pleural_Thickening, Hernia).
  - **Edges**: Co-occurrence relationships between pairs of diseases found on the same patient CXR.
  - **Edge Weights**: Frequency of co-occurrence or Jaccard similarity coefficient.
- Graph Algorithms:
  - **Degree Centrality**: Identify hub diseases (diseases with high co-morbidity).
  - **Connected Components & Community Detection**: Identify clusters of diseases that regularly manifest together.

### D. Spark MLlib (Post-Feature Extraction Analytics)
- Extract 1024-dimensional feature embeddings from the penultimate layer of DenseNet121.
- Perform classical ML & statistical modeling on Spark MLlib:
  - **Dimensionality Reduction**: Principal Component Analysis (PCA) for visual cluster separation.
  - **Clustering**: K-Means clustering on embeddings to reveal unsupervised patient groupings.
  - **Baseline Classifiers**: Logistic Regression & Random Forest classifiers trained on deep feature vectors for comparison against end-to-end neural network predictions.

### E. Distributed Deep Learning Benchmark
- Compare execution performance, throughput (images/sec), convergence time, and resource utilization between **Single Machine GPU training** vs **Spark Cluster Distributed training** (e.g. Horovod / TorchDistributor).

---

## 4. 3-Stage Deep Learning Pipeline

### Stage 1: Lung Field Segmentation (U-Net)
- **Model**: U-Net architecture (encoder-decoder with skip connections).
- **Input**: 1-channel or 3-channel frontal CXR ($256 \times 256$ or $512 \times 512$).
- **Ground Truth**: Binary lung masks from CheXmask (HybridGNet).
- **Objective**: Crop out redundant background, soft tissues, and rib edges, preserving only anatomical lung parenchyma.
- **Loss**: Combined Binary Cross Entropy + Dice Loss ($L = L_{BCE} + L_{Dice}$).

### Stage 2: Multi-Label Pathology Classification (DenseNet121 / ResNet50)
- **Model**: DenseNet121 (or ResNet50) pre-trained on ImageNet, adapted with a 14-output linear classification head with Sigmoid activation.
- **Pathologies (14 classes)**:
  1. Atelectasis
  2. Cardiomegaly
  3. Effusion
  4. Infiltration
  5. Mass
  6. Nodule
  7. Pneumonia
  8. Pneumothorax
  9. Consolidation
  10. Edema
  11. Emphysema
  12. Fibrosis
  13. Pleural_Thickening
  14. Hernia
- **Input**: Segmented/masked lung region obtained from Stage 1.
- **Loss Function**: `BCEWithLogitsLoss` with `pos_weight` to counteract severe class imbalance.
- **Evaluation Metrics**: Per-class AUC-ROC, mean AUC (mAUROC), F1-Score, Precision-Recall curves.

### Stage 3: Explainability & Heatmap Detection (Grad-CAM)
- **Method**: Gradient-weighted Class Activation Mapping (Grad-CAM).
- **Mechanism**: Compute gradients of the score for pathology $c$ with respect to feature activation maps of the final convolutional layer of DenseNet121 (`features.denseblock4.denselayer16.conv2`).
- **Output**: 2D heatmaps overlaid on the original CXR, localizing suspected pathological lesions (e.g., fluid accumulation in pleural effusion, dense consolidation in pneumonia).

---

## 5. Input & Output Contract

### Ingestion Contract
- **Input format**: PNG or DICOM format CXR images.
- **Metadata**: Image Index, Finding Labels, Follow-up #, Patient ID, Patient Age, Patient Gender, View Position.
- **Mask Matching**: Direct join on `Image ID` with CheXmask dataset.

### Inference Response Contract
For every CXR processed, the system outputs:
1. **Segmented Mask**: Binary/contour mask isolating left and right lung fields.
2. **Pathology Prediction List**: Array of 14 pathologies, each with:
   - `disease_name`: string
   - `probability`: float (0.0 to 1.0)
   - `threshold`: float (e.g., 0.5 or optimal operating point)
   - `is_positive`: boolean
3. **Grad-CAM Visual Heatmap**: Normalized overlay image highlighting lesion areas for clinician verification.

---

## 6. MLOps, Containerization & Application Stack

- **Docker Compose**:
  - `spark-master` & `spark-worker`
  - `minio`: S3-compatible storage with default buckets (`bronze`, `silver`, `gold`)
  - `postgres`: Database for metadata, Airflow backend, and MLflow backend
  - `airflow-webserver` & `airflow-scheduler`: Pipeline scheduling
  - `mlflow-server`: Experiment tracking & artifact repository
  - `fastapi-app`: Model serving REST API
  - `streamlit-app`: Interactive UI
- **FastAPI Endpoints**:
  - `POST /api/v1/predict`: Accepts uploaded CXR, executes 3-stage inference pipeline, returns JSON findings + base64/URL heatmap and mask.
  - `GET /api/v1/analytics/disease-distribution`: Retrieves Spark-aggregated statistics from Gold layer.
  - `GET /api/v1/graph/co-occurrence`: Returns GraphX disease co-occurrence nodes and edges for visualization.
- **Streamlit Dashboard**:
  - Image upload with DICOM/PNG viewer.
  - Side-by-side display of: Original CXR, Lung Mask, Segmented Lung, Grad-CAM Heatmap overlay.
  - Interactive probability scorecards & threshold adjustments.
  - Interactive disease co-occurrence network explorer.
