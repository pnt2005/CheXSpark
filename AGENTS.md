# AGENTS.md — CheXSpark OpenCode Agent Instructions

Chào mừng đến với không gian làm việc của dự án **CheXSpark**!

---

## 1. Tổng Quan Dự Án (Project Overview)
**Tên đề tài**: XÂY DỰNG HỆ THỐNG PHÁT HIỆN VÀ TRỰC QUAN HÓA ĐA BỆNH LÝ LỒNG NGỰC TỪ ẢNH X-QUANG NGỰC SỬ DỤNG DEEP LEARNING VÀ APACHE SPARK.

Hệ thống giải quyết bài toán phát hiện đa bệnh lý lồng ngực từ ảnh X-quang ngực thẳng (CXR) ở quy mô dữ liệu lớn (Big Data), tích hợp kiến trúc Data Lakehouse, xử lý phân tán với Apache Spark, pipeline học sâu 3 giai đoạn (Segmentation, Classification, Visualization), quản lý MLOps và giao diện phục vụ bác sĩ.

---

## 2. Kiến Trúc Hệ Thống (System Architecture)

### 2.1. Kiến Trúc Data Lakehouse (Bronze - Silver - Gold)
- **Lưu trữ & Dữ liệu**: **MinIO** (S3-compatible Object Storage cho ảnh và mask) kết hợp **PostgreSQL** (metadata catalog và relational storage).
- **Tầng Bronze**: Lưu trữ dữ liệu thô gồm ảnh X-quang (PNG/DICOM), file mask thô từ CheXmask, và file metadata CSV gốc.
- **Tầng Silver**: Dữ liệu đã làm sạch qua Spark Core & SQL, loại bỏ dữ liệu lỗi/thiếu, map chuẩn hóa Image ID giữa ảnh và mask, phân chia tập Train/Val/Test phân tầng (stratified) tránh rò rỉ bệnh nhân (patient leakage), lưu trữ dưới dạng **Parquet** được partition tối ưu.
- **Tầng Gold**: Feature vectors 1024 chiều trích xuất từ DenseNet121, ma trận đồng xuất hiện bệnh tật (co-occurrence matrix), đồ thị GraphX, kết quả phân tích MLlib và thống kê phục vụ dashboard.

### 2.2. Pipeline 3 Mô Hình Học Sâu (Deep Learning Pipeline)
1. **Giai đoạn 1 - Phân đoạn phổi (Lung Segmentation)**:
   - Kiến trúc: **U-Net**
   - Dữ liệu huấn luyện/fine-tune: Ảnh NIH ChestX-ray14 kết hợp ground-truth mask từ **CheXmask** (sinh từ HybridGNet).
   - Mục tiêu: Tách trường phổi, loại bỏ nhiễu nền, mô mềm và xương sườn chồng lấp để tăng độ chính xác cho mô hình phân loại.
2. **Giai đoạn 2 - Phân loại đa nhãn (Multi-Label Classification)**:
   - Kiến trúc: **DenseNet121** / **ResNet50**
   - Đầu vào: Vùng phổi đã được khoanh vùng/cắt lọc từ Giai đoạn 1.
   - Mục tiêu: Dự đoán xác suất cho **14 nhãn bệnh lý lồng ngực** (*Atelectasis, Cardiomegaly, Effusion, Infiltration, Mass, Nodule, Pneumonia, Pneumothorax, Consolidation, Edema, Emphysema, Fibrosis, Pleural_Thickening, Hernia*).
   - Xử lý mất cân bằng dữ liệu: `BCEWithLogitsLoss` kết hợp `pos_weight` hoặc Focal Loss.
3. **Giai đoạn 3 - Trực quan hóa & Giải thích (Explainability & Detection)**:
   - Kỹ thuật: **Grad-CAM (Gradient-weighted Class Activation Mapping)**
   - Mục tiêu: Sinh bản đồ nhiệt (heatmap) từ các feature map tích chập cuối của mô hình phân loại, khoanh vùng tổn thương nghi ngờ trên ảnh để bác sĩ đối chiếu và xác minh lâm sàng thay vì chỉ nhận một con số xác suất.

### 2.3. Hệ Sinh Thái Apache Spark
- **Spark Core**:
  - Đọc và xử lý metadata của hàng trăm nghìn ảnh (112.120 ảnh, ~42GB).
  - Phân tích phân bố bệnh, xử lý missing data, lọc dữ liệu lỗi.
  - Tạo phân chia tập Train/Validation/Test split đảm bảo tính độc lập giữa các bệnh nhân.
  - Thống kê số lượng từng bệnh, tính toán trọng số xử lý mất cân bằng nhãn (class imbalance).
  - Lưu trữ metadata chuẩn hóa thành định dạng Parquet kèm partition tối ưu.
- **Spark SQL**:
  - Truy vấn phân tích dịch tễ học và chẩn đoán: Bệnh nào phổ biến nhất? Có bao nhiêu ảnh Pneumonia? Có bao nhiêu ảnh chứa $\ge 2$ bệnh (bệnh đồng mắc - comorbidity)?
- **Spark GraphX**:
  - Xây dựng mạng lưới liên kết bệnh lý (**Disease Co-occurrence Network**): Nodes là 14 loại bệnh, Edges là tần suất cùng xuất hiện trên cùng một bệnh nhân.
  - Phân tích đồ thị: Bệnh nào có nhiều liên kết nhất (degree centrality), những bệnh nào thường đi cùng nhau, phát hiện cộng đồng bệnh (community detection).
- **Spark MLlib**:
  - Sử dụng feature vectors trích xuất từ DenseNet121 để thực hiện:
    - Giảm chiều dữ liệu: **PCA** trực quan hóa không gian đặc trưng.
    - Phân cụm: **K-Means** phát hiện các nhóm bệnh nhân/đặc trưng ẩn.
    - Mô hình baseline truyền thống: **Logistic Regression**, **Random Forest** so sánh với Deep Learning.
- **Distributed Deep Learning**:
  - Đánh giá và so sánh hiệu năng huấn luyện giữa Single Machine và Spark Cluster (thời gian huấn luyện, throughput ảnh/giây, khả năng mở rộng).

### 2.4. Hạ Tầng MLOps & Triển Khai
- **Docker Compose**: Đóng gói toàn bộ hệ thống (Spark Master/Worker, MinIO, PostgreSQL, Airflow, MLflow, FastAPI, Streamlit).
- **Apache Airflow**: Quản lý và lập lịch các data pipeline, DAGs từ Ingestion -> ETL Spark -> Model Inference/Retraining.
- **MLflow**: Theo dõi thực nghiệm (experiment tracking), ghi nhận loss, AUC-ROC, Dice score, và quản lý model registry.
- **FastAPI**: Cung cấp RESTful API endpoint phục vụ dự đoán thời gian thực.
- **Streamlit**: Giao diện web trực quan cho phép tải ảnh X-quang, hiển thị mask phân đoạn phổi, danh sách xác suất 14 bệnh và lớp phủ heatmap Grad-CAM.

---

## 3. Đặc Tả Dữ Liệu Đầu Vào & Đầu Ra (Input / Output Specs)

### 3.1. Dữ Liệu Đầu Vào
- **Ảnh X-quang ngực thẳng**: Định dạng PNG/DICOM từ bộ dữ liệu **NIH ChestX-ray14** (112.120 ảnh, 14 nhãn bệnh lý, kích thước ~42GB).
- **CheXmask**: Bộ mask phân vùng phổi được sinh sẵn cho chính tập NIH ChestX-ray14 bằng mô hình HybridGNet, ghép nối trực tiếp theo `Image ID` mà không cần huấn luyện qua dataset trung gian.

### 3.2. Kết Quả Đầu Ra
Với mỗi ảnh đầu vào, hệ thống cung cấp 3 thành phần:
1. **Mask phân vùng phổi**: Ranh giới trường phổi đã được phân đoạn bằng U-Net.
2. **Danh sách bệnh lý nghi ngờ kèm độ tin cậy**: Dự đoán đa nhãn trong 14 loại bệnh với xác suất định lượng (0.0 – 1.0) và cờ cảnh báo bất thường.
3. **Bản đồ nhiệt (Grad-CAM Heatmap)**: Khoanh vùng vị trí tổn thương nghi ngờ phủ trên ảnh gốc để bác sĩ đối chiếu chẩn đoán.

---

## 4. Danh Sách Kỹ Năng Đã Cài Đặt (Installed Skills)
- `chexspark-pipeline`: Đặc tả chi tiết về kiến trúc Data Lakehouse, Spark Core/SQL/GraphX/MLlib, pipeline học sâu 3 mô hình (U-Net, DenseNet121, Grad-CAM), và cấu hình MLOps.
- `github-automation`: Tự động hóa Git và GitHub CLI (`gh`), quy chuẩn commit conventions, tạo và đồng bộ repository.

---

## 5. Cấu Hình MCP Servers
- `github`: Tích hợp GitHub API qua `@modelcontextprotocol/server-github`.
- `fetch`: HTTP fetch lấy dữ liệu qua `@modelcontextprotocol/server-fetch`.
- `filesystem`: Truy cập và thao tác tệp tin trên thư mục cục bộ `d:\BigData\CheXSpark`.

---

## 6. Lệnh Vận Hành OpenCode
Chạy lệnh `opencode` trực tiếp tại thư mục này để tương tác với agent `chexspark-engineer`:
```powershell
opencode
```
Agent sẽ tự động nhận diện agent cấu hình mặc định `chexspark-engineer` và nạp các kỹ năng `chexspark-pipeline` cùng `github-automation`.
