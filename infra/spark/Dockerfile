# ─────────────────────────────────────────────────────────────
# CheXSpark: Custom Spark 3.5.1 image
# Bổ sung S3A jars (đọc/ghi MinIO) + MLflow client (experiment tracking).
# Dùng cho cả spark-master lẫn spark-worker-1/2 trong docker-compose.yml.
# ─────────────────────────────────────────────────────────────
FROM apache/spark:3.5.1

USER root

# S3A filesystem + AWS SDK để Spark truy cập MinIO (S3-compatible).
# Jars pin theo version Hadoop 3.3.4 của Spark 3.5.1. Tải qua RUN (cache theo layer)
# để build self-contained — không cần commit jar (215MB) vào git.
RUN curl -fsSL -o /opt/spark/jars/hadoop-aws-3.3.4.jar \
      https://repo1.maven.org/maven2/org/apache/hadoop/hadoop-aws/3.3.4/hadoop-aws-3.3.4.jar \
 && curl -fsSL -o /opt/spark/jars/aws-java-sdk-bundle-1.11.1021.jar \
      https://repo1.maven.org/maven2/com/amazonaws/aws-java-sdk-bundle/1.11.1021/aws-java-sdk-bundle-1.11.1021.jar

# MLflow client (khớp version mlflow server v2.9.2 trong docker-compose).
# boto3: cần khi log artifact (model, mask) trực tiếp lên MinIO artifact root.
RUN pip3 install --no-cache-dir --upgrade pip \
    && pip3 install --no-cache-dir "mlflow==2.9.2" "boto3>=1.28" "psycopg2-binary>=2.9"

# PyTorch CPU + torchvision — cho benchmark "Single machine vs Spark cluster"
# (src/benchmark/spark_vs_local.py): suy luận / huấn luyện DL trên cluster CPU.
# Image chạy Python 3.8 → pin bản cao nhất còn hỗ trợ (wheel CPU-only ~190MB).
RUN pip3 install --no-cache-dir \
      torch==2.4.1 torchvision==0.19.1 \
      --index-url https://download.pytorch.org/whl/cpu

# apache/spark image mặc định chạy bằng user root trong compose; giữ nguyên.
