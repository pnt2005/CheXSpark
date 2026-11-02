# CheXSpark Airflow image — orchestration cho pipeline Medallion (Bronze → Silver → Gold)
#
# Vì sao cần image riêng:
#   - SparkSubmitOperator cần binary `spark-submit` + PySpark client để nộp job
#     lên cluster standalone (driver chạy trong container Airflow, executor trên worker)
#   - Image chính thức của Airflow **chặn pip chạy bằng root**, nên cài package
#     bằng user `airflow` (mặc định) → rơi vào ~/.local/bin
#
# ⚠ Python 3.8 (không phải 3.11): PySpark **bắt buộc driver và worker cùng minor
# version**. Image `apache/spark:3.5.1` dùng Python 3.8.10, nên Airflow cũng phải
# là 3.8 — nếu không sẽ gặp PYTHON_VERSION_MISMATCH khi executor chạy task Python.
FROM apache/airflow:2.9.3-python3.8

# PySpark cần JRE (image Airflow có sẵn java, nhưng kiểm tra cho chắc)
USER root
RUN if ! command -v java >/dev/null 2>&1; then \
        apt-get update && \
        apt-get install -y --no-install-recommends default-jre-headless && \
        rm -rf /var/lib/apt/lists/*; \
    fi

USER airflow

# PySpark 3.5.1 = đúng phiên bản cluster (chexspark-spark:3.5.1) để tránh lệch protocol
RUN pip install --no-cache-dir \
        "pyspark==3.5.1" \
        "apache-airflow-providers-apache-spark==4.8.1"

# spark-submit được pip đặt trong ~/.local/bin
# LƯU Ý: KHÔNG set PYSPARK_PYTHON — biến này bị Spark lan sang executor
# (spark.executorEnv.PYSPARK_PYTHON), mà executor chạy image cluster chỉ có
# /usr/bin/python3 → "Cannot run program /usr/local/bin/python".
# Driver dùng python3 của Airflow (đã cài pyspark), executor dùng python3 của
# image spark: để mặc định là khớp.
ENV PATH="/home/airflow/.local/bin:${PATH}" \
    SPARK_HOME="/home/airflow/.local/lib/python3.8/site-packages/pyspark"
