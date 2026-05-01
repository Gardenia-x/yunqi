FROM python:3.11-slim

# 系统依赖：libsndfile（音频解码）、ffmpeg（格式转换）
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# 工作目录
WORKDIR /app

# 安装 Python 依赖
COPY web_app/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt -i https://mirrors.cloud.tencent.com/pypi/simple

# 复制代码
COPY src/ ./src/
COPY web_app/ ./web_app/

WORKDIR /app/web_app
EXPOSE 8080

# 启动 uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080", "--timeout-keep-alive", "180"]
