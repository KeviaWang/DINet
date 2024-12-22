# 使用 Python 3.6.13-slim 作为基础镜像
FROM python:3.6.13-slim

# 更新镜像并安装必要的系统库（包括 ffmpeg 和其他依赖）
RUN apt-get update --fix-missing && apt-get install -y \
    ffmpeg \
    libsm6 \
    libxext6 \
    libx264-dev \
    && rm -rf /var/lib/apt/lists/*

# 设置工作目录
WORKDIR /app

# 将当前目录中的所有文件复制到容器的 /app 目录
COPY . /app

RUN pip install --upgrade pip
# 安装 PyTorch、TorchVision 和 Torchaudio
#RUN pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
RUN pip install torch==1.10.1+cu111 torchvision==0.11.2+cu111 torchaudio==0.10.1 -f https://download.pytorch.org/whl/cu111/torch_stable.html
# 安装项目依赖项
RUN pip install --no-cache-dir -r requirements.txt

# 设置 ENTRYPOINT 为 python
ENTRYPOINT ["python"]

# 设置默认命令为 run_evaluate.py --process_num 1
CMD ["run_evaluate.py"]
