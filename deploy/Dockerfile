# Online demo (Cloud Run). The build context is the app/ folder made by scripts/build_app.py; see deploy/README.md.
FROM python:3.12.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/opt/hf \
    TORCH_HOME=/opt/torch

WORKDIR /app

# Dependencies first (CPU-only torch keeps the image small), code and data last.
# Gradio pins "pillow<13" but every 12.x release has known CVEs, so Pillow 13 is forced after the install.
COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip setuptools \
    && pip install --no-cache-dir torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir --no-deps pillow==13.0.0 \
    && useradd --create-home --uid 10001 guide

# SigLIP and the ImageNet ResNet-18 (frozen blocks 1-3) are fetched at build time: a cold start needs no network.
RUN python -c "from transformers import AutoModel, AutoProcessor; m='google/siglip-base-patch16-224'; AutoProcessor.from_pretrained(m); AutoModel.from_pretrained(m)" \
    && python -c "from torchvision.models import ResNet18_Weights, resnet18; resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)" \
    && chmod -R a+rX /opt/hf /opt/torch
ENV HF_HUB_OFFLINE=1

COPY . .

USER guide
EXPOSE 8080
# Cloud Run passes the port in $PORT
CMD ["sh", "-c", "exec python app.py --port ${PORT:-8080}"]
