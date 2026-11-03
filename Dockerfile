# Cloud Run image for scripts/serve.py: code + CPU torch only; model state (~5.5 GB) is pulled from $BUNDLE at start.
FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
# pins match the matt-pc venv that built the caches; no torchvision, so CLIP preprocessing stays on the PIL backend as in ingest
RUN pip install --no-cache-dir torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir transformers==5.17.0 huggingface-hub==1.33.0 lightgbm==4.7.0 pillow==12.3.0 numpy==2.4.6 \
    scipy==1.17.1 pandas==3.0.6 tqdm==4.70.1 google-cloud-storage==3.15.0
COPY scripts/ /app/scripts/
WORKDIR /app/scripts
ENV BACKBONE_DIR=/app/data/backbones HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
CMD ["python", "serve.py", "--host", "0.0.0.0"]
