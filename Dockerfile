# DamageScan serving image: runs the FastAPI prediction API.
# We install the CPU build of torch to keep the image small — the API uses
# onnxruntime for inference, and torch is only pulled in as a package dependency.
FROM python:3.11-slim

# Avoid interactive prompts and keep Python output unbuffered for logs.
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /srv

# System libs needed by Pillow / OpenCV-style image decoding.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libjpeg62-turbo libpng16-16 \
    && rm -rf /var/lib/apt/lists/*

# Install CPU-only torch first (smaller), then the project + serving extras.
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision \
    && pip install ".[serve]"

# Copy the app code and (optionally) a baked-in model.
COPY app ./app
# Models are git-ignored; mount them at runtime, e.g.
#   docker run -p 8000:8000 -v "$PWD/models:/srv/models" damagescan
ENV MODEL_PATH=/srv/models/resnet18.onnx

EXPOSE 8000
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
