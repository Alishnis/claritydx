FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# System dependencies:
# - build-essential/gfortran: some scientific wheels have no prebuilt
#   binary for this platform/Python combo and must compile from source
# - libgl1/libglib2.0-0: required by OpenCV at import time
# - tesseract-ocr: OCR backend used by pytesseract for report parsing
# - libpq5: PostgreSQL client library used by psycopg2
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        gfortran \
        libgl1 \
        libglib2.0-0 \
        tesseract-ocr \
        libpq5 \
        curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download the pretrained torchvision backbones (DenseNet121, EfficientNet-B4)
# at build time so the image works fully offline at runtime and the first
# request isn't slowed down (or broken by a flaky network) by a cold download.
RUN python -c "\
from torchvision.models import densenet121, DenseNet121_Weights; \
from torchvision.models import efficientnet_b4, EfficientNet_B4_Weights; \
densenet121(weights=DenseNet121_Weights.DEFAULT); \
efficientnet_b4(weights=EfficientNet_B4_Weights.DEFAULT)"

COPY . .

RUN mkdir -p media staticfiles data/uploads blood_reports

EXPOSE 8000

COPY docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

ENTRYPOINT ["docker-entrypoint.sh"]
CMD ["gunicorn", "mysite.wsgi:application", "--bind", "0.0.0.0:8000", "--workers", "2", "--timeout", "180"]
