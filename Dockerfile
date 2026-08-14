# Streamlit app image.
#
# The cross-encoder reranker is baked in at build time rather than downloaded
# on first request — otherwise the first user of a cold container waits on a
# ~90 MB model fetch before their first answer.

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    # Keep model + HF caches inside the app dir so they belong to the app user.
    HF_HOME=/app/.cache/huggingface

WORKDIR /app

# Build tooling needed by a few wheels; removed in the same layer to keep the
# image small.
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential curl \
    && rm -rf /var/lib/apt/lists/*

# Dependencies first, so source edits do not invalidate the install layer.
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt \
    && apt-get purge -y build-essential && apt-get autoremove -y

COPY . .

# Pre-download the reranker weights into the image.
RUN python -c "from sentence_transformers import CrossEncoder; \
    CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"

# Run as a non-root user.
RUN useradd --create-home --uid 1000 appuser \
    && mkdir -p /app/.docchat_cache /app/.cache \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD curl -fsS http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true"]
