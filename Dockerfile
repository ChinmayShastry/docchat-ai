# Streamlit app image.
#
# Reranking is off by default, and sentence-transformers is imported lazily, so
# neither it nor torch is installed here — that keeps the image roughly 800 MB
# smaller for a feature the measurements in eval/results/ showed does not help.
#
# To build an image with reranking available, add both lines below to the pip
# install step and set DOCCHAT_RETRIEVAL_MODE=hybrid_rerank at runtime:
#
#     -r requirements-rerank.txt
#     && python -c "from sentence_transformers import CrossEncoder; \
#        CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"
#
# Prefetching the weights at build time matters there, or the first request to
# a cold container waits on a model download.

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
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
