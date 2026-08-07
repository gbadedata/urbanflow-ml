# Multi-stage build. The builder installs dependencies into a virtualenv; the
# runtime copies only that environment and the application, so build tools and
# caches never reach the shipped image.

FROM python:3.12-slim AS builder

# uv is installed from PyPI rather than copied from a registry image. Pulling
# ghcr.io/astral-sh/uv requires a registry token that unauthenticated builds do
# not have, and the failure surfaces as an opaque "denied" during metadata
# resolution. The version is pinned so a release build cannot drift.
RUN pip install --no-cache-dir uv==0.12.2

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /app

# Dependencies are installed before the source is copied, so a code change does
# not invalidate the dependency layer.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project --no-dev

COPY src/ ./src/
COPY configs/ ./configs/
COPY data_contracts/ ./data_contracts/
COPY sql/ ./sql/
RUN uv sync --frozen --no-dev


FROM python:3.12-slim AS runtime

# Running as root would be the single easiest finding in any image scan.
RUN groupadd --system urbanflow \
 && useradd --system --gid urbanflow --create-home urbanflow

WORKDIR /app

COPY --from=builder --chown=urbanflow:urbanflow /app/.venv /app/.venv
COPY --from=builder --chown=urbanflow:urbanflow /app/src /app/src
COPY --from=builder --chown=urbanflow:urbanflow /app/configs /app/configs
COPY --from=builder --chown=urbanflow:urbanflow /app/data_contracts /app/data_contracts
COPY --from=builder --chown=urbanflow:urbanflow /app/sql /app/sql

# Model artefacts are baked in rather than mounted: the image then carries the
# exact model it was tested with, and a rollback is a single image tag.
COPY --chown=urbanflow:urbanflow models/ /app/models/

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MODEL_DIR=/app/models

USER urbanflow
EXPOSE 8000

# Hits /health rather than /ready: liveness should not fail because the database
# is briefly unreachable, or the orchestrator will restart a healthy process.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health', timeout=4).status == 200 else 1)"

CMD ["uvicorn", "urbanflow.serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
