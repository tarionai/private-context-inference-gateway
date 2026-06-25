# syntax=docker/dockerfile:1

FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Third-party deps first so this layer is cached across source edits. These mirror
# pyproject's [project.dependencies] + the [runtime] extra exactly:
#   fastapi, pydantic -> core;  uvicorn -> serves gateway.app;  openai -> self_hosted route.
# Hosted Claude routes are opt-in: add "anthropic>=0.40" below and set ANTHROPIC_API_KEY.
RUN pip install \
    "fastapi>=0.110" \
    "pydantic>=2.0" \
    "uvicorn>=0.29" \
    "openai>=1.0"

# Then the package itself (deps already satisfied above).
COPY . .
RUN pip install --no-deps .

# Non-root user + writable audit volume.
RUN useradd --create-home --uid 10001 appuser \
    && mkdir -p /app/state \
    && chown -R appuser:appuser /app/state
USER appuser

ENV AUDIT_PATH=/app/state/gateway_audit.jsonl \
    GATEWAY_OFFLINE=0

EXPOSE 8000

HEALTHCHECK --interval=15s --timeout=3s --start-period=10s --retries=5 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/healthz', timeout=2)" || exit 1

CMD ["uvicorn", "gateway.app:app", "--host", "0.0.0.0", "--port", "8000"]
