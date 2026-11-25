FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN groupadd --gid 10001 aegis && \
    useradd --uid 10001 --gid aegis --shell /bin/false --create-home aegis

COPY pyproject.toml README.md LICENSE ./
COPY src ./src

RUN pip install --upgrade pip && \
    pip install .

USER aegis

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)" || exit 1

CMD ["uvicorn", "aegis_scale.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
