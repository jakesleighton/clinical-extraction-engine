# Local dev image. Orchestration is CPU-only; the model server (Ollama) runs
# outside the container and is reached over HTTP.
#
#   docker build -f containers/Dockerfile -t clinical-extraction-engine .
#   docker run --rm clinical-extraction-engine                      # runs the tests
#   docker run --rm -v "$PWD/data:/app/data" -v "$PWD/runs:/app/runs" \
#       clinical-extraction-engine python -m engine.extract data/... --pack packs/prostate_pathology --model qwen2.5:7b
#
# Build context is the repo root (the trailing "." above), not containers/.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    # host.docker.internal resolves to the host on Docker Desktop (Mac/Windows);
    # on Linux add --add-host=host.docker.internal:host-gateway
    OLLAMA_HOST=http://host.docker.internal:11434

WORKDIR /app

COPY pyproject.toml README.md ./
COPY engine ./engine
COPY evalharness ./evalharness
COPY adapters ./adapters
COPY scripts ./scripts
COPY packs ./packs
COPY tests ./tests

RUN pip install -e ".[dev]"

CMD ["pytest", "-q"]
