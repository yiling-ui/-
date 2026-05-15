FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# System deps for cryptography + healthcheck (curl) + tini (init).
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        curl \
        tini \
    && rm -rf /var/lib/apt/lists/*

# Copy package metadata + source first for layer caching.
COPY pyproject.toml ./
COPY README.md ./
COPY src ./src

RUN pip install --upgrade pip setuptools wheel && pip install -e '.'

# Copy auxiliary content. config/.kiro are also bind-mounted in compose,
# so the COPY here just gives the image a working default.
COPY config ./config
COPY .kiro ./.kiro

# Non-root runtime user.
RUN useradd --create-home --shell /bin/bash agent \
    && mkdir -p /app/logs \
    && chown -R agent:agent /app
USER agent

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8080/healthz || exit 1

ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["python", "-m", "altcoin_agent.main", "--config", "/app/config/app.yaml"]
