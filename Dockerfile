FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=120

# -----------------------------------------------------------------------------
# Optional build args for restricted networks (China mirrors etc.).
# Pass via:
#   docker build --build-arg APT_MIRROR=mirrors.tuna.tsinghua.edu.cn \
#                --build-arg PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple .
# Defaults are upstream Debian + PyPI.
# -----------------------------------------------------------------------------
ARG APT_MIRROR=""
ARG PIP_INDEX_URL=""
ARG PIP_TRUSTED_HOST=""

WORKDIR /app

# Optionally rewrite apt sources to a faster mirror, then install deps with
# retries to survive transient network blips during build.
RUN set -eux; \
    if [ -n "$APT_MIRROR" ]; then \
        sed -i "s|deb.debian.org|${APT_MIRROR}|g; s|security.debian.org|${APT_MIRROR}|g" \
            /etc/apt/sources.list.d/debian.sources 2>/dev/null || \
        sed -i "s|deb.debian.org|${APT_MIRROR}|g; s|security.debian.org|${APT_MIRROR}|g" \
            /etc/apt/sources.list 2>/dev/null || true; \
    fi; \
    apt-get -o Acquire::Retries=3 update && \
    apt-get -o Acquire::Retries=3 install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        curl \
        tini \
    && rm -rf /var/lib/apt/lists/*

# Configure pip mirror if requested.
RUN set -eux; \
    if [ -n "$PIP_INDEX_URL" ]; then \
        pip config set global.index-url "$PIP_INDEX_URL"; \
        if [ -n "$PIP_TRUSTED_HOST" ]; then \
            pip config set global.trusted-host "$PIP_TRUSTED_HOST"; \
        else \
            host=$(echo "$PIP_INDEX_URL" | awk -F[/:] '{print $4}'); \
            [ -n "$host" ] && pip config set global.trusted-host "$host" || true; \
        fi; \
    fi

# Copy package metadata + source first for layer caching.
COPY pyproject.toml ./
COPY README.md ./
COPY src ./src

# pip install with up to 3 retries against transient errors.
RUN pip install --upgrade pip setuptools wheel && \
    ( pip install --retries 5 --timeout 120 -e '.' || \
      ( sleep 5 && pip install --retries 5 --timeout 120 -e '.' ) || \
      ( sleep 15 && pip install --retries 5 --timeout 120 -e '.' ) )

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
