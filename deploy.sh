#!/usr/bin/env bash
# =============================================================================
# Altcoin Agent - One-click Deploy (Linux / macOS / WSL)
# =============================================================================
# Usage:
#   chmod +x deploy.sh && ./deploy.sh
#
# What it does:
#   1. Checks Docker / docker compose
#   2. Auto-detects China network and switches pip + Docker mirrors
#   3. Generates .env from template if missing, prompts for required keys
#   4. Builds image with retry on transient network errors
#   5. Starts service, waits for /healthz, prints status
#
# Re-runnable safely. Will not overwrite an existing .env.
# =============================================================================

set -Eeuo pipefail

# ---- pretty output ----------------------------------------------------------
RED=$'\033[0;31m'; GREEN=$'\033[0;32m'; YELLOW=$'\033[1;33m'; BLUE=$'\033[0;34m'; BOLD=$'\033[1m'; NC=$'\033[0m'
log()  { echo "${BLUE}[deploy]${NC} $*"; }
ok()   { echo "${GREEN}[ ok ]${NC}   $*"; }
warn() { echo "${YELLOW}[warn]${NC}   $*"; }
err()  { echo "${RED}[err ]${NC}   $*" >&2; }
step() { echo; echo "${BOLD}==> $*${NC}"; }

trap 'err "deploy.sh failed at line $LINENO"; exit 1' ERR

cd "$(dirname "$0")"
ROOT="$(pwd)"
log "working dir: $ROOT"

# =============================================================================
# Step 1: docker / compose check
# =============================================================================
step "1/6 Checking Docker"

if ! command -v docker >/dev/null 2>&1; then
    err "Docker not found. Install Docker Desktop first:"
    err "  https://www.docker.com/products/docker-desktop/"
    exit 1
fi

if ! docker info >/dev/null 2>&1; then
    err "Docker is installed but the daemon is not running."
    err "  - macOS/Windows: open Docker Desktop and wait until it says 'Running'"
    err "  - Linux:         sudo systemctl start docker"
    exit 1
fi
ok "docker daemon is running"

# Detect compose v2 (preferred) or v1
if docker compose version >/dev/null 2>&1; then
    COMPOSE="docker compose"
elif command -v docker-compose >/dev/null 2>&1; then
    COMPOSE="docker-compose"
else
    err "docker compose not found. Update Docker Desktop or install docker-compose-plugin."
    exit 1
fi
ok "compose: $COMPOSE"

# =============================================================================
# Step 2: detect China network and configure mirrors
# =============================================================================
step "2/6 Detecting network region (for mirror selection)"

IS_CN=0
# Use a 3-second timeout test. If GitHub is slow/blocked we assume CN-style network.
if curl -fsS --max-time 3 https://www.google.com >/dev/null 2>&1; then
    ok "international network detected, using default registries"
elif curl -fsS --max-time 3 https://www.baidu.com >/dev/null 2>&1; then
    IS_CN=1
    warn "China-style network detected, will use mirrors:"
    warn "  - pip:    https://pypi.tuna.tsinghua.edu.cn/simple"
    warn "  - docker: registry.docker-cn.com / dockerproxy.com"
else
    warn "network test inconclusive, using default registries"
fi

export DEPLOY_PIP_INDEX_URL=""
if [ "$IS_CN" = "1" ]; then
    export DEPLOY_PIP_INDEX_URL="https://pypi.tuna.tsinghua.edu.cn/simple"
fi

# =============================================================================
# Step 3: prepare .env
# =============================================================================
step "3/6 Preparing .env"

if [ ! -f .env ]; then
    if [ ! -f .env.example ]; then
        err ".env.example missing — repository state is broken."
        exit 1
    fi
    cp .env.example .env
    ok "created .env from template"

    # Auto-generate dashboard token (32 hex chars)
    TOKEN="$(openssl rand -hex 16 2>/dev/null || head -c 32 /dev/urandom | xxd -p -c 32)"
    if grep -q '^DASHBOARD_TOKEN=' .env; then
        # in-place replace, portable across BSD/GNU sed
        if [[ "$OSTYPE" == "darwin"* ]]; then
            sed -i '' "s|^DASHBOARD_TOKEN=.*|DASHBOARD_TOKEN=${TOKEN}|" .env
        else
            sed -i "s|^DASHBOARD_TOKEN=.*|DASHBOARD_TOKEN=${TOKEN}|" .env
        fi
    fi
    ok "auto-generated DASHBOARD_TOKEN=${TOKEN}"
    echo "        ^ save this — needed to access the web dashboard"
else
    ok ".env already exists, keeping it"
fi

# =============================================================================
# Step 4: validate required keys
# =============================================================================
step "4/6 Validating required keys in .env"

# shellcheck disable=SC1091
set -a; source .env; set +a

MISSING=()

# LLM key check — at least one provider key must be set
LLM_KEY_SET=0
for var in DEEPSEEK_API_KEY OPENAI_API_KEY OPENROUTER_API_KEY MOONSHOT_API_KEY DASHSCOPE_API_KEY ANTHROPIC_API_KEY LLM_API_KEY; do
    val="${!var:-}"
    if [ -n "$val" ]; then LLM_KEY_SET=1; break; fi
done

if [ "$LLM_KEY_SET" = "0" ]; then
    MISSING+=("an LLM API key (e.g. DEEPSEEK_API_KEY)")
fi

if [ -z "${DASHBOARD_TOKEN:-}" ]; then
    MISSING+=("DASHBOARD_TOKEN")
fi

if [ -z "${BINANCE_API_KEY:-}" ] || [ -z "${BINANCE_API_SECRET:-}" ]; then
    warn "BINANCE_API_KEY / BINANCE_API_SECRET empty"
    warn "  the agent will run, but cannot place orders or fetch balances"
    warn "  get testnet keys at: https://testnet.binancefuture.com/"
fi

if [ ${#MISSING[@]} -gt 0 ]; then
    err "The following are missing in .env:"
    for m in "${MISSING[@]}"; do err "  - $m"; done
    err ""
    err "Edit .env and re-run ./deploy.sh"
    err "  $ ${EDITOR:-nano} .env"
    exit 1
fi
ok "required keys present"

# Safety lock — refuse to deploy live mode without acknowledgement
if [ "${DRY_RUN:-true}" = "false" ] && [ "${PAPER_TRADE:-false}" = "false" ]; then
    if [ "${LIVE_CONFIRM:-}" != "I_UNDERSTAND" ]; then
        err "DRY_RUN=false + PAPER_TRADE=false => LIVE TRADING"
        err "  Set LIVE_CONFIRM=I_UNDERSTAND in .env to acknowledge."
        err "  STRONGLY recommended: keep DRY_RUN=true for the first 30 days."
        exit 1
    fi
    warn "live trading mode enabled — last chance to abort (Ctrl+C in 10s)"
    sleep 10
fi

# =============================================================================
# Step 5: build image (with retry)
# =============================================================================
step "5/6 Building Docker image"

mkdir -p logs .kiro/steering

BUILD_ARGS=()
if [ -n "$DEPLOY_PIP_INDEX_URL" ]; then
    BUILD_ARGS+=(--build-arg "PIP_INDEX_URL=$DEPLOY_PIP_INDEX_URL")
fi

build_attempt() {
    local n=$1
    log "build attempt $n/3 ..."
    if [ ${#BUILD_ARGS[@]} -gt 0 ]; then
        $COMPOSE build "${BUILD_ARGS[@]}"
    else
        $COMPOSE build
    fi
}

if ! build_attempt 1; then
    warn "build failed — retrying after 5 s"
    sleep 5
    if ! build_attempt 2; then
        warn "build failed again — retrying after 15 s with --no-cache"
        sleep 15
        if [ ${#BUILD_ARGS[@]} -gt 0 ]; then
            $COMPOSE build --no-cache "${BUILD_ARGS[@]}"
        else
            $COMPOSE build --no-cache
        fi
    fi
fi
ok "image built"

# =============================================================================
# Step 6: start + healthcheck
# =============================================================================
step "6/6 Starting service"

$COMPOSE up -d
ok "container started"

PORT="${HEALTHZ_PORT:-8080}"
log "waiting for /healthz on port $PORT (up to 90 s) ..."

HEALTHY=0
for i in $(seq 1 30); do
    if curl -fsS "http://127.0.0.1:${PORT}/healthz" >/dev/null 2>&1; then
        HEALTHY=1
        break
    fi
    sleep 3
done

echo
if [ "$HEALTHY" = "1" ]; then
    ok "agent is healthy"
    echo
    echo "${BOLD}========================================${NC}"
    echo "${GREEN}  Deploy successful${NC}"
    echo "${BOLD}========================================${NC}"
    echo "  Dashboard:  http://localhost:${PORT}/dashboard"
    echo "  Token:      ${DASHBOARD_TOKEN}"
    echo "  Mode:       DRY_RUN=${DRY_RUN:-true}  PAPER_TRADE=${PAPER_TRADE:-false}"
    echo
    echo "Common commands:"
    echo "  $COMPOSE logs -f          # tail logs"
    echo "  $COMPOSE ps               # status"
    echo "  $COMPOSE restart          # restart"
    echo "  $COMPOSE down             # stop"
    echo "  ./deploy.sh               # re-run after .env edit"
    echo
else
    err "service failed to become healthy in 90 s"
    err "showing last 80 log lines:"
    echo
    $COMPOSE logs --tail=80
    err ""
    err "Common causes:"
    err "  - Wrong API key (LLM or exchange)  -> edit .env and re-run"
    err "  - Network blocked to LLM endpoint  -> set PROXY_POOL in .env"
    err "  - Port $PORT already in use        -> change HEALTHZ_PORT in .env"
    exit 1
fi
