# =============================================================================
# Altcoin Agent - One-click Deploy (Windows PowerShell)
# =============================================================================
# Usage (in PowerShell, in the project folder):
#   Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
#   .\deploy.ps1
#
# What it does:
#   1. Checks Docker Desktop is running
#   2. Generates .env from template if missing, auto-creates DASHBOARD_TOKEN
#   3. Validates required API keys
#   4. Builds image with retry
#   5. Starts service, waits for /healthz, prints status
#
# Re-runnable safely. Will not overwrite an existing .env.
# =============================================================================

$ErrorActionPreference = "Stop"

function Log($msg)  { Write-Host "[deploy] $msg" -ForegroundColor Blue }
function Ok($msg)   { Write-Host "[ ok ]   $msg" -ForegroundColor Green }
function Warn($msg) { Write-Host "[warn]   $msg" -ForegroundColor Yellow }
function Err($msg)  { Write-Host "[err ]   $msg" -ForegroundColor Red }
function Step($msg) { Write-Host ""; Write-Host "==> $msg" -ForegroundColor White -BackgroundColor DarkBlue }

Set-Location -Path $PSScriptRoot
Log "working dir: $PSScriptRoot"

# =============================================================================
# Step 1: Docker check
# =============================================================================
Step "1/6 Checking Docker"

try {
    $null = docker --version
} catch {
    Err "Docker not found. Install Docker Desktop first:"
    Err "  https://www.docker.com/products/docker-desktop/"
    exit 1
}

try {
    $null = docker info 2>$null
    if ($LASTEXITCODE -ne 0) { throw "daemon not running" }
} catch {
    Err "Docker Desktop is not running."
    Err "  Open Docker Desktop and wait until the whale icon stops animating."
    exit 1
}
Ok "docker daemon is running"

try {
    $null = docker compose version 2>$null
    $COMPOSE_CMD = "docker compose"
} catch {
    Err "docker compose not found. Update Docker Desktop."
    exit 1
}
Ok "compose: $COMPOSE_CMD"

# =============================================================================
# Step 2: prepare .env
# =============================================================================
Step "2/6 Preparing .env"

if (-Not (Test-Path ".env")) {
    if (-Not (Test-Path ".env.example")) {
        Err ".env.example missing — repository state is broken."
        exit 1
    }
    Copy-Item ".env.example" ".env"
    Ok "created .env from template"

    # Auto-generate dashboard token
    $bytes = New-Object byte[] 16
    [System.Security.Cryptography.RandomNumberGenerator]::Create().GetBytes($bytes)
    $TOKEN = -join ($bytes | ForEach-Object { $_.ToString("x2") })
    (Get-Content ".env") -replace '^DASHBOARD_TOKEN=.*', "DASHBOARD_TOKEN=$TOKEN" | Set-Content ".env"
    Ok "auto-generated DASHBOARD_TOKEN=$TOKEN"
    Write-Host "        ^ save this — needed to access the web dashboard"
} else {
    Ok ".env already exists, keeping it"
}

# Read .env into a hashtable
$envVars = @{}
Get-Content ".env" | ForEach-Object {
    if ($_ -match '^\s*([A-Z_][A-Z0-9_]*)=(.*)$') {
        $envVars[$Matches[1]] = $Matches[2]
    }
}

# =============================================================================
# Step 3: validate required keys
# =============================================================================
Step "3/6 Validating required keys"

$missing = @()

$llmKeys = @("DEEPSEEK_API_KEY", "OPENAI_API_KEY", "OPENROUTER_API_KEY",
             "MOONSHOT_API_KEY", "DASHSCOPE_API_KEY", "ANTHROPIC_API_KEY", "LLM_API_KEY")
$llmSet = $false
foreach ($k in $llmKeys) {
    if ($envVars[$k] -and $envVars[$k].Trim() -ne "") { $llmSet = $true; break }
}
if (-not $llmSet) { $missing += "an LLM API key (e.g. DEEPSEEK_API_KEY)" }

if (-not $envVars["DASHBOARD_TOKEN"] -or $envVars["DASHBOARD_TOKEN"].Trim() -eq "") {
    $missing += "DASHBOARD_TOKEN"
}

if (-not $envVars["BINANCE_API_KEY"] -or -not $envVars["BINANCE_API_SECRET"]) {
    Warn "BINANCE_API_KEY / BINANCE_API_SECRET empty"
    Warn "  the agent will run, but cannot place orders or fetch balances"
    Warn "  get testnet keys at: https://testnet.binancefuture.com/"
}

if ($missing.Count -gt 0) {
    Err "The following are missing in .env:"
    foreach ($m in $missing) { Err "  - $m" }
    Err ""
    Err "Edit .env and re-run .\deploy.ps1"
    Err "  notepad .env"
    exit 1
}
Ok "required keys present"

# Live-mode safety lock
$dryRun = if ($envVars["DRY_RUN"]) { $envVars["DRY_RUN"] } else { "true" }
$paper  = if ($envVars["PAPER_TRADE"]) { $envVars["PAPER_TRADE"] } else { "false" }
if ($dryRun -eq "false" -and $paper -eq "false") {
    if ($envVars["LIVE_CONFIRM"] -ne "I_UNDERSTAND") {
        Err "DRY_RUN=false + PAPER_TRADE=false => LIVE TRADING"
        Err "  Set LIVE_CONFIRM=I_UNDERSTAND in .env to acknowledge."
        Err "  STRONGLY recommended: keep DRY_RUN=true for the first 30 days."
        exit 1
    }
    Warn "live trading mode enabled — last chance to abort (Ctrl+C in 10s)"
    Start-Sleep -Seconds 10
}

# =============================================================================
# Step 4: build (with retry)
# =============================================================================
Step "4/6 Building Docker image"

New-Item -ItemType Directory -Force -Path "logs" | Out-Null
New-Item -ItemType Directory -Force -Path ".kiro/steering" | Out-Null

function Invoke-Build($extraArgs) {
    $cmd = "$COMPOSE_CMD build $extraArgs"
    Log "running: $cmd"
    Invoke-Expression $cmd
    return $LASTEXITCODE
}

$rc = Invoke-Build ""
if ($rc -ne 0) {
    Warn "build failed — retrying after 5 s"
    Start-Sleep -Seconds 5
    $rc = Invoke-Build ""
}
if ($rc -ne 0) {
    Warn "build failed again — retrying with --no-cache"
    Start-Sleep -Seconds 15
    $rc = Invoke-Build "--no-cache"
}
if ($rc -ne 0) {
    Err "build failed after 3 attempts. Common causes:"
    Err "  - Network issue: check Docker Desktop -> Settings -> Resources -> Proxies"
    Err "  - Disk full:     docker system prune -a"
    exit 1
}
Ok "image built"

# =============================================================================
# Step 5: start
# =============================================================================
Step "5/6 Starting service"

Invoke-Expression "$COMPOSE_CMD up -d"
if ($LASTEXITCODE -ne 0) {
    Err "compose up failed"
    exit 1
}
Ok "container started"

# =============================================================================
# Step 6: healthcheck
# =============================================================================
Step "6/6 Waiting for /healthz"

$port = if ($envVars["HEALTHZ_PORT"]) { $envVars["HEALTHZ_PORT"] } else { "8080" }
$healthy = $false

for ($i = 1; $i -le 30; $i++) {
    try {
        $resp = Invoke-WebRequest -Uri "http://127.0.0.1:$port/healthz" -UseBasicParsing -TimeoutSec 3 2>$null
        if ($resp.StatusCode -eq 200) { $healthy = $true; break }
    } catch { }
    Start-Sleep -Seconds 3
}

Write-Host ""
if ($healthy) {
    Ok "agent is healthy"
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Green
    Write-Host "  Deploy successful" -ForegroundColor Green
    Write-Host "========================================" -ForegroundColor Green
    Write-Host "  Dashboard:  http://localhost:$port/dashboard"
    Write-Host "  Token:      $($envVars['DASHBOARD_TOKEN'])"
    Write-Host "  Mode:       DRY_RUN=$dryRun  PAPER_TRADE=$paper"
    Write-Host ""
    Write-Host "Common commands:"
    Write-Host "  $COMPOSE_CMD logs -f         # tail logs"
    Write-Host "  $COMPOSE_CMD ps              # status"
    Write-Host "  $COMPOSE_CMD restart         # restart"
    Write-Host "  $COMPOSE_CMD down            # stop"
    Write-Host "  .\deploy.ps1                 # re-run after .env edit"
    Write-Host ""
} else {
    Err "service failed to become healthy in 90 s"
    Err "showing last 80 log lines:"
    Write-Host ""
    Invoke-Expression "$COMPOSE_CMD logs --tail=80"
    Err ""
    Err "Common causes:"
    Err "  - Wrong API key (LLM or exchange)  -> edit .env and re-run"
    Err "  - Network blocked to LLM endpoint  -> set PROXY_POOL in .env"
    Err "  - Port $port already in use        -> change HEALTHZ_PORT in .env"
    exit 1
}
