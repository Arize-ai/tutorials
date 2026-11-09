#!/usr/bin/env bash
# Run the collector without Docker. Downloads the binary into bin/ on first use.
#   scripts/run-collector.sh                 # Dynatrace distribution
#   DISTRO=contrib scripts/run-collector.sh  # otelcol-contrib
#   DEBUG=1 scripts/run-collector.sh         # also print every exported span
set -euo pipefail
cd "$(dirname "$0")/.."

DISTRO="${DISTRO:-dynatrace}"
DT_VERSION="${DT_VERSION:-0.60.0}"
CONTRIB_VERSION="${CONTRIB_VERSION:-0.165.0}"

os="$(uname -s | tr '[:upper:]' '[:lower:]')"
arch="$(uname -m)"
case "$arch" in x86_64|amd64) arch=amd64 ;; arm64|aarch64) arch=arm64 ;; esac

mkdir -p bin
if [[ "$DISTRO" == "contrib" ]]; then
  bin="bin/otelcol-contrib"
  url="https://github.com/open-telemetry/opentelemetry-collector-releases/releases/download/v${CONTRIB_VERSION}/otelcol-contrib_${CONTRIB_VERSION}_${os}_${arch}.tar.gz"
  member="otelcol-contrib"
else
  bin="bin/dynatrace-otel-collector"
  dt_os="$(uname -s)"                        # Darwin / Linux
  dt_arch="$([[ "$arch" == amd64 ]] && echo x86_64 || echo arm64)"
  url="https://github.com/Dynatrace/dynatrace-otel-collector/releases/download/v${DT_VERSION}/dynatrace-otel-collector_${DT_VERSION}_${dt_os}_${dt_arch}.tar.gz"
  member="dynatrace-otel-collector"
fi

if [[ ! -x "$bin" ]]; then
  echo "downloading $url"
  curl -fsSL "$url" | tar -xz -C bin "$member"
  xattr -d com.apple.quarantine "$bin" 2>/dev/null || true
fi

[[ -f .env ]] || { echo "missing .env (copy .env.example)"; exit 1; }
set -a; source .env; set +a

configs=(--config collector/config.yaml)
[[ -n "${DEBUG:-}" ]] && configs+=(--config collector/debug.yaml)

"$bin" validate "${configs[@]}"
exec "$bin" "${configs[@]}"
