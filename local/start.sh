#!/usr/bin/env bash
# Voiceduck (Voice-Gender-Analyzer) local launcher for this machine.
#
# Deviations from upstream README, all deliberate:
#   * Redis and the Engine C sidecar run as rootless podman containers (no docker here).
#   * Backend listens on 8090 because Steam's webhelper squats on 127.0.0.1:8080.
#     web/vite.config.js was patched to proxy /api to $BACKEND_DEV_PORT.
#   * .env points REDIS_URL / ENGINE_C_SIDECAR_URL at 127.0.0.1 (bare-metal, not compose).
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
export BACKEND_DEV_PORT="${BACKEND_DEV_PORT:-8090}"
export TF_CPP_MIN_LOG_LEVEL=3

for c in voiceya-redis voiceya-engine-c; do
  if ! podman container exists "$c"; then
    echo "!! container $c missing; see local/README.md to recreate it" >&2; exit 1
  fi
  podman start "$c" >/dev/null 2>&1 || true
done
echo "waiting for redis + engine-c sidecar…"
until podman exec voiceya-redis redis-cli ping 2>/dev/null | grep -q PONG; do sleep 1; done
for _ in $(seq 1 60); do
  curl -fsS http://127.0.0.1:8001/healthz >/dev/null 2>&1 && break; sleep 2
done
curl -fsS http://127.0.0.1:8001/healthz || echo "!! engine-c sidecar not healthy yet; Engine C results will be null until it is"
echo
cd "$REPO"
exec .venv/bin/python run_app.py
