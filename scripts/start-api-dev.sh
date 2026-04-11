#!/bin/sh
set -e

if ! /app/.venv/bin/python -c "import importlib.util; raise SystemExit(0 if importlib.util.find_spec('lark_oapi') else 1)"; then
  echo "[api-dev] lark_oapi missing, installing..."
  /app/.venv/bin/python -m ensurepip --upgrade || true
  /app/.venv/bin/python -m pip install --no-cache-dir lark-oapi
else
  echo "[api-dev] lark_oapi already installed"
fi

exec /app/.venv/bin/uvicorn src.main:app --host 0.0.0.0 --port 8000 --workers 1 --loop asyncio --reload --reload-dir /app/src
