#!/usr/bin/env bash
# Run the UI against shrub's live DB. Uses the shrub-app image for its Python
# deps (fastapi, aiomysql, numpy) so milton needs no image of its own yet.
set -euo pipefail
cd "$(dirname "$0")/.."
SHRUB=../shrub
exec docker run --rm --network shrub_default -p 8200:8200 \
  -v "$PWD:/w" -w /w -e PYTHONPATH=/w \
  -e MILTON_DB_HOST=db -e MILTON_DB_PORT=3306 \
  -e MILTON_DB_USER="$(grep '^MYSQL_USER=' $SHRUB/.env | cut -d= -f2)" \
  -e MILTON_DB_PASSWORD="$(grep '^MYSQL_PASSWORD=' $SHRUB/.env | cut -d= -f2)" \
  -e MILTON_DB_NAME="$(grep '^MYSQL_DATABASE=' $SHRUB/.env | cut -d= -f2)" \
  --entrypoint python shrub-app -m uvicorn milton.web:app --host 0.0.0.0 --port 8200
