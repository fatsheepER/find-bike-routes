#!/usr/bin/env bash
set -Eeuo pipefail

: "${BIKE_ROUTES_IMPORT_PASSWORD:?set BIKE_ROUTES_IMPORT_PASSWORD}"
: "${BIKE_ROUTES_API_PASSWORD:?set BIKE_ROUTES_API_PASSWORD}"

psql -v ON_ERROR_STOP=1 \
    --username "$POSTGRES_USER" \
    --dbname "$POSTGRES_DB" <<'SQL'
\getenv import_password BIKE_ROUTES_IMPORT_PASSWORD
\getenv api_password BIKE_ROUTES_API_PASSWORD
ALTER ROLE bike_routes_import PASSWORD :'import_password';
ALTER ROLE bike_routes_api PASSWORD :'api_password';
SQL
