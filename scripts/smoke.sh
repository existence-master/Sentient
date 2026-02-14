#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
npm install --no-package-lock
if [ -f src/client/package.json ]; then
  (cd src/client && npm install && npm run build --if-present)
fi
echo "sentient smoke passed"
