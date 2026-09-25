#!/usr/bin/env bash
set -Eeuo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
python="$repo_root/.venv/bin/python"
if [[ ! -x "$python" ]]; then
    echo "TeraSort environment not found. Run ./install_and_start.sh first." >&2
    exit 1
fi
cd "$repo_root"
exec "$python" -u -m terasort.cli web "$@"
