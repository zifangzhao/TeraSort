#!/usr/bin/env bash
set -Eeuo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
python_arg=""
if [[ "${1:-}" == "--python" ]]; then
    if [[ $# -lt 2 ]]; then
        echo "Usage: $0 [--python /path/to/python] [dashboard options...]" >&2
        exit 2
    fi
    python_arg="$2"
    shift 2
elif [[ -n "${1:-}" && "${1:0:1}" != "-" ]]; then
    python_arg="$1"
    shift
fi

if [[ -n "$python_arg" ]]; then
    bash "$repo_root/scripts/install-linux.sh" "$python_arg"
else
    bash "$repo_root/scripts/install-linux.sh"
fi
exec "$repo_root/start_server.sh" "$@"
