#!/usr/bin/env bash
# Pull the serving image(s) on this host, or on the given hosts over ssh.
#   scripts/pull_image.sh                                   # default image, local
#   scripts/pull_image.sh bodhanai-node001 bodhanai-node004 # default image on those hosts
# The legacy presets use local-only images: copy them with scripts/copy_image.sh.
set -euo pipefail
LIB_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${LIB_ROOT}/.venv/bin/python"; [[ -x "${PY}" ]] || PY=python3
IMAGES=$(PYTHONPATH="${LIB_ROOT}/src" "${PY}" -c "
import os
from inference_lib.registry import DEFAULT_IMAGE
print(os.environ.get('INFERENCE_IMAGE') or DEFAULT_IMAGE)")
pull() { for img in ${IMAGES}; do docker pull -q "${img}"; done; }
if [[ $# -eq 0 ]]; then pull; exit; fi
for h in "$@"; do ssh -o BatchMode=yes "${h}" "for img in ${IMAGES}; do docker pull -q \$img; done" | sed "s/^/${h}: /" & done
wait
