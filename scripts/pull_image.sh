#!/usr/bin/env bash
# Pull the serving image(s) on this host, or on the given hosts over ssh.
#   scripts/pull_image.sh                                   # default image, local
#   scripts/pull_image.sh bodhanai-node001 bodhanai-node004 # default image on those hosts
#   LEGACY=1 scripts/pull_image.sh ...                      # also the legacy GLM image
# Images are private on GHCR: `docker login ghcr.io` once (shared home = all nodes).
set -euo pipefail
LIB_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${LIB_ROOT}/.venv/bin/python"; [[ -x "${PY}" ]] || PY=python3
IMAGES=$(PYTHONPATH="${LIB_ROOT}/src" "${PY}" -c "
import os
from inference_lib.registry import DEFAULT_IMAGE, REGISTRY
imgs = [os.environ.get('INFERENCE_IMAGE') or DEFAULT_IMAGE]
if os.environ.get('LEGACY') == '1':
    imgs.append(REGISTRY['glm-5.3-flash-legacy'].image)
print(' '.join(imgs))")
pull() { for img in ${IMAGES}; do docker pull -q "${img}"; done; }
if [[ $# -eq 0 ]]; then pull; exit; fi
for h in "$@"; do ssh -o BatchMode=yes "${h}" "for img in ${IMAGES}; do docker pull -q \$img; done" | sed "s/^/${h}: /" & done
wait
