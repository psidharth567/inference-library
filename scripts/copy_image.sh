#!/usr/bin/env bash
# Copy a local-only image (e.g. the legacy GLM / DSV4 images) from one node to others:
#   scripts/copy_image.sh toolkit/inference-glm53:12.8 bodhanai-node001 bodhanai-node021 bodhanai-node024
set -euo pipefail
IMG=$1 SRC=$2; shift 2
for DST in "$@"; do
  if ssh -o BatchMode=yes "${DST}" "docker image inspect ${IMG} >/dev/null 2>&1"; then echo "${DST}: already has ${IMG}"; continue; fi
  ssh -o BatchMode=yes "${SRC}" "docker save ${IMG}" | ssh -o BatchMode=yes "${DST}" "docker load" | sed "s/^/${DST}: /"
done
