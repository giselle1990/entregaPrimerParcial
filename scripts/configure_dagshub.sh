#!/usr/bin/env bash
set -euo pipefail

: "${DAGSHUB_DVC_URL:?Definir DAGSHUB_DVC_URL, por ejemplo https://dagshub.com/usuario/repositorio.dvc}"

dvc remote remove dagshub 2>/dev/null || true
dvc remote add -d dagshub "$DAGSHUB_DVC_URL"

echo "Remote DVC 'dagshub' configurado en .dvc/config."
echo "Las credenciales deben configurarse fuera de Git, por ejemplo en .dvc/config.local o variables de entorno."
