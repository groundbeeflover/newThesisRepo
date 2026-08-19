#!/usr/bin/env bash
set -euo pipefail

#where it all goes
DATA_ROOT="${1:-./data}"
GWHD_DIR="${DATA_ROOT}/gwhd_2021"
ZIP_PATH="${DATA_ROOT}/gwhd_2021.zip"

mkdir -p "${DATA_ROOT}"

echo "Downloading GWHD 2021..."
#the official archive on zenodo
curl -L \
  "https://zenodo.org/records/5092309/files/gwhd_2021.zip?download=1" \
  -o "${ZIP_PATH}"

echo "Extracting GWHD 2021..."
mkdir -p "${GWHD_DIR}"
unzip -q "${ZIP_PATH}" -d "${GWHD_DIR}"

#some of these unzip into a nested folder, flatten it if that happened
if [ -d "${GWHD_DIR}/gwhd_2021" ]; then
  shopt -s dotglob nullglob
  mv "${GWHD_DIR}/gwhd_2021"/* "${GWHD_DIR}/"
  rmdir "${GWHD_DIR}/gwhd_2021"
fi

echo "Checking expected files..."
test -d "${GWHD_DIR}/images"
test -f "${GWHD_DIR}/competition_train.csv"
test -f "${GWHD_DIR}/competition_val.csv"
test -f "${GWHD_DIR}/competition_test.csv"

echo "GWHD dataset installed successfully at: ${GWHD_DIR}"
echo
echo "Expected structure:"
echo "${GWHD_DIR}/images"
echo "${GWHD_DIR}/competition_train.csv"
echo "${GWHD_DIR}/competition_val.csv"
echo "${GWHD_DIR}/competition_test.csv"