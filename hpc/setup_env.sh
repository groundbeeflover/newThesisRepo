#!/usr/bin/env bash
# One-time conda environment setup on the FSE CSlab HPC login node.
# Installs Miniconda into persistent storage (/data), symlinks it to
# ~/miniconda3 so it matches the path convention already used by
# run_gwhd_dg.sh / run_gwhd_baseline.sh, creates the DGOD env, and
# installs requirements.txt.
#
# Run from the root of newThesisRepo: bash hpc/setup_env.sh
set -euo pipefail

DATA_DIR="/i6356965"
CONDA_DIR="${DATA_DIR}/miniconda3"
ENV_NAME="DGOD"
PYTHON_VERSION="3.12"

mkdir -p "${DATA_DIR}"

if [ ! -d "${CONDA_DIR}" ]; then
  echo "Installing Miniconda into ${CONDA_DIR}..."
  curl -L https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o /tmp/miniconda.sh
  bash /tmp/miniconda.sh -b -p "${CONDA_DIR}"
  rm /tmp/miniconda.sh
else
  echo "Miniconda already installed at ${CONDA_DIR}, skipping install."
fi

# Symlink so ~/miniconda3 works too (matches the fallback path already
# hardcoded in run_gwhd_dg.sh / run_gwhd_baseline.sh).
if [ ! -e "${HOME}/miniconda3" ]; then
  ln -s "${CONDA_DIR}" "${HOME}/miniconda3"
fi

source "${CONDA_DIR}/etc/profile.d/conda.sh"

if ! conda env list | grep -q "^${ENV_NAME} "; then
  echo "Creating conda env '${ENV_NAME}' (python=${PYTHON_VERSION})..."
  conda create -n "${ENV_NAME}" "python=${PYTHON_VERSION}" -y
else
  echo "Conda env '${ENV_NAME}' already exists, skipping create."
fi

conda activate "${ENV_NAME}"

pip install --upgrade pip
pip install -r requirements.txt

echo "Checking CUDA availability (won't show a GPU unless run on a worker node via srun)..."
python - <<'EOF'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
EOF

echo ""
echo "Done. Add this to ~/.bashrc so conda is available in every new shell:"
echo "  echo 'source ~/miniconda3/etc/profile.d/conda.sh' >> ~/.bashrc"
echo ""
echo "Then activate any time with:"
echo "  conda activate ${ENV_NAME}"
