#!/usr/bin/env bash
# smoke_test/run_smoke.sh
# Author: Ou Deng on Oct 2025
# Minimal smoke test wrapper for LGO repo (conda-based)
# - robust repo detection / clone (with NO_CLONE switch)
# - creates/activates conda env from env_setup/env_py310.yml (env name: py310_test)
# - runs run_v3_8.py for ICU (example)
# - runs threshold generation/audit & plotting scripts
#
# Usage:
#   # normal (if you're in repo root)
#   bash smoke_test/run_smoke.sh
#
#   # run from parent directory and allow clone
#   bash smoke_test/run_smoke.sh
#
#   # run from parent directory but forbid any network action (no clone/pull)
#   NO_CLONE=1 bash smoke_test/run_smoke.sh
set -euo pipefail

##########################
# Configuration - edit if needed
##########################
REPO_URL="https://github.com/oudeng/LGO.git"
REPO_DIR="LGO"                     # target clone dir (if not already in repo)
CONDA_ENV_NAME="py310_test"
ENV_YML_PATH="env_setup/env_py310.yml"
SMOKE_OUTDIR="smoke_test"
ICU_CSV="data/ICU/ICU_composite_risk_score.csv"
ICU_DATASET_NAME="ICU_composite_risk_score"
SEEDS="1,2,3"
TEST_SIZE="0.2"
EXPERIMENTS="lgo_soft,lgo_hard"

##########################
# Helpers
##########################
err() { echo "ERROR: $*" >&2; exit 1; }
info() { echo "--> $*"; }

# --------- robust repo detection / clone logic ------------
# If script is run from inside existing repo (i.e. run_v3_8.py exists), use current dir
# You can force skipping any network action by exporting NO_CLONE=1 before running:
#   NO_CLONE=1 bash smoke_test/run_smoke.sh
if [ -f "run_v3_8.py" ]; then
  info "Detected run_v3_8.py in current directory. Using current directory as repo root."
  ROOT_DIR="$(pwd)"
else
  # If a local LGO folder exists...
  if [ -d "${REPO_DIR}" ]; then
    # If it contains run_v3_8.py, reuse it
    if [ -f "${REPO_DIR}/run_v3_8.py" ]; then
      info "Found existing ${REPO_DIR} with run_v3_8.py. Using it."
      ROOT_DIR="$(cd "${REPO_DIR}" && pwd)"
    else
      # REPO_DIR exists but looks incomplete. If NO_CLONE set, fail early; otherwise try to update or reclone.
      if [ "${NO_CLONE:-0}" = "1" ]; then
        err "Directory ${REPO_DIR} exists but is missing expected files and NO_CLONE=1 is set. Please fix repository or run from repo root."
      fi
      info "Directory ${REPO_DIR} exists but seems incomplete. Attempting to update via git pull..."
      if [ -d "${REPO_DIR}/.git" ]; then
        git -C "${REPO_DIR}" pull --ff-only || {
          info "git pull failed; removing ${REPO_DIR} and re-cloning..."
          rm -rf "${REPO_DIR}"
          git clone --depth 1 "${REPO_URL}" "${REPO_DIR}" || err "git clone failed"
        }
      else
        info "${REPO_DIR} is not a git repo. Removing and re-cloning..."
        rm -rf "${REPO_DIR}"
        git clone --depth 1 "${REPO_URL}" "${REPO_DIR}" || err "git clone failed"
      fi
      ROOT_DIR="$(cd "${REPO_DIR}" && pwd)"
    fi
  else
    # REPO_DIR does not exist
    if [ "${NO_CLONE:-0}" = "1" ]; then
      err "Repository ${REPO_DIR} not found and NO_CLONE=1 set. Please clone the repo first or unset NO_CLONE."
    fi
    info "Cloning repository ${REPO_URL} into ${REPO_DIR}..."
    git clone --depth 1 "${REPO_URL}" "${REPO_DIR}" || err "git clone failed"
    ROOT_DIR="$(cd "${REPO_DIR}" && pwd)"
  fi
fi

info "Repository root: ${ROOT_DIR}"
# change to repository root
cd "${ROOT_DIR}"
# ---------------- end robust logic -------------------------

##########################
# Ensure conda available
##########################
if ! command -v conda >/dev/null 2>&1; then
  err "conda not found in PATH. Please install Miniconda/Anaconda and ensure 'conda' is available."
fi

# source conda.sh to allow 'conda activate' in scripts
CONDA_BASE="$(conda info --base 2>/dev/null || true)"
if [ -z "${CONDA_BASE}" ]; then
  err "Unable to determine conda base. Ensure conda is initialized for this shell."
fi
# shellcheck source=/dev/null
source "${CONDA_BASE}/etc/profile.d/conda.sh"

##########################
# Create/activate conda env
##########################
if conda env list | awk '{print $1}' | grep -qx "${CONDA_ENV_NAME}"; then
  info "Conda environment '${CONDA_ENV_NAME}' already exists. Skipping create."
else
  if [ ! -f "${ENV_YML_PATH}" ]; then
    err "Environment file ${ENV_YML_PATH} not found in repo. Please check path."
  fi
  info "Creating conda environment '${CONDA_ENV_NAME}' from ${ENV_YML_PATH} (this may take a few minutes)..."
  conda env create -f "${ENV_YML_PATH}" || err "Failed to create conda env from ${ENV_YML_PATH}"
fi

info "Activating conda environment '${CONDA_ENV_NAME}'..."
# Temporarily disable 'unbound variable' checking to avoid issues with conda's activate/deactivate hooks
set +u
conda activate "${CONDA_ENV_NAME}" || err "Failed to activate conda env ${CONDA_ENV_NAME}"
set -u

# show python version for debug
info "Using python: $(which python) ($(python --version 2>&1))"

##########################
# Prepare smoke directories
##########################
mkdir -p "${ROOT_DIR}/${SMOKE_OUTDIR}"
ICU_OUTDIR="${ROOT_DIR}/${SMOKE_OUTDIR}/ICU"
mkdir -p "${ICU_OUTDIR}"
FIG_OUTDIR="${ROOT_DIR}/${SMOKE_OUTDIR}/fig"
mkdir -p "${FIG_OUTDIR}"

##########################
# 1) Run main SR script for ICU
##########################
info "Running symbolic regression (ICU) with run_v3_8.py ..."
if [ ! -f "${ICU_CSV}" ]; then
  err "ICU CSV not found at ${ICU_CSV}. Please ensure the data file is present (or update ICU_CSV path)."
fi

python run_v3_8.py \
  --csv "${ICU_CSV}" \
  --target composite_risk_score \
  --task regression \
  --experiments "${EXPERIMENTS}" \
  --seeds "${SEEDS}" \
  --test_size "${TEST_SIZE}" \
  --outdir "${ICU_OUTDIR}" \
  --dataset "${ICU_DATASET_NAME}" \
  || err "run_v3_8.py failed for ICU"

info "SR finished for ICU. Outputs in ${ICU_OUTDIR}"

##########################
# 2) Threshold generation & audit (utility_analysis)
##########################
info "Generating thresholds in natural units for ICU (07_gen_thresholds_units.py)..."
python utility_analysis/07_gen_thresholds_units.py \
  --dataset_dir "${ICU_OUTDIR}" \
  --dataset "${ICU_DATASET_NAME}" \
  --method lgo --topk 10 --experiments lgo_hard \
  || err "07_gen_thresholds_units.py failed"

info "Running threshold audit (08_threshold_audit.py) for ICU..."
python utility_analysis/08_threshold_audit.py \
  --dataset_dir "${ICU_OUTDIR}" \
  --dataset "${ICU_DATASET_NAME}" \
  --guidelines config/guidelines.yaml \
  || err "08_threshold_audit.py failed"

info "Threshold generation & audit completed. Audit outputs should be in ${ICU_OUTDIR}"

##########################
# 3) Visualization - produce Figure 2 like outputs
##########################
info "Producing threshold visualizations (04_thresholds_v3_7.py --> 04_thresholds_plot_r1.py)..."

# 04_thresholds_v3_7.py expects an 'overall_*' dataset_dirs argument (script-specific).
# We assume that the threshold audit produced an overall_* folder or CSVs; adapt if needed.
python utility_plots/04_thresholds_v3_7.py \
  --dataset_dirs "overall_${ICU_DATASET_NAME}" \
  --config_dir config \
  --method lgo --experiment lgo_hard \
  --only_anchored \
  --topk 5 \
  --annotate \
  --outdir "${FIG_OUTDIR}" \
  || err "04_thresholds_v3_7.py failed"

python utility_plots/04_thresholds_plot_r1.py \
  --csv "${FIG_OUTDIR}/v3_thresholds_summary.csv" \
  --outdir "${FIG_OUTDIR}" \
  || err "04_thresholds_plot_r1.py failed"

info "Visualizations generated in ${FIG_OUTDIR} (e.g. v3_thresholds_summary.csv and plot files)."

##########################
# OPTIONAL: NHANES run template (uncomment & edit csv path if you want to run NHANES)
##########################
: <<'NHANES_TEMPLATE'
# Example NHANES run - uncomment and set correct CSV path to run
NHANES_CSV="data/NHANES/NHANES2_processed.csv"
NHANES_OUTDIR="${ROOT_DIR}/${SMOKE_OUTDIR}/NHANES"
mkdir -p "${NHANES_OUTDIR}"

python run_v3_8.py \
  --csv "${NHANES_CSV}" \
  --target <your_target_column> \
  --task regression \
  --experiments "${EXPERIMENTS}" \
  --seeds "${SEEDS}" \
  --test_size "${TEST_SIZE}" \
  --outdir "${NHANES_OUTDIR}" \
  --dataset "NHANES_dataset_name"
NHANES_TEMPLATE

##########################
# Wrap up
##########################
info "Smoke test completed successfully."
info "Summary of outputs:"
info "  ICU SR outputs: ${ICU_OUTDIR}"
info "  Figures/plots: ${FIG_OUTDIR}"
info ""
info "If you will push the smoke_test directory to GitHub, ensure smoke_test/run_smoke.sh is executable: chmod +x smoke_test/run_smoke.sh"
info "If any script fails, inspect the log printed above and confirm paths & dependencies."

# deactivate env (optional)
set +u
conda deactivate || true
set -u