#!/usr/bin/env bash
# =============================================================================
# LGO Smoke Test Script
# =============================================================================
# Quick reproducibility verification for reviewers and readers
# 
# Usage:
#   bash smoke_test/run_smoke_test.sh              # Default: eICU dataset
#   bash smoke_test/run_smoke_test.sh --dataset NHANES    # NHANES dataset
#   bash smoke_test/run_smoke_test.sh --quick          # Reduced parameters
#   SKIP_VIZ=1 bash smoke_test/run_smoke_test.sh       # Skip visualization
#
# Expected runtime: ~10 minutes on standard laptop
# =============================================================================

set -euo pipefail

# =============================================================================
# Configuration
# =============================================================================
CONDA_ENV_NAME="py310_smoke"

# Default dataset (NHANES is smaller and faster for smoke test)
DATASET="NHANES"
QUICK_MODE=0

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --dataset)
            DATASET="$2"
            shift 2
            ;;
        --quick)
            QUICK_MODE=1
            shift
            ;;
        --help|-h)
            echo "Usage: bash smoke_test/run_smoke_test.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --dataset NHANES|eICU  Dataset to test (default: NHANES)"
            echo "  --quick                Use reduced parameters for faster testing"
            echo "  --help                 Show this help message"
            echo ""
            echo "Environment variables:"
            echo "  SKIP_VIZ=1            Skip visualization step"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Dataset-specific configuration
case "$DATASET" in
    NHANES)
        CSV_PATH="data/NHANES/NHANES_metabolic_score.csv"
        TARGET="metabolic_score"
        DATASET_NAME="NHANES_metabolic_score"
        # Use existing guidelines from exp_results
        GUIDELINES_SOURCE="exp_results/overall_NHANES_metabolic_score/config/guidelines.yaml"
        UNIT_MAP='{"systolic_bp":"mmHg","triglycerides":"mg/dL","waist_circumference":"cm","fasting_glucose":"mg/dL","hdl_cholesterol":"mg/dL","age":"years"}'
        ;;
    eICU)
        CSV_PATH="data/eICU/eICU_composite_risk_score.csv"
        TARGET="composite_risk_score"
        DATASET_NAME="eICU_composite_risk_score"
        # Use existing guidelines from exp_results
        GUIDELINES_SOURCE="exp_results/overall_eICU_composite_risk_score/config/guidelines.yaml"
        UNIT_MAP='{"map_mmhg": "mmHg","sbp_min": "mmHg","dbp_min": "mmHg","lactate_mmol_l": "mmol/L","creatinine_mg_dl": "mg/dL","hemoglobin_min": "g/dL","sodium_min": "mmol/L","age_years": "years","hr_max": "bpm","resprate_max": "/min","spo2_min": "%","gcs": "","urine_output_min": "mL"}'
        ;;
    *)
        echo "ERROR: Unknown dataset '$DATASET'. Use NHANES or eICU."
        exit 1
        ;;
esac

# Hyperparameters
if [ "$QUICK_MODE" = "1" ]; then
    HPARAMS='{
        "gate_expr_enable": true,
        "pop_size": 200,
        "ngen": 30,
        "local_opt_steps": 50,
        "micro_mutation_prob": 0.2,
        "cv_proxy_weight": 0.15,
        "cv_proxy_weight_final": 0.3,
        "cv_proxy_warmup_frac": 0.7,
        "cv_proxy_subsample": 0.3,
        "cv_proxy_folds": 2,
        "typed_mode": "light",
        "typed_grouping": "none",
        "include_lgo_multi": true,
        "include_lgo_and3": false,
        "include_lgo_pair": false
    }'
    SEEDS="1,2"
    EXPERIMENTS="lgo_hard"
else
    HPARAMS='{
        "gate_expr_enable": true,
        "pop_size": 1000,
        "ngen": 100,
        "local_opt_steps": 150,
        "micro_mutation_prob": 0.2,
        "cv_proxy_weight": 0.15,
        "cv_proxy_weight_final": 0.3,
        "cv_proxy_warmup_frac": 0.7,
        "cv_proxy_subsample": 0.3,
        "cv_proxy_folds": 2,
        "typed_mode": "light",
        "typed_grouping": "none",
        "include_lgo_multi": true,
        "include_lgo_and3": true,
        "include_lgo_pair": false
    }'
    SEEDS="1,2,3"
    EXPERIMENTS="base,lgo_soft,lgo_hard"
fi

# =============================================================================
# Helper Functions
# =============================================================================
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

err() { echo -e "${RED}[ERROR]${NC} $*" >&2; exit 1; }
warn() { echo -e "${YELLOW}[WARN]${NC} $*" >&2; }
info() { echo -e "${BLUE}[INFO]${NC} $*"; }
success() { echo -e "${GREEN}[OK]${NC} $*"; }

print_header() {
    echo ""
    echo "============================================================"
    echo -e "${BLUE}$1${NC}"
    echo "============================================================"
}

# =============================================================================
# Step 0: Repository Detection
# =============================================================================
print_header "Step 0: Repository Setup"

# Must be run from repository root (where run_v3_8_2.py exists)
if [ ! -f "run_v3_8_2.py" ]; then
    err "run_v3_8_2.py not found. Please run this script from the LGO repository root:
    
    git clone https://github.com/oudeng/LGO.git && cd LGO
    bash smoke_test/run_smoke_test.sh"
fi

ROOT_DIR="$(pwd)"
success "Repository root: ${ROOT_DIR}"

# Set PYTHONPATH to include exp_engins directory (for lgo_v3 and other engine modules)
export PYTHONPATH="${ROOT_DIR}/exp_engins:${ROOT_DIR}:${PYTHONPATH:-}"

# Create output directories
RESULTS_DIR="${ROOT_DIR}/smoke_test/results"
DATASET_OUTDIR="${RESULTS_DIR}/${DATASET_NAME}"
FIG_OUTDIR="${RESULTS_DIR}/figs"
LOG_FILE="${RESULTS_DIR}/smoke_test.log"

mkdir -p "${DATASET_OUTDIR}" "${FIG_OUTDIR}"

# Start logging (append mode)
exec > >(tee -a "${LOG_FILE}") 2>&1

info "Logging to: ${LOG_FILE}"
echo "Smoke test started at $(date)"
echo "Dataset: ${DATASET}"
echo "Quick mode: ${QUICK_MODE}"

# =============================================================================
# Step 1: Conda Environment Setup
# =============================================================================
print_header "Step 1: Environment Setup"

if ! command -v conda >/dev/null 2>&1; then
    err "conda not found. Please install Miniconda: https://docs.conda.io/en/latest/miniconda.html"
fi

CONDA_BASE="$(conda info --base 2>/dev/null || true)"
if [ -z "${CONDA_BASE}" ]; then
    err "Unable to determine conda base directory."
fi

# shellcheck source=/dev/null
source "${CONDA_BASE}/etc/profile.d/conda.sh"

# Environment file options (in order of preference)
ENV_YML_OPTIONS=(
    "smoke_test/env_py310_smoke.yml"
    "env_setup/env_py310_smoke.yml"
    "env_setup/env_py310.yml"
)

# Check if environment exists
if conda env list | awk '{print $1}' | grep -qx "${CONDA_ENV_NAME}"; then
    info "Conda environment '${CONDA_ENV_NAME}' already exists."
else
    # Find available environment file
    YML_TO_USE=""
    for yml_path in "${ENV_YML_OPTIONS[@]}"; do
        if [ -f "${yml_path}" ]; then
            YML_TO_USE="${yml_path}"
            break
        fi
    done
    
    if [ -z "${YML_TO_USE}" ]; then
        err "No environment file found. Tried: ${ENV_YML_OPTIONS[*]}"
    fi
    
    info "Creating conda environment '${CONDA_ENV_NAME}' from ${YML_TO_USE} (this may take 2-3 minutes)..."
    conda env create -f "${YML_TO_USE}" -n "${CONDA_ENV_NAME}" || err "Failed to create conda environment"
fi

# Activate environment
info "Activating conda environment..."
set +u
conda activate "${CONDA_ENV_NAME}" || err "Failed to activate ${CONDA_ENV_NAME}"
set -u

success "Python: $(which python) ($(python --version 2>&1))"

# Quick dependency check
python -c "from deap import gp; print('[OK] DEAP imported')" || err "DEAP not available"
python -c "import pandas; print(f'[OK] Pandas {pandas.__version__}')" || err "Pandas not available"
python -c "import numpy; print(f'[OK] NumPy {numpy.__version__}')" || err "NumPy not available"

# Verify lgo_v3 is importable (should be in exp_engins/lgo_v3/)
info "Verifying lgo_v3 module..."
python -c "import lgo_v3; print('[OK] lgo_v3 module imported')" || err "lgo_v3 module not importable. Expected at: exp_engins/lgo_v3/"

# =============================================================================
# Step 2: Run LGO Symbolic Regression
# =============================================================================
print_header "Step 2: LGO Symbolic Regression (${DATASET})"

if [ ! -f "${CSV_PATH}" ]; then
    err "Dataset not found: ${CSV_PATH}"
fi

info "Running LGO with experiments: ${EXPERIMENTS}"
info "Seeds: ${SEEDS}"
info "This will take approximately 5-10 minutes..."

python run_v3_8_2.py \
    --csv "${CSV_PATH}" \
    --target "${TARGET}" \
    --task regression \
    --experiments "${EXPERIMENTS}" \
    --seeds "${SEEDS}" \
    --test_size 0.2 \
    --outdir "${DATASET_OUTDIR}" \
    --dataset "${DATASET_NAME}" \
    --save_predictions \
    --hparams_json "${HPARAMS}" \
    --unit_map_json "${UNIT_MAP}" \
    || err "run_v3_8_2.py failed"

success "LGO completed. Outputs in: ${DATASET_OUTDIR}"

# Quick validation
if [ -d "${DATASET_OUTDIR}/candidates" ]; then
    N_CANDIDATES=$(find "${DATASET_OUTDIR}/candidates" -name "*.csv" 2>/dev/null | wc -l)
    info "Found ${N_CANDIDATES} candidate files"
else
    warn "No candidates directory found"
fi

# =============================================================================
# Step 3: Threshold Extraction
# =============================================================================
print_header "Step 3: Threshold Extraction"

info "Extracting thresholds in physical units..."

# Check if utility_analysis scripts exist
if [ -f "utility_analysis/07_gen_thresholds_units.py" ]; then
    python utility_analysis/07_gen_thresholds_units.py \
        --dataset_dir "${DATASET_OUTDIR}" \
        --dataset "${DATASET_NAME}" \
        --method lgo \
        --topk 10 \
        --experiments lgo_hard \
        || warn "Threshold extraction had issues (may be OK if no lgo_hard gates)"

    if [ -f "${DATASET_OUTDIR}/aggregated/thresholds_units.csv" ]; then
        success "Thresholds extracted: ${DATASET_OUTDIR}/aggregated/thresholds_units.csv"
        echo ""
        echo "Sample thresholds:"
        head -5 "${DATASET_OUTDIR}/aggregated/thresholds_units.csv" 2>/dev/null || true
    else
        warn "thresholds_units.csv not created (may be OK for base experiment)"
    fi
else
    warn "utility_analysis/07_gen_thresholds_units.py not found, skipping threshold extraction"
fi

# =============================================================================
# Step 4: Threshold Audit
# =============================================================================
print_header "Step 4: Threshold Audit Against Clinical Guidelines"

# Use dataset-specific guidelines from exp_results (set in dataset config above)
# Fall back to config/guidelines.yaml if not found
GUIDELINES_PATH=""
for cfg_path in "${GUIDELINES_SOURCE}" "config/guidelines.yaml" "configs/guidelines.yaml"; do
    if [ -f "${cfg_path}" ]; then
        GUIDELINES_PATH="${cfg_path}"
        break
    fi
done

if [ -n "${GUIDELINES_PATH}" ]; then
    info "Using guidelines: ${GUIDELINES_PATH}"
    if [ -f "utility_analysis/08_threshold_audit.py" ]; then
        info "Auditing thresholds against clinical guidelines..."
        
        python utility_analysis/08_threshold_audit.py \
            --dataset_dir "${DATASET_OUTDIR}" \
            --dataset "${DATASET_NAME}" \
            --guidelines "${GUIDELINES_PATH}" \
            || warn "Threshold audit had issues"
        
        if [ -f "${DATASET_OUTDIR}/aggregated/threshold_audit.csv" ]; then
            success "Audit complete: ${DATASET_OUTDIR}/aggregated/threshold_audit.csv"
            
            # Show summary
            if [ -f "${DATASET_OUTDIR}/aggregated/threshold_audit_summary.csv" ]; then
                echo ""
                echo "Audit Summary:"
                cat "${DATASET_OUTDIR}/aggregated/threshold_audit_summary.csv"
            fi
        fi
    else
        warn "utility_analysis/08_threshold_audit.py not found, skipping audit"
    fi
else
    warn "guidelines.yaml not found (tried: ${GUIDELINES_SOURCE}, config/guidelines.yaml), skipping audit"
fi

# =============================================================================
# Step 5: Visualization (Optional)
# =============================================================================
if [ "${SKIP_VIZ:-0}" = "1" ]; then
    info "Skipping visualization (SKIP_VIZ=1)"
else
    print_header "Step 5: Visualization"
    
    info "Generating visualizations..."
    
    # Try aggregation first - pass guidelines for proper matching
    if [ -f "utility_plots/05_01_aggregate_thresholds.py" ]; then
        # Build aggregation command with optional guidelines
        AGG_CMD="python utility_plots/05_01_aggregate_thresholds.py --dataset_dirs ${DATASET_OUTDIR} --outdir ${FIG_OUTDIR}"
        
        # Add guidelines if available
        if [ -n "${GUIDELINES_PATH}" ] && [ -f "${GUIDELINES_PATH}" ]; then
            AGG_CMD="${AGG_CMD} --guidelines ${GUIDELINES_PATH}"
            info "Using guidelines: ${GUIDELINES_PATH}"
        fi
        
        if eval "${AGG_CMD}" 2>/dev/null; then
            success "Threshold aggregation complete"
            
            # Generate heatmap if aggregation succeeded
            if [ -f "${FIG_OUTDIR}/all_thresholds_summary.csv" ] && [ -f "utility_plots/05_02_agreement_heatmap.py" ]; then
                # Check if summary has data
                N_ROWS=$(wc -l < "${FIG_OUTDIR}/all_thresholds_summary.csv")
                if [ "$N_ROWS" -gt 1 ]; then
                    python utility_plots/05_02_agreement_heatmap.py \
                        --csv "${FIG_OUTDIR}/all_thresholds_summary.csv" \
                        --outdir "${FIG_OUTDIR}" \
                        --annotate 2>/dev/null && success "Heatmap generated" || warn "Heatmap generation failed (may need more data)"
                else
                    warn "all_thresholds_summary.csv is empty, skipping heatmap"
                fi
            fi
        else
            warn "Threshold aggregation failed or no data available"
        fi
    else
        warn "Visualization scripts not available"
    fi
fi

# =============================================================================
# Summary
# =============================================================================
print_header "Smoke Test Complete!"

echo ""
echo "Results Summary:"
echo "----------------"
echo "  Dataset:     ${DATASET}"
echo "  Output dir:  ${DATASET_OUTDIR}"
echo "  Figures:     ${FIG_OUTDIR}"
echo "  Log file:    ${LOG_FILE}"
echo ""

# Check key outputs
echo "Key Output Files:"
for f in \
    "${DATASET_OUTDIR}/aggregated/overall_metrics.csv" \
    "${DATASET_OUTDIR}/aggregated/thresholds_units.csv" \
    "${DATASET_OUTDIR}/aggregated/threshold_audit.csv"; do
    if [ -f "$f" ]; then
        echo -e "  ${GREEN}✓${NC} $(basename "$f")"
    else
        echo -e "  ${YELLOW}○${NC} $(basename "$f") (not created)"
    fi
done

echo ""
echo "============================================================"
echo -e "${GREEN}Smoke test completed at $(date)${NC}"
echo "============================================================"

# Cleanup
set +u
conda deactivate 2>/dev/null || true
