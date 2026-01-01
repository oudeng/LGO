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
if [ "${SKIP_LGO:-0}" = "1" ]; then
    print_header "Step 2: LGO Symbolic Regression (SKIPPED)"
    info "SKIP_LGO=1, using existing results in ${DATASET_OUTDIR}"
    
    if [ ! -d "${DATASET_OUTDIR}/candidates" ]; then
        err "No existing results found. Run without SKIP_LGO first."
    fi
else
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
fi

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
    mkdir -p "${FIG_OUTDIR}"
    
    VIZ_SUCCESS=0
    
    # Method 1: Run 05_01 to create aggregated summary CSV
    if [ -f "utility_plots/05_01_aggregate_thresholds.py" ]; then
        python utility_plots/05_01_aggregate_thresholds.py \
            --dataset_dirs "${DATASET_OUTDIR}" \
            --outdir "${FIG_OUTDIR}" 2>&1 || true
        
        if [ -f "${FIG_OUTDIR}/all_thresholds_summary.csv" ]; then
            success "Created: ${FIG_OUTDIR}/all_thresholds_summary.csv"
        fi
    fi
    
    # Method 2: Generate threshold heatmap from threshold_audit.csv + guidelines.yaml
    # This method directly reads guidelines.yaml and matches by original feature name
    if [ -f "${DATASET_OUTDIR}/aggregated/threshold_audit.csv" ] && [ -f "${GUIDELINES_PATH}" ]; then
        info "Generating threshold heatmap..."
        
        export DATASET_OUTDIR="${DATASET_OUTDIR}"
        export FIG_OUTDIR="${FIG_OUTDIR}"
        export DATASET_NAME="${DATASET_NAME}"
        export GUIDELINES_PATH="${GUIDELINES_PATH}"
        
        python3 << 'PYEOF'
import pandas as pd
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

dataset_outdir = os.environ.get('DATASET_OUTDIR')
fig_outdir = os.environ.get('FIG_OUTDIR')
dataset_name = os.environ.get('DATASET_NAME')
guidelines_path = os.environ.get('GUIDELINES_PATH')

# Load threshold data
audit_path = f"{dataset_outdir}/aggregated/threshold_audit.csv"
df = pd.read_csv(audit_path)

# Load guidelines
with open(guidelines_path) as f:
    guidelines = yaml.safe_load(f)

# Get dataset-specific guidelines, fallback to global
ds_guidelines = guidelines.get('datasets', {}).get(dataset_name, {})
global_guidelines = guidelines.get('global', {})

# Merge: dataset-specific takes priority
all_guidelines = {**global_guidelines, **ds_guidelines}

print(f"Loaded {len(all_guidelines)} guidelines")

# Match guidelines using original feature name (not feature_norm)
df['guideline_matched'] = df['feature'].map(all_guidelines)

# Calculate relative error
df['rel_error_calc'] = (df['median'] - df['guideline_matched']) / df['guideline_matched']

print("Matching results:")
for _, row in df.iterrows():
    g = row['guideline_matched']
    m = row['median']
    m_str = f"{m:.2f}" if pd.notna(m) else "N/A"
    g_str = f"{g}" if pd.notna(g) else "N/A"
    print(f"  {row['feature']}: median={m_str}, guideline={g_str}")

# Filter valid rows
df_plot = df[df['median'].notna()].copy()
df_plot = df_plot[~df_plot['feature'].isin(['unknown', 'zero', 'one', 'gender_std'])].copy()

if len(df_plot) == 0:
    print("No valid features to plot")
    exit(0)

print(f"Plotting {len(df_plot)} features...")

# Create traffic-light style figure
fig, ax = plt.subplots(figsize=(6, max(3, len(df_plot) * 0.9)))

colors = []
for _, row in df_plot.iterrows():
    if pd.isna(row['guideline_matched']) or pd.isna(row['rel_error_calc']):
        colors.append('#808080')
    elif abs(row['rel_error_calc']) <= 0.1:
        colors.append('#2ecc71')
    elif abs(row['rel_error_calc']) <= 0.2:
        colors.append('#f39c12')
    else:
        colors.append('#e74c3c')

y_pos = np.arange(len(df_plot))
bars = ax.barh(y_pos, [1]*len(df_plot), color=colors, edgecolor='white', height=0.7)

# Add text annotations
for i, (_, row) in enumerate(df_plot.iterrows()):
    val_text = f"{row['median']:.1f}"
    if pd.notna(row['rel_error_calc']):
        val_text += f"\nΔ{abs(row['rel_error_calc'])*100:.0f}%"
    ax.text(0.5, i, val_text, ha='center', va='center', fontsize=11, fontweight='bold', color='white')

labels = [row['feature'].replace('_', ' ').title() for _, row in df_plot.iterrows()]
ax.set_yticks(y_pos)
ax.set_yticklabels(labels, fontsize=10)
ax.set_xlim(0, 1)
ax.set_xticks([])
ax.set_title(f"{dataset_name.replace('_', ' ')}", fontsize=12, fontweight='bold')

for spine in ax.spines.values():
    spine.set_visible(False)

from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#2ecc71', label='≤10%'),
    Patch(facecolor='#f39c12', label='≤20%'),
    Patch(facecolor='#e74c3c', label='>20%'),
    Patch(facecolor='#808080', label='No guideline')
]
ax.legend(handles=legend_elements, loc='lower right', title='Relative Error', fontsize=8)

plt.tight_layout()
os.makedirs(fig_outdir, exist_ok=True)
out_path = f"{fig_outdir}/threshold_heatmap_{dataset_name}.png"
plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white')
print(f"[OK] Saved: {out_path}")
PYEOF
        
        if [ -f "${FIG_OUTDIR}/threshold_heatmap_${DATASET_NAME}.png" ]; then
            VIZ_SUCCESS=1
        fi
    fi
    
    # Report results
    echo ""
    if [ "$VIZ_SUCCESS" = "1" ] || ls "${FIG_OUTDIR}"/*.png 1>/dev/null 2>&1; then
        success "Visualization complete"
        info "Generated files:"
        ls -la "${FIG_OUTDIR}"/*.png "${FIG_OUTDIR}"/*.csv 2>/dev/null | head -10
    else
        warn "No PNG visualizations generated"
        info "CSV data available for manual plotting:"
        ls -la "${DATASET_OUTDIR}/aggregated/"*.csv 2>/dev/null || true
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
