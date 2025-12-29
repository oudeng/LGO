#!/bin/bash
# -*- coding: utf-8 -*-
# 05_00_run_all_analysis.sh - Run complete threshold analysis pipeline
#
# Usage:
#   bash utility_analysis/05_00_run_all_analysis.sh [options] [dataset_dirs...]
#
# Options:
#   --data_root DIR       Root directory containing overall_* directories
#   --roots DIR [DIR...]  Dataset directories to process (alternative to --data_root)
#   --outdir DIR          Output directory for all results (default: threshold_analysis)
#   --skip_aggregate      Skip aggregation step (use existing CSV)
#
# Examples:
#   # Auto-find datasets in current directory
#   bash 05_00_run_all_analysis.sh --outdir results
#
#   # Specify datasets directly
#   bash 05_00_run_all_analysis.sh \
#     overall_ICU_composite_risk_score \
#     overall_eICU_composite_risk_score \
#     overall_NHANES_metabolic_score \
#     --outdir results
#
#   # Use --roots flag
#   bash 05_00_run_all_analysis.sh \
#     --roots overall_ICU_* overall_eICU_* overall_NHANES_* \
#     --outdir results
#
# This script orchestrates the complete threshold analysis workflow:
# 1. Aggregate threshold data from all datasets (including eICU)
# 2. Generate agreement heatmaps
# 3. Generate distribution plots
# 4. Analyze gating parsimony
# 5. Create summary tables
# 6. Generate publication figure
#
# Supported datasets:
#   - ICU (MIMIC-IV)
#   - eICU
#   - NHANES
#   - UCI CTG
#   - UCI Cleveland
#   - UCI Hydraulic

set -e  # Exit on error

# Default parameters
DATA_ROOT=""
OUTDIR="threshold_analysis"
SKIP_AGGREGATE=false
DATASET_DIRS=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --data_root)
            DATA_ROOT="$2"
            shift 2
            ;;
        --roots|--dataset_dirs)
            # Collect all following arguments until next option
            shift
            while [[ $# -gt 0 ]] && [[ ! "$1" =~ ^-- ]]; do
                DATASET_DIRS="$DATASET_DIRS $1"
                shift
            done
            ;;
        --outdir)
            OUTDIR="$2"
            shift 2
            ;;
        --skip_aggregate)
            SKIP_AGGREGATE=true
            shift
            ;;
        *)
            # Assume it's a dataset directory if it starts with "overall_"
            if [[ "$1" =~ ^overall_ ]] || [[ -d "$1" ]]; then
                DATASET_DIRS="$DATASET_DIRS $1"
                shift
            else
                echo "Unknown option: $1"
                exit 1
            fi
            ;;
    esac
done

# Find dataset directories if not provided directly
if [ -z "$DATASET_DIRS" ]; then
    if [ -z "$DATA_ROOT" ]; then
        DATA_ROOT="."
    fi
    DATASET_DIRS=$(find "$DATA_ROOT" -maxdepth 1 -type d -name "overall_*" | sort)
fi

# Trim leading/trailing whitespace
DATASET_DIRS=$(echo "$DATASET_DIRS" | xargs)

if [ -z "$DATASET_DIRS" ]; then
    echo "[ERROR] No overall_* directories found in $DATA_ROOT"
    exit 1
fi

echo "============================================================"
echo "LGO Threshold Analysis Pipeline"
echo "============================================================"
echo "Output: $OUTDIR"
echo "Datasets:"
for d in $DATASET_DIRS; do
    basename_d=$(basename $d)
    # Highlight clinical datasets
    if [[ "$basename_d" == *"ICU"* ]] || [[ "$basename_d" == *"eICU"* ]] || [[ "$basename_d" == *"NHANES"* ]]; then
        echo "  - $basename_d [CLINICAL]"
    else
        echo "  - $basename_d"
    fi
done
echo ""

# Create output directory
mkdir -p "$OUTDIR"

# Determine script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Step 1: Aggregate threshold data
SUMMARY_CSV="$OUTDIR/all_thresholds_summary.csv"

if [ "$SKIP_AGGREGATE" = false ] || [ ! -f "$SUMMARY_CSV" ]; then
    echo "============================================================"
    echo "Step 1: Aggregating threshold data"
    echo "============================================================"
    python "$SCRIPT_DIR/05_01_aggregate_thresholds.py" \
        --dataset_dirs $DATASET_DIRS \
        --outdir "$OUTDIR" \
        --only_anchored
else
    echo "[SKIP] Using existing summary: $SUMMARY_CSV"
fi

# Check if summary was created
if [ ! -f "$SUMMARY_CSV" ]; then
    echo "[ERROR] Summary CSV not created"
    exit 1
fi

# Step 2: Generate agreement heatmaps
echo ""
echo "============================================================"
echo "Step 2: Generating agreement heatmaps"
echo "============================================================"
python "$SCRIPT_DIR/05_02_agreement_heatmap.py" \
    --csv "$SUMMARY_CSV" \
    --outdir "$OUTDIR/heatmaps" \
    --combined \
    --annotate

# Step 3: Generate distribution plots
echo ""
echo "============================================================"
echo "Step 3: Generating distribution plots"
echo "============================================================"
python "$SCRIPT_DIR/05_03_distribution_plot.py" \
    --csv "$SUMMARY_CSV" \
    --outdir "$OUTDIR/distributions" \
    --combined

# Step 4: Analyze gating parsimony
echo ""
echo "============================================================"
echo "Step 4: Analyzing gating parsimony"
echo "============================================================"
python "$SCRIPT_DIR/05_04_gating_parsimony.py" \
    --dataset_dirs $DATASET_DIRS \
    --outdir "$OUTDIR/gating"

# Step 5: Create summary tables
echo ""
echo "============================================================"
echo "Step 5: Creating summary tables"
echo "============================================================"
python "$SCRIPT_DIR/05_05_cross_dataset_table.py" \
    --csv "$SUMMARY_CSV" \
    --outdir "$OUTDIR/tables"

# Step 6: Generate publication figure
echo ""
echo "============================================================"
echo "Step 6: Generating publication figure"
echo "============================================================"
python "$SCRIPT_DIR/05_06_publication_figure.py" \
    --csv "$SUMMARY_CSV" \
    --outdir "$OUTDIR/publication" \
    --dpi 300

# Summary
echo ""
echo "============================================================"
echo "ANALYSIS COMPLETE"
echo "============================================================"
echo ""
echo "Output files:"
echo "  Summary CSV:     $OUTDIR/all_thresholds_summary.csv"
echo "  Heatmaps:        $OUTDIR/heatmaps/"
echo "  Distributions:   $OUTDIR/distributions/"
echo "  Gating analysis: $OUTDIR/gating/"
echo "  Tables:          $OUTDIR/tables/"
echo "  Publication:     $OUTDIR/publication/"
echo ""
echo "Key figures for paper:"
echo "  - $OUTDIR/publication/figure2_thresholds.pdf"
echo "  - $OUTDIR/publication/figure2_thresholds_2x3.pdf"
echo "  - $OUTDIR/gating/gate_usage_comparison.pdf"
echo "  - $OUTDIR/tables/threshold_table.tex"
echo "  - $OUTDIR/tables/summary_table.tex"