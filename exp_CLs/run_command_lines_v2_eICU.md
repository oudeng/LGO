# SRBench-Aligned Experimental Commands for eICU_composite_risk_score

## ICU_composite_risk_score Experiments

### Part 1: LGO + PySR + Operon (SRBench-Aligned)

conda activate py310

python run_v3_8_2.py \
  --csv data/eICU/eICU_composite_risk_score.csv \
  --target composite_risk_score \
  --task regression \
  --experiments base,lgo_soft,lgo_hard,pysr,operon \
  --seeds 1,2,3,5,8,13,21,34,55,89 \
  --test_size 0.2 \
  --outdir overall_eICU_composite_risk_score \
  --hparams_json '{
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
    "include_lgo_pair": false,
    
    "pysr_niterations": 40,
    "pysr_population_size": 1000,
    "pysr_populations": 15,
    "pysr_maxsize": 30,
    "pysr_timeout_in_seconds": 3600,
    "pysr_parsimony": 0.001,
    "pysr_ncyclesperiteration": 550,
    "pysr_binary_operators": ["+", "-", "*", "/"],
    "pysr_unary_operators": ["sqrt", "exp", "log", "sin", "cos"],
    
    "operon_generations": 500,
    "operon_population_size": 1000,
    "operon_max_length": 50,
    "operon_max_evaluations": 500000,
    "operon_tournament_size": 5,
    "operon_local_iterations": 5,
    "operon_allowed_symbols": "add,mul,aq,exp,log,sin,cos,sqrt,constant,variable"
  }' \
  --unit_map_json '{
   "map_mmhg": "mmHg","sbp_min": "mmHg","dbp_min": "mmHg","lactate_mmol_l": "mmol/L","creatinine_mg_dl": "mg/dL","hemoglobin_min": "g/dL","sodium_min": "mmol/L","age_years": "years","hr_max": "bpm","resprate_max": "/min","spo2_min": "%","gcs": "","urine_output_min": "mL"
  }' \
  --dataset eICU_composite_risk_score \
  --save_predictions

conda activate pstree

python run_v3_8_2.py \
  --csv data/eICU/eICU_composite_risk_score.csv \
  --target composite_risk_score \
  --task regression \
  --experiments pstree \
  --seeds 1,2,3,5,8,13,21,34,55,89 \
  --test_size 0.2 \
  --hparams_json '{"pst_max_depth":10,"pst_max_leaf_nodes":32,"pst_min_samples_leaf":20,"pst_pop_size":600,"pst_ngen":80}' \
  --outdir overall_eICU_composite_risk_score \
  --dataset eICU_composite_risk_score \
  --save_predictions

conda activate rils-rols

python run_v3_8_2.py \
  --csv data/eICU/eICU_composite_risk_score.csv \
  --target composite_risk_score \
  --task regression \
  --experiments rils_rols \
  --seeds 1,2,3,5,8,13,21,34,55,89 \
  --test_size 0.2 \
  --hparams_json '{"max_fit_calls":500000,"max_time":3600,"max_complexity":50,"sample_size":1.0,"complexity_penalty":0.001,"verbose": false}' \
  --outdir overall_eICU_composite_risk_score \
  --dataset eICU_composite_risk_score \
  --save_predictions

---

## Post-Processing / Utility Analysis

```bash
conda activate py310

# Collect hyperparameters
python utility_analysis/01_collect_hyperparams.py \
  --dataset_dir overall_eICU_composite_risk_score \
  --dataset eICU_composite_risk_score

# Runtime profiling
python utility_analysis/02_collect_runtime_profile.py \
  --dataset_dir overall_eICU_composite_risk_score

# Export expressions with complexity
python utility_analysis/03_export_expressions.py \
  --dataset_dir overall_eICU_composite_risk_score \
  --method all \
  --dataset eICU_composite_risk_score \
  --topk 10

# Complexity and stability analysis
python utility_analysis/04_gen_complexity_and_stability.py \
  --dataset_dir overall_eICU_composite_risk_score \
  --method all \
  --dataset eICU_composite_risk_score

# Pareto front generation
python utility_analysis/05_gen_pareto_front.py \
  --dataset_dir overall_eICU_composite_risk_score \
  --method all \
  --dataset eICU_composite_risk_score

# LGO-specific: Gating usage frequency (Top-K)
python utility_analysis/06_gen_gating_usage.py \
  --dataset_dir overall_eICU_composite_risk_score \
  --method lgo \
  --dataset eICU_composite_risk_score 

# Threshold and unit analysis
python utility_analysis/07_gen_thresholds_units.py \
  --dataset_dir overall_eICU_composite_risk_score \
  --dataset eICU_composite_risk_score \
  --method lgo --topk 10 --experiments lgo_hard

# Threshold audit against clinical guidelines
#python utility_analysis/08_convert_ground_truth_to_guidelines.py \
#  --ground_truth /home/dengou/data_MIMIC_ICU/output/ground_truth.json \
#  --dataset ICU_composite_risk_score \
#  --output config/guidelines.yaml

python utility_analysis/08_convert_ground_truth_to_guidelines.py \
  --ground_truth data/eICU/ground_truth.json \
  --dataset eICU_composite_risk_score \
  --output overall_eICU_composite_risk_score/config/guidelines.yaml

#python utility_analysis/08_threshold_audit.py \
#  --dataset_dir overall_ICU_composite_risk_score \
#  --dataset ICU_composite_risk_score \
#  --guidelines overall_ICU_composite_risk_score/config/guidelines.yaml

# Ablation analysis
python utility_analysis/09_build_ablation_matrix.py \
  --dataset_dir overall_eICU_composite_risk_score \
  --dataset eICU_composite_risk_score

# Stability summary
python utility_analysis/10_stability_summary.py \
  --dataset_dir overall_eICU_composite_risk_score \
  --dataset eICU_composite_risk_score

```
# Add Train evaluation 
move to: overall_eICU_composite_risk_score/aggregated/
如果输出到一个新的目录（如下），就没有问题。看来是和原本有的 overall_metrics.csv 发生冲突，先不改，节约时间。

python run_v3_8_3.py \
  --csv data/eICU/eICU_composite_risk_score.csv \
  --target composite_risk_score --task regression \
  --experiments base,lgo_soft,lgo_hard,pysr,operon \
  --outdir overall_eICU_composite_risk_score \
  --dataset eICU_composite_risk_score \
  --seeds 1,2,3,5,8,13,21,34,55,89 \
  --topk 0 --save_train_metrics --compute_overfit_summary

