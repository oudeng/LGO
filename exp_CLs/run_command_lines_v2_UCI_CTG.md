# SRBench-Aligned Experimental Commands for UCI CTG

## Revision Summary (R2 Response to Reviewer 1)

### Changes Made for Fair Baseline Comparison

Based on the Reviewer's concern about baseline method configurations, we have aligned PySR and Operon parameters with SRBench recommendations. The key changes are:

| Parameter | Original | SRBench-Aligned | Reference |
|-----------|----------|-----------------|-----------|
| **PySR** ||||
| `niterations` | 20 (ngen//4) | 40 | SRBench default; PySR paper [Cranmer 2023] |
| `population_size` | 150 (pop_size//4) | 1000 | SRBench Table 5: `npop=1000` |
| `maxsize` | 20 | 30 | SRBench hyperparameter space |
| `populations` | 1 | 15 | PySR multi-island default |
| `timeout_in_seconds` | None | 3600 | SRBench: 1 hour max per run |
| `parsimony` | 0 | 0.001 | SRBench complexity penalty |
| **Operon** ||||
| `generations` | 80 | 500 | SRBench Table 6: large generation count |
| `population_size` | 600 | 1000 | SRBench Table 6: `popsize=1000` |
| `max_length` | 30 | 50 | SRBench: `max_length=50` |
| `max_evaluations` | None | 500000 | SRBench: 500k eval budget |
| `tournament_size` | 5 | 5 | SRBench default |
| `local_iterations` | 0 | 5 | SRBench: local search enabled |
| **LGO (Aligned)** ||||
| `pop_size` | 800 | 1000 | Aligned with baselines |
| `ngen` | 100 | 500 | Aligned with Operon generations |
| `max_evals` (implicit) | ~80k | 500000 | Aligned with SRBench budget |

### Evidence Sources

1. **SRBench NeurIPS 2021 Paper** (La Cava et al.): Tables 4-6 document hyperparameter spaces
   - URL: https://arxiv.org/abs/2107.14351
   - Termination criteria: "500k evaluations per training or 48 hours" (Section 4, Table 2)

2. **SRBench GitHub Repository**: https://github.com/cavalab/srbench
   - `algorithms/` folder contains official configurations for each method

3. **PySR Official Documentation**: https://astroautomata.com/PySR/
   - Default `niterations=40`, `population_size` per island, multi-population scheme

4. **Operon GECCO 2020 Paper** (Burlacu et al.): Original Operon publication
   - DOI: 10.1145/3377929.3398099

---

## ICU_composite_risk_score Experiments

### Part 1: LGO + PySR + Operon (SRBench-Aligned)

```bash
conda activate py310

python run_v3_8_2.py \
  --csv data/UCI/CTG_nsp_bin.csv \
  --target NSP_bin \
  --task binary_classification \
  --experiments base,lgo_soft,lgo_hard,pysr,operon \
  --seeds 1,2,3,5,8,13,21,34,55,89 \
  --test_size 0.2 \
  --outdir overall_UCI_CTG_NSPbin \
  --hparams_json '{
    "gate_expr_enable": true,
    "pop_size": 1000,
    "ngen": 500,
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
    "LB":"bpm", "ASTV":"%", "MSTV":"ms", "ALTV":"%", "MLTV":"ms", "DL":"count",
    "DS":"count", "DP":"count", "AC":"count", "FM":"count", "UC":"count",
    "Width":"bpm", "MinHist":"bpm", "MaxHist":"bpm", "ModeHist":"bpm",
    "MeanHist":"bpm", "MedianHist":"bpm", "VarHist":"bpm^2",
    "Tendency":"", "Nmax":"count", "Nzeros":"count"
  }' \
  --dataset UCI_CTG_NSPbin \
  --save_predictions
```

### Part 2: PSTree (Separate Environment)

```bash
conda activate pstree

python run_v3_8_2.py \
  --csv data/UCI/CTG_nsp_bin.csv \
  --target NSP_bin \
  --task binary_classification \
  --experiments pstree \
  --seeds 1,2,3,5,8,13,21,34,55,89 \
  --test_size 0.2 \
  --hparams_json '{"pst_max_depth":10,"pst_max_leaf_nodes":32,"pst_min_samples_leaf":20,"pst_pop_size":600,"pst_ngen":80}' \
  --outdir overall_UCI_CTG_NSPbin \
  --dataset UCI_CTG_NSPbin \
  --save_predictions
```

### Part 3: RILS-ROLS (Separate Environment)

```bash
conda activate rils-rols

python run_v3_8_2.py \
  --csv data/UCI/CTG_nsp_bin.csv \
  --target NSP_bin \
  --task binary_classification \
  --experiments rils_rols \
  --seeds 1,2,3,5,8,13,21,34,55,89 \
  --test_size 0.2 \
  --hparams_json '{"max_fit_calls":100000,"max_time":3600,"max_complexity":50,"sample_size":1.0,"complexity_penalty":0.001,"verbose": false}' \
  --outdir overall_UCI_CTG_NSPbin \
  --dataset UCI_CTG_NSPbin \
  --save_predictions
```

---

## Post-Processing / Utility Analysis

```bash
conda activate py310

# Collect hyperparameters
python utility_analysis/01_collect_hyperparams.py \
  --dataset_dir overall_UCI_CTG_NSPbin \
  --dataset UCI_CTG_NSPbin

# Runtime profiling
python utility_analysis/02_collect_runtime_profile.py \
  --dataset_dir overall_UCI_CTG_NSPbin 

# Export expressions with complexity
python utility_analysis/03_export_expressions.py \
  --dataset_dir overall_UCI_CTG_NSPbin \
  --method all \
  --dataset UCI_CTG_NSPbin \
  --topk 10

# Complexity and stability analysis
python utility_analysis/04_gen_complexity_and_stability.py \
  --dataset_dir overall_UCI_CTG_NSPbin \
  --method all \
  --dataset UCI_CTG_NSPbin

# Pareto front generation
python utility_analysis/05_gen_pareto_front.py \
  --dataset_dir overall_UCI_CTG_NSPbin \
  --method all \
  --dataset UCI_CTG_NSPbin

# LGO-specific: Gating usage frequency (Top-K)
python utility_analysis/06_gen_gating_usage.py \
  --dataset_dir overall_UCI_CTG_NSPbin \
  --method lgo \
  --dataset UCI_CTG_NSPbin

# Threshold and unit analysis
python utility_analysis/07_gen_thresholds_units.py \
  --dataset_dir overall_UCI_CTG_NSPbin \
  --dataset UCI_CTG_NSPbin \
  --method lgo --topk 10 --experiments lgo_hard

# Threshold audit against clinical guidelines
python utility_analysis/08_convert_ground_truth_to_guidelines.py \
  --ground_truth data/UCI/CTG_ground_truth.json \
  --dataset UCI_CTG_NSPbin \
  --output overall_UCI_CTG_NSPbin/config/guidelines.yaml

# Ablation analysis
python utility_analysis/09_build_ablation_matrix.py \
  --dataset_dir overall_UCI_CTG_NSPbin \
  --dataset UCI_CTG_NSPbin 

# Stability summary
python utility_analysis/10_stability_summary.py \
  --dataset_dir overall_UCI_CTG_NSPbin \
  --dataset UCI_CTG_NSPbin
```

# Add Train evaluation 

python run_v3_8_3.py \
  --csv data/UCI/CTG_nsp_bin.csv \
  --target NSP_bin \
  --task binary_classification \
  --experiments base,lgo_soft,lgo_hard,pysr,operon \
  --outdir overall_UCI_CTG_NSPbin  \
  --dataset UCI_CTG_NSPbin \
  --seeds 1,2,3,5,8,13,21,34,55,89 \
  --topk 0 --save_train_metrics --compute_overfit_summary