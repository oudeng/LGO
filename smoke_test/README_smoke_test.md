
# Quick run
```bash
chmod +x smoke_test/run_smoke_NHANES.sh
bash smoke_test/run_smoke_NHANES.sh
```

# What will be done via this sh file
On LGO root path, run ```bash smoke_test/run_smoke.sh```. t will proceed with smoke tests within about 5 minutes, including downloading the GitHub repo, installing the environment, running smoke test scripts, and finally visualizing the results for quick confirmation.
If the following steps 1 and 2 are already completed, bash will skip these two steps and go to step 3 directly. For details, see "robust repo detection / clone logic" in the sh file.

1. Clone Repository，including the necessary scripts and dataset for smoke test.
```bash
  git clone https://github.com/oudeng/LGO.git
  cd LGO
```

2. Install the necessary environment for smoke test.
```bash
   conda env create -f env_setup/env_py310_test.yml
   conda activate py310_test
```

3. run_v3_8.py
```basg
python run_v3_8.py \
  --csv data/NHANES/NHANES_metabolic_score.csv \
  --target metabolic_score \
  --task regression \
  --experiments lgo_soft,lgo_hard\
  --seeds 1,2,3 \
  --test_size 0.2 \
  --outdir smoke_test/NHANES \
  --dataset NHANES_metabolic_score \
  --save_predictions \
  --hparams_json '{
    "gate_expr_enable": true, "pop_size": 800, "ngen": 100,
    "include_lgo_multi": true, "include_lgo_and3": true, 
    "micro_mutation_prob": 0.10, "cv_proxy_weight": 0.0
  }' \
  --unit_map_json '{
  "systolic_bp":"mmHg",
  "triglycerides":"mg/dL",
  "waist_circumference":"cm",
  "fasting_glucose":"mg/dL",
  "hdl_cholesterol":"mg/dL",
  "age":"years"
   }' 
```

4. Threshold audit
Execute 08_threshold_audit.py in /utility_analysis and add the results to the specified output directory above.
```bash
python utility_analysis/07_gen_thresholds_units.py \
  --dataset_dir smoke_test/NHANES \
  --dataset NHANES_metabolic_score \
  --method lgo --topk 10 --experiments lgo_hard

python utility_analysis/08_threshold_audit.py \
  --dataset_dir  smoke_test/NHANES \
  --dataset NHANES_metabolic_score \
  --guidelines config/guidelines.yaml
```

5. Visualization
Execute `08_threshold_audit.py` in `/utility_analysis` to visualize the results from the output directory specified above.

```bash
python utility_plots/04_thresholds_v3_7.py \
  --dataset_dirs smoke_test/NHANES \
  --config_dir config \
  --method lgo --experiment lgo_hard \
  --only_anchored \
  --annotate \
  --outdir smoke_test/fig

python  utility_plots/04_thresholds_plot_r1.py \
  --csv smoke_test/fig/v3_thresholds_summary.csv \
  --outdir smoke_test/fig
```

## Option: ICU with the same procedures.
```bash
python run_v3_8.py \
  --csv data/ICU/ICU_composite_risk_score.csv \
  --target composite_risk_score \
  --task regression \
  --experiments lgo_soft,lgo_hard \
  --seeds 1,2,3 \
  --test_size 0.2 \
  --outdir smoke_test/ICU \
  --dataset ICU_composite_risk_score \
  --save_predictions \
  --hparams_json '{
    "gate_expr_enable": true,
    "pop_size": 800,
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
  }' \
  --unit_map_json '{
    "map_mmhg":"mmHg","sbp_min":"mmHg","dbp_min":"mmHg",
    "lactate_mmol_l":"mmol/L","creatinine_mg_dl":"mg/dL","hemoglobin_min":"g/dL",
    "sodium_min":"mmol/L","age_years":"years","hr_max":"bpm","resprate_max":"/min","spo2_min":"%"
  }' 

python utility_analysis/07_gen_thresholds_units.py \
  --dataset_dir smoke_test/ICU \
  --dataset ICU_composite_risk_score \
  --method lgo --topk 10 --experiments lgo_hard

python utility_analysis/08_threshold_audit.py \
  --dataset_dir smoke_test/ICU  \
  --dataset ICU_composite_risk_score \
  --guidelines config/guidelines.yaml

python utility_plots/04_thresholds_v3_7.py \
  --dataset_dirs smoke_test/ICU \
  --config_dir config \
  --method lgo --experiment lgo_hard \
  --only_anchored \
  --annotate \
  --outdir smoke_test/fig

python  utility_plots/04_thresholds_plot_r1.py \
  --csv smoke_test/fig/v3_thresholds_summary.csv \
  --outdir smoke_test/fig
```
