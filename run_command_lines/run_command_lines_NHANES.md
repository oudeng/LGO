
## NHANES_metabolic_score
## run_v3_8.py

```bash
conda activate py310

python run_v3_8.py \
  --csv data/NHANES/NHANES_metabolic_score.csv \
  --target metabolic_score \
  --task regression \
  --experiments base,lgo_soft,lgo_hard,pysr,operon \
  --seeds 1,2,3,5,8,13,21,34,55,89 \
  --test_size 0.2 \
  --outdir overall_NHANES_metabolic_score \
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
   }' \
  --dataset NHANES_metabolic_score \
  --save_predictions


conda activate pstree

python run_v3_8.py \
  --csv data/NHANES/NHANES_metabolic_score.csv \
  --target metabolic_score \
  --task regression \
  --experiments pstree \
  --seeds 1,2,3,5,8,13,21,34,55,89 \
  --test_size 0.2 \
  --hparams_json '{"pst_max_depth":10,"pst_max_leaf_nodes":32,"pst_min_samples_leaf":20,"pst_pop_size":600,"pst_ngen":80}' \
  --outdir overall_NHANES_metabolic_score \
  --dataset NHANES_metabolic_score \
  --save_predictions


conda activate rils-rols

python run_v3_8.py \
  --csv data/NHANES/NHANES_metabolic_score.csv \
  --target metabolic_score \
  --task regression \
  --experiments rils_rols \
  --seeds 1,2,3,5,8,13,21,34,55,89 \
  --test_size 0.2 \
  --hparams_json '{"max_fit_calls":100000,"max_time":3600,"max_complexity":50,"sample_size":1.0,"complexity_penalty":0.001,"verbose": false}' \
  --outdir overall_NHANES_metabolic_score \
  --dataset NHANES_metabolic_score \
  --save_predictions

# ./utility_analysis/

python utility_analysis/01_collect_hyperparams.py \
  --dataset_dir overall_NHANES_metabolic_score \
  --dataset NHANES_metabolic_score

python utility_analysis/02_collect_runtime_profile.py \
  --dataset_dir overall_NHANES_metabolic_score

# 	表达式与复杂度
python utility_analysis/03_export_expressions.py \
  --dataset_dir overall_NHANES_metabolic_score \
  --dataset NHANES_metabolic_score \
  --method all \
  --topk 10

python utility_analysis/04_gen_complexity_and_stability.py \
  --dataset_dir overall_NHANES_metabolic_score \
  --dataset NHANES_metabolic_score \
  --method all 

python utility_analysis/05_gen_pareto_front.py \
  --dataset_dir overall_NHANES_metabolic_score \
  --dataset NHANES_metabolic_score \
  --method all 

# lgo 专用：门使用频次（Top‑K）
python utility_analysis/06_gen_gating_usage.py \
  --dataset_dir overall_NHANES_metabolic_score \
  --dataset NHANES_metabolic_score \
  --method lgo 

# 阈值与审计
python utility_analysis/07_gen_thresholds_units.py \
  --dataset_dir overall_NHANES_metabolic_score \
  --dataset NHANES_metabolic_score \
  --method lgo --topk 10 --experiments lgo_hard

######################
  --ground_truth /home/dengou/data_NHANES/output/ground_truth.json \
   --dataset NHANES_metabolic_score \
#######################

python utility_analysis/08_convert_ground_truth_to_guidelines.py \
  --ground_truth /home/dengou/data_NHANES/output/ground_truth.json \
   --dataset NHANES_metabolic_score \
  --output config/guidelines.yaml

python utility_analysis/08_threshold_audit.py \
  --dataset_dir overall_NHANES_metabolic_score \
  --dataset NHANES_metabolic_score \
  --guidelines config/guidelines.yaml

# 消融与稳定性
python utility_analysis/09_build_ablation_matrix.py \
  --dataset_dir overall_NHANES_metabolic_score \
  --dataset NHANES_metabolic_score 

python utility_analysis/10_stability_summary.py \
  --dataset_dir overall_NHANES_metabolic_score \
  --dataset NHANES_metabolic_score 
```

