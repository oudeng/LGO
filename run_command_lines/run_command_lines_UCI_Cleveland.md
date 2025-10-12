## UCI HeartDisease Cleveland
## run_v3_8.py
```bash
conda activate py310

python run_v3_8.py \
  --csv data/UCI/Heart_Cleveland_num.csv \
  --target num \
  --task regression \
  --experiments base,lgo_soft,lgo_hard,pysr,operon \
  --seeds 1,2,3,5,8,13,21,34,55,89 \
  --test_size 0.2 \
  --outdir overall_UCI_Heart_Cleveland_num \
  --hparams_json '{
    "gate_expr_enable": true,
    "pop_size": 800,
    "ngen": 100
  }' \
  --unit_map_json '{
    "age":"years",
    "sex":"",
    "cp":"",
    "trestbps":"mmHg",
    "chol":"mg/dL",
    "fbs":"",
    "restecg":"",
    "thalach":"bpm",
    "exang":"",
    "oldpeak":"",
    "slope":"",
    "ca":"count",
    "thal":""
  }' \
  --dataset UCI_Heart_Cleveland_num \
  --save_predictions

conda activate pstree

python run_v3_8.py \
  --csv data/UCI/Heart_Cleveland_num.csv \
  --target num \
  --task regression \
  --experiments pstree \
  --seeds 1,2,3,5,8,13,21,34,55,89 \
  --test_size 0.2 \
  --hparams_json '{"pst_max_depth":10,"pst_max_leaf_nodes":32,"pst_min_samples_leaf":20,"pst_pop_size":600,"pst_ngen":80}' \
  --outdir overall_UCI_Heart_Cleveland_num \
  --dataset UCI_Heart_Cleveland_num \
  --save_predictions


conda activate rils-rols

python run_v3_8.py \
  --csv data/UCI/Heart_Cleveland_num.csv \
  --target num \
  --task regression \
  --experiments rils_rols \
  --seeds 1,2,3,5,8,13,21,34,55,89 \
  --test_size 0.2 \
  --hparams_json '{"max_fit_calls":100000,"max_time":3600,"max_complexity":50,"sample_size":1.0,"complexity_penalty":0.001,"verbose": false}' \
  --outdir overall_UCI_Heart_Cleveland_num \
  --dataset UCI_Heart_Cleveland_num \
  --save_predictions

# ./utility_analysis/

python utility_analysis/01_collect_hyperparams.py \
  --dataset_dir overall_UCI_Heart_Cleveland_num \
  --dataset UCI_Heart_Cleveland_num

python utility_analysis/02_collect_runtime_profile.py \
  --dataset_dir overall_UCI_Heart_Cleveland_num

# 	表达式与复杂度
python utility_analysis/03_export_expressions.py \
  --dataset_dir overall_UCI_Heart_Cleveland_num \
  --dataset UCI_Heart_Cleveland_numc \
  --method all \
  --topk 10

python utility_analysis/04_gen_complexity_and_stability.py \
  --dataset_dir overall_UCI_Heart_Cleveland_num \
  --dataset UCI_Heart_Cleveland_num \
  --method all 

python utility_analysis/05_gen_pareto_front.py \
  --dataset_dir overall_UCI_Heart_Cleveland_num \
  --dataset UCI_Heart_Cleveland_num \
  --method all 

# lgo 专用：门使用频次（Top‑K）
python utility_analysis/06_gen_gating_usage.py \
  --dataset_dir overall_UCI_Heart_Cleveland_num \
  --dataset UCI_Heart_Cleveland_num \
  --method lgo 

# 阈值与审计
python utility_analysis/07_gen_thresholds_units.py \
  --dataset_dir overall_UCI_Heart_Cleveland_num \
  --dataset UCI_Heart_Cleveland_num \
  --method lgo --topk 10 --experiments lgo_hard

######################
  --ground_truth /home/dengou/data_UCI_HeartDisease/output/ground_truth.json \
  --dataset UCI_Heart_Cleveland_num \
#######################

python utility_analysis/08_convert_ground_truth_to_guidelines.py \
  --ground_truth /home/dengou/data_UCI_HeartDisease/output/ground_truth.json \
  --dataset UCI_Heart_Cleveland_num \
  --output config/guidelines.yaml

python utility_analysis/08_threshold_audit.py \
  --dataset_dir overall_UCI_Heart_Cleveland_num \
  --dataset UCI_Heart_Cleveland_num \
  --guidelines config/guidelines.yaml

# 消融与稳定性
python utility_analysis/09_build_ablation_matrix.py \
  --dataset_dir overall_UCI_Heart_Cleveland_num \
  --dataset UCI_Heart_Cleveland_num 

python utility_analysis/10_stability_summary.py \
  --dataset_dir overall_UCI_Heart_Cleveland_num \
  --dataset UCI_Heart_Cleveland_num 
```

