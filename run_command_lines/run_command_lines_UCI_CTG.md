## UCI CTG
## run_v3_8.py  

``bash
conda activate py310

python run_v3_8.py \
  --csv data/UCI/CTG_nsp_bin.csv \
  --target NSP_bin \
  --task binary_classification \
  --experiments base,lgo_soft,lgo_hard,pysr,operon \
  --seeds 1,2,3,5,8,13,21,34,55,89 \
  --test_size 0.2 \
  --outdir overall_UCI_CTG_NSPbin \
  --hparams_json '{
    "gate_expr_enable": true, "pop_size": 800, "ngen": 100,
    "include_lgo_multi": true, "include_lgo_and3": true, 
    "micro_mutation_prob": 0.10, "cv_proxy_weight": 0.0
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


conda activate pstree

python run_v3_8.py \
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


conda activate rils-rols

python run_v3_8.py \
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

# ./utility_analysis/

python utility_analysis/01_collect_hyperparams.py \
  --dataset_dir overall_UCI_CTG_NSPbin \
  --dataset UCI_CTG_NSPbin

python utility_analysis/02_collect_runtime_profile.py \
  --dataset_dir overall_UCI_CTG_NSPbin

# 	表达式与复杂度
python utility_analysis/03_export_expressions.py \
  --dataset_dir overall_UCI_CTG_NSPbin \
  --dataset UCI_CTG_NSPbin \
  --method all \
  --topk 10

python utility_analysis/04_gen_complexity_and_stability.py \
  --dataset_dir overall_UCI_CTG_NSPbin \
  --dataset UCI_CTG_NSPbin \
  --method all 

python utility_analysis/05_gen_pareto_front.py \
  --dataset_dir overall_UCI_CTG_NSPbin \
  --dataset UCI_CTG_NSPbin \
  --method all 

# lgo 专用：门使用频次（Top‑K）
python utility_analysis/06_gen_gating_usage.py \
  --dataset_dir overall_UCI_CTG_NSPbin \
  --dataset UCI_CTG_NSPbin \
  --method lgo 

# 阈值与审计
python utility_analysis/07_gen_thresholds_units.py \
  --dataset_dir overall_UCI_CTG_NSPbin \
  --dataset UCI_CTG_NSPbin \
  --method lgo --topk 10 --experiments lgo_hard

######################
  --ground_truth /home/dengou/data_UCI_Cardiotocography/output/ground_truth.json \
  --dataset UCI_CTG_NSPbin \
#######################

python utility_analysis/08_convert_ground_truth_to_guidelines.py \
  --ground_truth /home/dengou/data_UCI_Cardiotocography/output/ground_truth.json \
  --dataset UCI_CTG_NSPbin \
  --output config/guidelines.yaml

python utility_analysis/08_threshold_audit.py \
  --dataset_dir overall_UCI_CTG_NSPbin \
  --dataset UCI_CTG_NSPbin \
  --guidelines config/guidelines.yaml

# 消融与稳定性
python utility_analysis/09_build_ablation_matrix.py \
  --dataset_dir overall_UCI_CTG_NSPbin \
  --dataset UCI_CTG_NSPbin 

python utility_analysis/10_stability_summary.py \
  --dataset_dir overall_UCI_CTG_NSPbin \
  --dataset UCI_CTG_NSPbin 


(Option)

# 分析所有方法和实验
python utility_analysis/opt_gen_calibration.py \
  --dataset_dir overall_UCI_CTG_NSPbin \
  --dataset UCI_CTG_NSPbin  \
  --calibrator isotonic

# 只分析 lgo 方法的 base 实验
python utility_analysis/opt_gen_calibration.py \
  --dataset_dir overall_UCI_CTG_NSPbin \
  --dataset UCI_CTG_NSPbin  \
  --method lgo \
  --experiment base \
  --calibrator isotonic

# 只分析 lgo 方法的 lgo_thre 实验
python utility_analysis/opt_gen_calibration.py \
  --dataset_dir overall_UCI_CTG_NSPbin \
  --dataset UCI_CTG_NSPbin  \
  --method lgo \
  --experiment lgo_hard \
  --calibrator isotonic

# 比较多种校准方法
python utility_analysis/opt_gen_calibration.py \
  --dataset_dir overall_UCI_CTG_NSPbin \
  --dataset UCI_CTG_NSPbin  \
  --calibrator both

文件结构要求
overall_UCI_CTG_NSPbin/
├── predictions/
│   ├── test_predictions_lgo_base_seed1.csv
│   ├── test_predictions_lgo_base_seed2.csv
│   ├── test_predictions_lgo_lgo_seed1.csv
│   ├── test_predictions_lgo_lgo_thre_seed1.csv
│   ├── test_predictions_operon_base_seed1.csv
│   └── ...
└── aggregated/  # 自动创建
    ├── calibration_bins.csv
    └── calibration_ece.csv


## Platt calibration (requires y_score_raw)
python utility_analysis/opt_gen_calibration.py \
  --dataset_dir overall_UCI_CTG \
  --dataset UCI_CTG_NSPbin  \
  --method lgo \
  --calibrator platt \
  --cv_folds 3

## Isotonic calibration
python utility_analysis/opt_gen_calibration.py \
  --dataset_dir overall_UCI_CTG \
  --dataset UCI_CTG_NSPbin  \
  --method lgo \
  --calibrator isotonic \
  --cv_folds 3

## 新版本还支持更细粒度的控制：

# 只处理 lgo 方法的 base 实验
python utility_analysis/opt_gen_calibration.py \
  --dataset_dir overall_UCI_CTG \
  --dataset UCI_CTG_NSPbin  \
  --method lgo \
  --experiment base \
  --calibrator platt \
  --cv_folds 3

# 只处理 lgo 方法的 lgo_thre 实验
python opt_gen_calibration.py \
  --dataset_dir overall_UCI_CTG \
  --dataset UCI_CTG_NSPbin  \
  --method lgo \
  --experiment lgo_thre \
  --calibrator isotonic \
  --cv_folds 3

# 比较两种校准方法
python utility_analysis/opt_gen_calibration.py \
  --dataset_dir overall_UCI_CTG \
  --dataset UCI_CTG_NSPbin  \
  --method lgo \
  --calibrator both \
  --cv_folds 3
  ```
