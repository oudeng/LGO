
How to plot the results

still stay at ```(py310) dengou@eis06:~/goGitHub/LGO_v3_7$ ```

# 使用数据集根目录
python utility_plots/01_median_performance_violin.py \
--roots overall_NHANES_metabolic_score overall_ICU_composite_risk_score overall_UCI_CTG_NSPbin overall_UCI_Heart_Cleveland_num overall_UCI_HydraulicSys_fault_score 

(bak: barplot)
python utility_plots/01_median_performance_barplot.py \
--roots overall_NHANES_metabolic_score overall_ICU_composite_risk_score overall_UCI_CTG_NSPbin overall_UCI_Heart_Cleveland_num overall_UCI_HydraulicSys_fault_score 


python utility_plots/02_pareto_front.py \
--roots overall_NHANES_metabolic_score overall_ICU_composite_risk_score overall_UCI_CTG_NSPbin overall_UCI_Heart_Cleveland_num overall_UCI_HydraulicSys_fault_score 

(# 使用单个CSV文件 python 02_pareto_front.py --csv pareto_front.csv)



python utility_plots/03_gating_usage.py \
--roots overall_NHANES_metabolic_score overall_ICU_composite_risk_score overall_UCI_CTG_NSPbin overall_UCI_Heart_Cleveland_num overall_UCI_HydraulicSys_fault_score 



## For publication figures (top features only):
python utility_plots/04_thresholds_v3_7.py \
  --dataset_dirs overall_ICU_composite_risk_score overall_NHANES_metabolic_score \
  --config_dir config \
  --method lgo --experiment lgo_hard \
  --only_anchored \
  --topk 5 \
  --annotate \
  --outdir utility_plots/figs/publication37


python  utility_plots/04_thresholds_plot.py \
  --csv utility_plots/figs/publication37/v3_thresholds_summary.csv \
  --outdir utility_plots/figs/publication37




# 只显示有锚点的特征（去掉灰色 N/A 行）
python utility_plots/04_thresholds.py \
  --dataset_dirs overall_NHANES_metabolic_score overall_ICU_composite_risk_score overall_UCI_CTG_NSPbin overall_UCI_Heart_Cleveland_num overall_UCI_HydraulicSys_fault_score \
  --config_dir config \
  --method lgo \
  --experiment lgo_hard \
  --only_anchored \
  --annotate

# 指定特征白名单（名称自动归一化；例如 ICU/NHANES 的关键指标）
python utility_plots/04_thresholds.py \
  --dataset_dirs overall_NHANES_metabolic_score overall_ICU_composite_risk_score overall_UCI_CTG_NSPbin overall_UCI_Heart_Cleveland_num overall_UCI_HydraulicSys_fault_score \
  --config_dir config \
  --method lgo --experiment lgo_hard \
  --feature_whitelist spo2_min sbp_min lactate_mmol_l systolic_bp fasting_glucose triglycerides \
  --annotate




# Thresholds v3 0921

python utility_plots/04_thresholds_v3.py \
  --dataset_dirs overall_ICU_composite_risk_score overall_NHANES_metabolic_score \
                overall_UCI_HydraulicSys_fault_score \
                 overall_UCI_Heart_Cleveland_num \
                 overall_UCI_CTG_NSPbin \
  --config_dir config \
  --method lgu --experiment lgu_hard \
  --only_anchored \
  --abs_band 0.5 \
  --annotate \
  --topk 5 \
  --outdir utility_plots/figs/thresholds_v3_clean

## For publication figures (top features only):
python utility_plots/04_thresholds_v3_7.py \
  --dataset_dirs overall_ICU_composite_risk_score overall_NHANES_metabolic_score \
  --config_dir config \
  --method lgu --experiment lgu_hard \
  --only_anchored \
  --topk 5 \
  --annotate \
  --outdir utility_plots/figs/publication37


python  utility_plots/04_thresholds_plot.py \
  --csv utility_plots/figs/publication37/v3_thresholds_summary.csv \
  --outdir utility_plots/figs/publication37









###############################
## Basic usage with clean output:
python utility_plots/04_thresholds_v3.py \
  --dataset_dirs overall_ICU_composite_risk_score overall_NHANES_metabolic_score \
  --config_dir config \
  --method lgu --experiment lgu_hard \
  --annotate \
  --outdir utility_plots/figs/thresholds_v3_basic

## complete usage
python utility_plots/04_thresholds_v3.py \
  --dataset_dirs overall_ICU_composite_risk_score overall_NHANES_metabolic_score \
                 overall_UCI_HydraulicSys_fault_score \
                 overall_UCI_Heart_Cleveland_num \
                 overall_UCI_CTG_NSPbin \
  --config_dir config \
  --method lgu --experiment lgu_hard \
  --abs_band 0.5 \
  --draw_coverage \
  --annotate \
  --topk 10 \
  --outdir utility_plots/figs/thresholds_v3_final

## Full-featured analysis:
python utility_plots/04_thresholds_v3.py \
  --dataset_dirs overall_ICU_composite_risk_score overall_NHANES_metabolic_score \
                 overall_UCI_HydraulicSys_fault_score \
  --config_dir config \
  --method lgu --experiment lgu_hard \
  --abs_band 0.5 \
  --drop_tokens add sub mul div zero one unknown lgu_thre \
  --draw_coverage \
  --topk 10 \
  --annotate \
  --outdir utility_plots/figs/thresholds_v3_full

## For publication figures (top features only):
python utility_plots/04_thresholds_v3.py \
  --dataset_dirs overall_ICU_composite_risk_score overall_NHANES_metabolic_score \
  --config_dir config \
  --method lgu --experiment lgu_hard \
  --only_anchored \
  --topk 5 \
  --annotate \
  --draw_coverage \
  --outdir utility_plots/figs/publication


