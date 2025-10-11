
# How to plot the results

## Use datasets root results
```bash
python utility_plots/01_median_performance_violin.py \
--roots overall_NHANES_metabolic_score overall_ICU_composite_risk_score overall_UCI_CTG_NSPbin overall_UCI_Heart_Cleveland_num overall_UCI_HydraulicSys_fault_score 

(bak: barplot)
python utility_plots/01_median_performance_barplot.py \
--roots overall_NHANES_metabolic_score overall_ICU_composite_risk_score overall_UCI_CTG_NSPbin overall_UCI_Heart_Cleveland_num overall_UCI_HydraulicSys_fault_score 


python utility_plots/02_pareto_front.py \
--roots overall_NHANES_metabolic_score overall_ICU_composite_risk_score overall_UCI_CTG_NSPbin overall_UCI_Heart_Cleveland_num overall_UCI_HydraulicSys_fault_score 

(### if want single csv: python 02_pareto_front.py --csv pareto_front.csv)

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

## (Option) -- only_anchored for anchored features only, without N/A ones.
python utility_plots/04_thresholds.py \
  --dataset_dirs overall_NHANES_metabolic_score overall_ICU_composite_risk_score overall_UCI_CTG_NSPbin overall_UCI_Heart_Cleveland_num overall_UCI_HydraulicSys_fault_score \
  --config_dir config \
  --method lgo \
  --experiment lgo_hard \
  --only_anchored \
  --annotate

## (option) --feature_whitelist for specific features (e.g.: ICU/NHANES key features)
python utility_plots/04_thresholds.py \
  --dataset_dirs overall_NHANES_metabolic_score overall_ICU_composite_risk_score overall_UCI_CTG_NSPbin overall_UCI_Heart_Cleveland_num overall_UCI_HydraulicSys_fault_score \
  --config_dir config \
  --method lgo --experiment lgo_hard \
  --feature_whitelist spo2_min sbp_min lactate_mmol_l systolic_bp fasting_glucose triglycerides \
  --annotate

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
```
