
# Run all in Utility_analysis

## 01.
python utility_plots/01_median_performance.py \
--roots overall_NHANES_metabolic_score overall_ICU_composite_risk_score overall_UCI_CTG_NSPbin overall_UCI_Heart_Cleveland_num overall_UCI_HydraulicSys_fault_score 

## 02. 
python utility_plots/02_pareto_front.py \
--roots overall_NHANES_metabolic_score overall_ICU_composite_risk_score overall_UCI_CTG_NSPbin overall_UCI_Heart_Cleveland_num overall_UCI_HydraulicSys_fault_score 

(# if want see sigle csv: python 02_pareto_front.py --csv pareto_front.csv)

## 03.
python utility_plots/03_gating_usage.py \
--roots overall_NHANES_metabolic_score overall_ICU_composite_risk_score overall_UCI_CTG_NSPbin overall_UCI_Heart_Cleveland_num overall_UCI_HydraulicSys_fault_score 

## 04.
python utility_plots/04_complexity_distribution.py \
--roots overall_NHANES_metabolic_score overall_ICU_composite_risk_score overall_UCI_CTG_NSPbin overall_UCI_Heart_Cleveland_num overall_UCI_HydraulicSys_fault_score 

## 05.
python utility_plots/05_threshold_analysis.py \
--roots overall_NHANES_metabolic_score overall_ICU_composite_risk_score overall_UCI_CTG_NSPbin overall_UCI_Heart_Cleveland_num overall_UCI_HydraulicSys_fault_score 

## 06.
python utility_plots/06_stability_comparison.py \
--roots overall_NHANES_metabolic_score overall_ICU_composite_risk_score overall_UCI_CTG_NSPbin overall_UCI_Heart_Cleveland_num overall_UCI_HydraulicSys_fault_score 

## 07.
python utility_plots/07_ablation_heatmap.py \
--roots overall_NHANES_metabolic_score overall_ICU_composite_risk_score overall_UCI_CTG_NSPbin overall_UCI_Heart_Cleveland_num overall_UCI_HydraulicSys_fault_score 

## 08.
python utility_plots/08_runtime_efficiency.py \
--roots overall_NHANES_metabolic_score overall_ICU_composite_risk_score overall_UCI_CTG_NSPbin overall_UCI_Heart_Cleveland_num overall_UCI_HydraulicSys_fault_score 
