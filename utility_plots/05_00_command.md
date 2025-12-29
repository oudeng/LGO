

# 运行完整分析流程

方式1：直接传入目录（推荐）

bash utility_plots/05_00_run_all_analysis.sh \
  overall_ICU_composite_risk_score \
  overall_eICU_composite_risk_score \
  overall_NHANES_metabolic_score \
  overall_UCI_CTG_NSPbin \
  overall_UCI_Heart_Cleveland_num \
  overall_UCI_HydraulicSys_fault_score \
  --outdir utility_plots/figs/05threshold_analysis

方式2：使用 --roots 参数
bash utility_plots/05_00_run_all_analysis.sh \
  --roots overall_ICU_composite_risk_score \
          overall_eICU_composite_risk_score \
          overall_NHANES_metabolic_score \
          overall_UCI_CTG_NSPbin \
          overall_UCI_Heart_Cleveland_num \
          overall_UCI_HydraulicSys_fault_score \
  --outdir utility_plots/05threshold_analysis

方式3：自动查找（如果目录都在当前目录下）
bash utility_plots/05_00_run_all_analysis.sh \
  --data_root . \
  --outdir utility_plots/05threshold_analysis
