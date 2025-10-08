##Script Structure

## Output File Structure

```bash
overall_dataset_name/
├── candidates/
│   ├── candidates_{method}_{experiment}_seed{N}.csv
│   └── ...
├── aggregated/
│   ├── overall_metrics.csv          # Performance metrics
│   ├── expressions.csv              # Expression summary
│   ├── selfcheck_report.csv         # self-checks
│   ├── hyperparams.csv              # From 01
│   ├── runtime_profile.csv          # From 02
│   ├── complexity_by_model.csv      # From 04
│   ├── complexity_stats.csv         # From 04
│   ├── pareto_front.csv             # From 05
│   ├── gating_usage.csv             # From 06
│   ├── thresholds_units.csv         # From 07
│   ├── ablation_table.csv           # From 09
│   ├── threshold_audit.csv          # From 08
│   ├── threshold_audit_summary.csv  # From 08
│   ├── stability_summary.csv        # From 10
│   ├── calibration_bins.csv         # From opt_gen_calibration
│   └── calibration_ece.csv          # From opt_gen_calibration
├── expressions/
│   ├── topk_expressions.csv         # From 03
│   └── top1_expressions.txt         # From 03
├── predictions/                     # If --save_predictions used
│   └── test_predictions_{method}_{experiment}_seed{N}.csv
└── config/
    ├── scaler.json
    ├── units.yaml
    └── guidelines.yaml              # Created by 08_convert_ground_truth

```