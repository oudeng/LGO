

这种“红点”特征的 sanity check 非常适合写进 Discussion，当成具体 case study。下面我直接给你 ICU / eICU / NHANES 三个数据集的一整套命令行，你可以按需挑选跑哪几个特征。

我按下面约定选了标签阈值（你后面可以改）：
	•	ICU：composite_risk_score ≥ 5 记为高危（中位数为 5）
	•	eICU：composite_risk_score ≥ 8 记为高危（中位数为 8）
	•	NHANES：metabolic_score ≥ 2 记为有代谢综合征风险（中位数为 2）

特征阈值全部来自 all_thresholds_summary.csv 的 guideline / median(LGO-hard)。

1. ICU_composite_risk_score

1.1 Lactate（红得最明显）
```bash
python utility_analysis/09_feature_threshold_sanity.py \
  --csv data/ICU/ICU_composite_risk_score.csv \
  --label_col composite_risk_score \
  --label_threshold 5 \
  --feature_col lactate_mmol_l \
  --guideline 2.0 \
  --lgo_threshold 4.7177 \
  --higher_is_riskier \
  --outdir utility_analysis/sanity_plots/ICU/lactate
  ```
	•	含义：用 score ≥ 5 作为高危，比较乳酸 2.0 vs 4.72 mmol/L 在 ROC 曲线上的位置。

1.2 Creatinine（偏高但没 lactate 那么极端）
```bash
python utility_analysis/09_feature_threshold_sanity.py \
  --csv data/ICU/ICU_composite_risk_score.csv \
  --label_col composite_risk_score \
  --label_threshold 5 \
  --feature_col creatinine_mg_dl \
  --guideline 1.2 \
  --lgo_threshold 1.7053 \
  --higher_is_riskier \
  --outdir utility_analysis/sanity_plots/ICU/creatinine
```

1.3 MAP（对齐很好的 positive case）
```bash
python utility_analysis/09_feature_threshold_sanity.py \
  --csv data/ICU/ICU_composite_risk_score.csv \
  --label_col composite_risk_score \
  --label_threshold 5 \
  --feature_col map_mmhg \
  --guideline 65.0 \
  --lgo_threshold 69.9475 \
  --outdir utility_analysis/sanity_plots/ICU/map_mmhg
```
注意这里 没有 --higher_is_riskier，默认规则是 “低于阈值更危险”（符合低血压危险）。

2. eICU_composite_risk_score

这里建议重点看：GCS / SpO₂ / HR_max / Lactate / Creatinine / RespRate_max。标签阈值我用的是 score ≥ 8。

2.1 GCS（典型红点）
```bash
python utility_analysis/09_feature_threshold_sanity.py \
  --csv data/eICU/eICU_composite_risk_score.csv \
  --label_col composite_risk_score \
  --label_threshold 8 \
  --feature_col gcs \
  --guideline 8.0 \
  --lgo_threshold 12.9158 \
  --outdir utility_analysis/sanity_plots/eICU/gcs
```

2.2 SpO₂_min（中度偏离）
```bash
python utility_analysis/09_feature_threshold_sanity.py \
  --csv data/eICU/eICU_composite_risk_score.csv \
  --label_col composite_risk_score \
  --label_threshold 8 \
  --feature_col spo2_min \
  --guideline 92.0 \
  --lgo_threshold 75.9974 \
  --outdir utility_analysis/sanity_plots/eICU/spo2_min
```

同样，默认“低饱和度更危险”，不加 --higher_is_riskier。

2.3 HR_max
```bash
python utility_analysis/09_feature_threshold_sanity.py \
  --csv data/eICU/eICU_composite_risk_score.csv \
  --label_col composite_risk_score \
  --label_threshold 8 \
  --feature_col hr_max \
  --guideline 100.0 \
  --lgo_threshold 79.9151 \
  --higher_is_riskier \
  --outdir utility_analysis/sanity_plots/eICU/hr_max
```

2.4 Lactate
```bash
python utility_analysis/09_feature_threshold_sanity.py \
  --csv data/eICU/eICU_composite_risk_score.csv \
  --label_col composite_risk_score \
  --label_threshold 8 \
  --feature_col lactate_mmol_l \
  --guideline 2.0 \
  --lgo_threshold 2.5166 \
  --higher_is_riskier \
  --outdir utility_analysis/sanity_plots/eICU/lactate
```

2.5 Creatinine
```bash
python utility_analysis/09_feature_threshold_sanity.py \
  --csv data/eICU/eICU_composite_risk_score.csv \
  --label_col composite_risk_score \
  --label_threshold 8 \
  --feature_col creatinine_mg_dl \
  --guideline 1.5 \
  --lgo_threshold 1.3239 \
  --higher_is_riskier \
  --outdir utility_analysis/sanity_plots/eICU/creatinine
```

2.6 Respiratory rate
```bash
python utility_analysis/09_feature_threshold_sanity.py \
  --csv data/eICU/eICU_composite_risk_score.csv \
  --label_col composite_risk_score \
  --label_threshold 8 \
  --feature_col resprate_max \
  --guideline 24.0 \
  --lgo_threshold 20.4686 \
  --higher_is_riskier \
  --outdir utility_analysis/sanity_plots/eICU/resprate_max
```

3. NHANES_metabolic_score

标签用 metabolic_score ≥ 2 作为“有代谢风险”。

重点建议看：TG（红） + BMI / Waist / Fasting glucose（绿）。

3.1 Triglycerides（最有争议的那个）
```bash
python utility_analysis/09_feature_threshold_sanity.py \
  --csv data/NHANES/NHANES_metabolic_score.csv \
  --label_col metabolic_score \
  --label_threshold 2 \
  --feature_col triglycerides \
  --guideline 150.0 \
  --lgo_threshold 29.9107 \
  --higher_is_riskier \
  --outdir utility_analysis/sanity_plots/NHANES/triglycerides
```
这里仍然设定 “高 TG 更危险”（--higher_is_riskier），这样 ROC 图上能直观看出：
guideline=150 和 LGO=30 分别对应的 (FPR,TPR) 在曲线上差异多大，从而支撑“模型最优 cut‑off 明显不同于 textbook 阈值”的论点。
如果后面想探索“低 TG 是否真的更危险”，可以再跑一版不加 --higher_is_riskier 的。

3.2 BMI
```bash
python utility_analysis/09_feature_threshold_sanity.py \
  --csv data/NHANES/NHANES_metabolic_score.csv \
  --label_col metabolic_score \
  --label_threshold 2 \
  --feature_col bmi \
  --guideline 25.0 \
  --lgo_threshold 27.2463 \
  --higher_is_riskier \
  --outdir utility_analysis/sanity_plots/NHANES/bmi
```

3.3 Fasting glucose
```bash
python utility_analysis/09_feature_threshold_sanity.py \
  --csv data/NHANES/NHANES_metabolic_score.csv \
  --label_col metabolic_score \
  --label_threshold 2 \
  --feature_col fasting_glucose \
  --guideline 100.0 \
  --lgo_threshold 99.0435 \
  --higher_is_riskier \
  --outdir utility_analysis/sanity_plots/NHANES/fasting_glucose
```

3.4 Waist circumference
```bash
python utility_analysis/09_feature_threshold_sanity.py \
  --csv data/NHANES/NHANES_metabolic_score.csv \
  --label_col metabolic_score \
  --label_threshold 2 \
  --feature_col waist_circumference \
  --guideline 88.0 \
  --lgo_threshold 92.7466 \
  --higher_is_riskier \
  --outdir utility_analysis/sanity_plots/NHANES/waist_circumference
```

4. 小提示（可以后面再微调）
	•	这里选的 label_threshold（5, 8, 2）是基于三套 score 的中位数，主要是为了把样本 roughly 二分；
如果你在设计 composite risk score 时有更“临床”的 cut‑off（例如 ICU 里 score≥7 才算高危），改成那个会更好。
	•	--higher_is_riskier 的逻辑：
	•	血压 / SpO₂ / GCS：低值更危险 → 不加这个 flag；
	•	乳酸 / 肌酐 / 心率 / 呼吸频率 / BMI / 腰围 / 血糖 / TG：高值更危险 → 加 --higher_is_riskier。

可以先跑完这些命令，看每个特征生成的两张图（分布 + ROC），然后挑 3–4 个最“故事性”的 case 写进 Discussion：
比如 ICU lactate、eICU GCS、NHANES triglycerides + BMI/腰围，基本就能撑起一段非常扎实的 case study。


