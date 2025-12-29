

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


--------------------------------------------------------------------------------
再补充 CTG，Cleveland和HydraulicSys的Sainity结果

	1.	要不要补 CTG/Cleveland/Hydraulic 的 sanity？
建议：要补，而且放在 SI 里最合适。主文依然把 ICU + eICU（再加 NHANES）当主角，UCI 三个数据集更多是“证明方法一般性 + 展示 failure/平滑场景”的辅助材料。
	2.	主文要不要改表述？
是的，可以在 Results 和 Discussion 里稍微改一下话术：
	•	明确 “本研究的 primary datasets 是 ICU/eICU（critical care）+ NHANES（人群心代谢风险）”；
	•	CTG/Cleveland/Hydraulic 作为 secondary benchmarks 与方法学验证，详细结果见 SI。 ￼

下面重点给你要的东西：CTG / Cleveland / Hydraulic 的 09_feature_threshold_sanity.py 命令行，风格和前面 ICU/eICU/NHANES 那套完全一致。你后面可以把这些画出来放在 SI 的附图里。

说明：
	•	--csv 路径和 --label_col / --feature_col 名字，需要和你自己的 CSV 一致；我下面按“常见预处理命名”来写，如果列名不同，改一下就好。
	•	阈值（guideline / lgo_threshold）用的是你之前汇总的阈值表（all_thresholds_summary）里的中位数，保留到 2 位小数左右就够了。


一、CTG：UCI_CTG_NSPbin（胎心监护）

CTG 对你这篇医学期刊来说是“非临床主战场”，但可以在 SI 里展示：
	•	一个“正常”的特征（MeanHist，轻微偏离）；
	•	一个“锚点本身就奇怪”的特征（FM，anchor=0，LGO 学到负数）。

1.1 MeanHist（“正常”例子）

假设：
	•	CSV：data/CTG/UCI_CTG_NSPbin.csv
	•	标签列（二分类）：NSPbin（0/1）
	•	特征列：MeanHist（胎心直方图均值）
	•	阈值（来自你之前的表）：
	•	guideline ≈ 135
	•	LGO 中位阈值 ≈ 117
	•	理解：数值越大越异常 → 用 --higher_is_riskier
```bash
python utility_analysis/09_feature_threshold_sanity.py \
  --csv data/UCI/CTG_nsp_bin.csv \
  --label_col NSP_bin \
  --feature_col MeanHist \
  --guideline 135.0 \
  --lgo_threshold 117.0 \
  --higher_is_riskier \
  --outdir utility_analysis/sanity_plots/CTG/MeanHist
```

1.2 FM（fetal movement，明显“锚点失效”的例子，可选）
	•	特征列：FM
	•	guideline：0
	•	LGO 阈值：≈ -48

这个例子本身“物理意义就不太对”，你可以在 SI 里拿来说明：
	•	guideline 设成 0（有/无胎动），
	•	但模型在标准化空间学到的阈值反映的是编码/预处理的问题，而不是有意义的医学 cut‑off。

命令行（方向先按“高一点更危险”来跑一版，主要看 ROC 点位置，供 SI 里吐槽）：
```bash
python utility_analysis/09_feature_threshold_sanity.py \
  --csv data/UCI/CTG_nsp_bin.csv \
  --label_col NSP_bin \
  --feature_col FM \
  --guideline 0.0 \
  --lgo_threshold -48.0 \
  --higher_is_riskier \
  --outdir utility_analysis/sanity_plots/CTG/FM
```

二、Cleveland：UCI_Heart_Cleveland_num（心脏病评分）

Cleveland 对临床审稿人来说：
	•	有一点医学相关性（血压、胸痛类型等），
	•	但远不如 ICU/eICU；所以主文带一两句概述就够，细节放 SI。

建议 sanity 做三个特征：
	1.	cp（chest pain type，基本对齐 anchor）
	2.	trestbps（静息血压，几乎对齐）
	3.	oldpeak（偏差大的红点，说明“特征编码的局限”）
（slope 那个负阈值太怪，可以只在 SI 里略提）

假设：
	•	CSV：data/Cleveland/UCI_Heart_Cleveland_num.csv
	•	标签列：num（0–4 或已二值化）
	•	如果你已经在预处理里把它变成 0/1，就不用 --label_threshold；
	•	如果还是 0–4，可以设 --label_threshold 1，用 “num ≥ 1” 当有病。
	•	特征列：cp, trestbps, oldpeak, slope

2.1 cp（chest pain type，正例）

阈值（来自之前分析）：
	•	guideline：3
	•	LGO 阈值：≈ 2.81
	•	方向：类型编码越大越“危险” → 用 --higher_is_riskier
```bash
python utility_analysis/09_feature_threshold_sanity.py \
  --csv data/UCI/Heart_Cleveland_num.csv \
  --label_col num \
  --label_threshold 1 \
  --feature_col cp \
  --guideline 3.0 \
  --lgo_threshold 2.81 \
  --higher_is_riskier \
  --outdir utility_analysis/sanity_plots/Cleveland/cp
```

2.2 trestbps（静息血压，正例）
	•	guideline：140 mmHg
	•	LGO 阈值：≈ 143.99
```bash
python utility_analysis/09_feature_threshold_sanity.py \
  --csv data/UCI/Heart_Cleveland_num.csv \
  --label_col num \
  --label_threshold 1 \
  --feature_col trestbps \
  --guideline 140.0 \
  --lgo_threshold 143.99 \
  --higher_is_riskier \
  --outdir utility_analysis/sanity_plots/Cleveland/trestbps
```

2.3 oldpeak（明显偏离的红点）
	•	guideline：1.0
	•	LGO 阈值：≈ 1.86
```bash
python utility_analysis/09_feature_threshold_sanity.py \
  --csv data/UCI/Heart_Cleveland_num.csv \
  --label_col num \
  --label_threshold 1 \
  --feature_col oldpeak \
  --guideline 1.0 \
  --lgo_threshold 1.86 \
  --higher_is_riskier \
  --outdir utility_analysis/sanity_plots/Cleveland/oldpeak
```

2.4 （可选）slope：编码+负阈值的“坏例子”
	•	guideline：2
	•	LGO 阈值：≈ -4.07

```bash
python utility_analysis/09_feature_threshold_sanity.py \
  --csv data/UCI/Heart_Cleveland_num.csv \
  --label_col num \
  --label_threshold 1 \
  --feature_col slope \
  --guideline 2.0 \
  --lgo_threshold -4.07 \
  --higher_is_riskier \
  --outdir utility_analysis/sanity_plots/Cleveland/slope
```
这个结果你可以在 SI 里用一句话点出：

“如 Cleveland 的 slope 所示，编码特征在自然单位下没有有意义的常数 anchor，LGO 学到的阈值也缺乏可解释性，因此被我们标记为不适合做 anchor‑based audit。”

三、Hydraulic：UCI_HydraulicSys_fault_score（工业系统）

Hydraulic 虽然不是临床，但特别适合在 SI 里当“工程正例”：
	•	关系非常平滑，
	•	LGO 的阈值几乎和工程设定阈值重合。

假设：
	•	CSV：data/Hydraulic/UCI_HydraulicSys_fault_score.csv
	•	标签（连续故障评分）：fault_score
	•	可以选一个阈值（比如 0.5）把它二值化成“严重 vs 非严重”；你可以按你构造的 score 再微调，这里先用 0.5 作为示例。
	•	特征：PS1_q50，SE_mean

3.1 PS1_q50（完美对齐）
	•	guideline：156.305
	•	LGO 阈值：156.358

```bash
python utility_analysis/09_feature_threshold_sanity.py \
  --csv data/UCI/HydraulicSys_fault_score.csv \
  --label_col fault_score \
  --label_threshold 0.5 \
  --feature_col PS1_q50 \
  --guideline 156.305 \
  --lgo_threshold 156.358 \
  --higher_is_riskier \
  --outdir utility_analysis/sanity_plots/Hydraulic/PS1_q50
```

3.2 SE_mean（轻微偏差）
	•	guideline：59.6569
	•	LGO 阈值：57.88
```bash
python utility_analysis/09_feature_threshold_sanity.py \
  --csv data/UCI/HydraulicSys_fault_score.csv \
  --label_col fault_score \
  --label_threshold 0.5 \
  --feature_col SE_mean \
  --guideline 59.6569 \
  --lgo_threshold 57.88 \
  --higher_is_riskier \
  --outdir utility_analysis/sanity_plots/Hydraulic/SE_mean
```

四、主文表述怎么改更“临床向”

结合现在的结果和这三套 sanity，我建议你在主文做两类小修改（幅度不大）：
	1.	在 Results 开头 / Summary 里加一句：
“Our primary focus is on two ICU cohorts (ICU and eICU) and a population cardiometabolic dataset (NHANES); additional UCI benchmarks (CTG, Cleveland, Hydraulic) are included to test generality and appear in the Supplementary Information.”
	2.	在 Discussion 的 case‑study 小节里，把 ICU + eICU 明确写成重点：
	•	先讲 ICU lactate / MAP，eICU GCS / SpO₂ 的故事；
	•	NHANES 腰围 / TG 作为“人群风险”的延伸；
	•	最后一小段用 1–2 句提到：
	•	CTG：几乎可分，LGO 阈值对 MeanHist 还算合理，但 FM 说明 anchor 设计的问题；
	•	Cleveland：cp/trestbps 对齐，oldpeak/slope 受限于编码；
	•	Hydraulic：平滑任务上 LGO 阈值与工程设定高度一致，但 gates 对提升预测帮助有限。

这一套下来，ICU/eICU 在主文显然是“C 位”，CTG/Cleveland/Hydraulic 的 sanity 图和故事都放在 SI，既回应了审稿人对 ICU/eICU 的关注，又展现了方法在非临床场景的行为模式。

如果你跑完这三组命令，把关键几张图选出来，我也可以帮你把 SI 的那一小节英文段落（针对 CTG/Cleveland/Hydraulic 的解释）写好，和之前 ICU/eICU/NHANES 那一节风格保持一致。