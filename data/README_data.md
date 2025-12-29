# LGO Benchmark Datasets

Version 1.0 — December 2025

[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

This repository contains curated datasets for evaluating **Logistic-Gated Operators (LGO)** and other interpretable machine learning methods. Each dataset includes clinically or domain-relevant threshold values (`ground_truth.json`) for validating discovered thresholds.

---

## Table of Contents

- [Overview](#overview)
- [Directory Structure](#directory-structure)
- [Datasets](#datasets)
  - [ICU (MIMIC-IV)](#icu-mimic-iv)
  - [eICU](#eicu)
  - [NHANES](#nhanes)
  - [UCI Machine Learning Repository](#uci-machine-learning-repository)
- [Ground Truth Files](#ground-truth-files)
- [Data Access Requirements](#data-access-requirements)
- [Citation](#citation)

---

## Overview

| Dataset | Domain | Samples | Features | Target | Task |
|---------|--------|---------|----------|--------|------|
| ICU (MIMIC-IV) | Healthcare/ICU | 2,939 | 20 | `composite_risk_score` | Binary (threshold=5) |
| eICU | Healthcare/ICU | 4,536 | 10 | `composite_risk_score` | Binary (threshold=5) |
| NHANES | Public Health | 2,281 | 12 | `metabolic_score` | Binary (threshold=5) |
| Heart Cleveland | Cardiology | 303 | 14 | `num` | Multi-class (0-4) |
| CTG | Obstetrics | 2,126 | 23 | `NSP_bin` | Binary |
| Hydraulic System | Engineering | 2,205 | 44 | `fault_score` | Regression |

---

## Directory Structure

```
data/
├── ICU/
│   ├── ICU_composite_risk_score.csv    # Extracted dataset (2,939 samples)
│   ├── ground_truth.json               # Clinical thresholds
│   └── mimic_extract_v7.py             # Extraction script (requires MIMIC-IV access)
│
├── eICU/
│   ├── eICU_composite_risk_score.csv   # Extracted dataset (4,536 samples)
│   ├── ground_truth.json               # Clinical thresholds
│   └── eicu_extract_v2_1.py            # Extraction script (requires eICU access)
│
├── NHANES/
│   ├── NHANES_metabolic_score.csv      # Processed dataset (2,281 samples)
│   ├── ground_truth.json               # Metabolic syndrome thresholds (NCEP ATP III)
│   └── fm_XPT_toCSV_v4_3.py            # Processing script
│
├── UCI/
│   ├── Heart_Cleveland_num.csv         # Heart disease dataset (303 samples)
│   ├── Heart_Cleveland_ground_truth.json
│   ├── CTG_nsp_bin.csv                 # Cardiotocography dataset (2,126 samples)
│   ├── CTG_ground_truth.json
│   ├── HydraulicSys_fault_score.csv    # Hydraulic system dataset (2,205 samples)
│   └── HydraulicSys_ground_truth.json
│
└── README_data.md                      # This document
```

---

## Datasets

### ICU (MIMIC-IV)

**Source:** MIMIC-IV v3.1 ([PhysioNet](https://physionet.org/content/mimiciv/3.1/))

**Description:** ICU mortality risk prediction dataset extracted from MIMIC-IV. Features include vital signs, laboratory values, and clinical scores relevant to ICU mortality prediction.

**File:** `ICU/ICU_composite_risk_score.csv`

| Feature | Unit | Description |
|---------|------|-------------|
| `map_mmhg` | mmHg | Mean Arterial Pressure |
| `lactate_mmol_l` | mmol/L | Serum lactate level |
| `creatinine_mg_dl` | mg/dL | Serum creatinine |
| `age_years` | years | Patient age |
| `gcs` | score | Glasgow Coma Scale (3-15) |
| `vasopressor_dose` | mcg/kg/min | Norepinephrine equivalent dose |
| `charlson_index` | score | Charlson Comorbidity Index |
| `hours_since_admission` | hours | Time since ICU admission |
| `sbp_min` | mmHg | Minimum systolic blood pressure |
| `dbp_min` | mmHg | Minimum diastolic blood pressure |
| `hr_max` | bpm | Maximum heart rate |
| `resprate_max` | /min | Maximum respiratory rate |
| `spo2_min` | % | Minimum SpO2 |
| `hemoglobin_min` | g/dL | Minimum hemoglobin |
| `sodium_min` | mEq/L | Minimum sodium |
| `urine_output_min` | mL/kg/hr | Minimum urine output |
| `mechanical_ventilation_std` | 0/1 | Mechanical ventilation (binary) |
| `vasopressor_use_std` | 0/1 | Vasopressor use (binary) |
| `age_band` | category | Age group (0-4) |
| `gender_std` | 0/1 | Gender (0=Female, 1=Male) |
| `composite_risk_score` | 0-14 | **Target:** Composite risk score |

**Key Clinical Thresholds (ground_truth.json):**
- MAP < 65 mmHg (hypotension)
- Lactate > 2.0 / 4.0 mmol/L (tissue hypoperfusion)
- GCS ≤ 8 / 12 (altered consciousness)
- SpO2 < 92% (hypoxemia)
- Creatinine > 1.5 / 2.0 mg/dL (AKI)

---

### eICU

**Source:** eICU Collaborative Research Database ([PhysioNet](https://physionet.org/content/eicu-crd/))

**Description:** External validation dataset from the eICU database with similar clinical variables to MIMIC-IV ICU dataset.

**File:** `eICU/eICU_composite_risk_score.csv`

| Feature | Unit | Description |
|---------|------|-------------|
| `map_mmhg` | mmHg | Mean Arterial Pressure |
| `lactate_mmol_l` | mmol/L | Serum lactate level |
| `creatinine_mg_dl` | mg/dL | Serum creatinine |
| `resprate_max` | /min | Maximum respiratory rate |
| `hr_max` | bpm | Maximum heart rate |
| `spo2_min` | % | Minimum SpO2 |
| `gcs` | score | Glasgow Coma Scale |
| `vasopressor_use_std` | 0/1 | Vasopressor use (binary) |
| `mechanical_ventilation_std` | 0/1 | Mechanical ventilation (binary) |
| `composite_risk_score` | 0-14 | **Target:** Composite risk score |

---

### NHANES

**Source:** NHANES 2017-2018 (Cycle J) ([CDC](https://wwwn.cdc.gov/nchs/nhanes/continuousnhanes/default.aspx?BeginYear=2017))

**Description:** National Health and Nutrition Examination Survey data for metabolic syndrome risk assessment. Thresholds based on NCEP ATP III and IDF criteria.

**File:** `NHANES/NHANES_metabolic_score.csv`

| Feature | Unit | Description |
|---------|------|-------------|
| `waist_circumference` | cm | Waist circumference |
| `systolic_bp` | mmHg | Systolic blood pressure (average of 3 readings) |
| `diastolic_bp` | mmHg | Diastolic blood pressure (average of 3 readings) |
| `triglycerides` | mg/dL | Fasting triglycerides |
| `hdl_cholesterol` | mg/dL | HDL cholesterol |
| `fasting_glucose` | mg/dL | Fasting plasma glucose |
| `age` | years | Age |
| `bmi` | kg/m² | Body Mass Index |
| `hba1c` | % | Glycated hemoglobin |
| `gender_std` | 0/1 | Gender (0=Female, 1=Male) |
| `age_band` | category | Age group |
| `metabolic_score` | 0-5+ | **Target:** Metabolic syndrome component count |

**Key Clinical Thresholds (ground_truth.json):**
- Waist: ≥102 cm (M) / ≥88 cm (F) — NCEP ATP III
- Triglycerides: ≥150 mg/dL
- HDL: <40 mg/dL (M) / <50 mg/dL (F)
- Blood pressure: ≥130/85 mmHg
- Fasting glucose: ≥100 mg/dL (prediabetes), ≥126 mg/dL (diabetes)
- BMI: ≥25 (overweight), ≥30 (obese) — WHO
- HbA1c: ≥5.7% (prediabetes), ≥6.5% (diabetes) — ADA

---

### UCI Machine Learning Repository

#### Heart Disease (Cleveland)

**Source:** UCI ML Repository ([Link](https://archive.ics.uci.edu/dataset/45/heart+disease))

**File:** `UCI/Heart_Cleveland_num.csv`

| Feature | Description |
|---------|-------------|
| `age` | Age in years |
| `sex` | Sex (1=male, 0=female) |
| `cp` | Chest pain type (1-4) |
| `trestbps` | Resting blood pressure (mmHg) |
| `chol` | Serum cholesterol (mg/dL) |
| `fbs` | Fasting blood sugar > 120 mg/dL |
| `restecg` | Resting ECG results (0-2) |
| `thalach` | Maximum heart rate achieved |
| `exang` | Exercise-induced angina |
| `oldpeak` | ST depression induced by exercise |
| `slope` | Slope of peak exercise ST segment |
| `ca` | Number of major vessels colored by fluoroscopy |
| `thal` | Thalassemia (3=normal, 6=fixed defect, 7=reversible defect) |
| `num` | **Target:** Diagnosis (0=no disease, 1-4=disease severity) |

---

#### Cardiotocography (CTG)

**Source:** UCI ML Repository ([Link](https://archive.ics.uci.edu/dataset/193/cardiotocography))

**Description:** Fetal heart rate and uterine contraction measurements for fetal state classification.

**File:** `UCI/CTG_nsp_bin.csv`

| Feature | Description |
|---------|-------------|
| `LB` | Baseline fetal heart rate (bpm) |
| `ASTV` | % of time with abnormal short-term variability |
| `MSTV` | Mean short-term variability |
| `ALTV` | % of time with abnormal long-term variability |
| `MLTV` | Mean long-term variability |
| `DL`, `DS`, `DP` | Light/Severe/Prolonged decelerations |
| `AC`, `FM`, `UC` | Accelerations, Fetal movements, Uterine contractions |
| `Width`, `Min/Max/Mode/Mean/MedianHist` | FHR histogram features |
| `NSP` | Fetal state (1=Normal, 2=Suspect, 3=Pathologic) |
| `NSP_bin` | **Target:** Binary (0=Normal, 1=Suspect/Pathologic) |

**Key Clinical Thresholds:**
- Baseline FHR: 110-160 bpm (normal range)
- ASTV: <20% (normal)
- Decelerations: 0 (normal)

---

#### Hydraulic System

**Source:** UCI ML Repository ([Link](https://archive.ics.uci.edu/dataset/447/condition+monitoring+of+hydraulic+systems))

**Description:** Condition monitoring data from a hydraulic test rig with sensor measurements for fault detection.

**File:** `UCI/HydraulicSys_fault_score.csv`

| Feature Group | Description |
|---------------|-------------|
| `PS1-PS6` | Pressure sensors (bar) — mean, std, quantiles |
| `EPS1` | Motor power (W) — mean, quantiles |
| `FS1-FS2` | Flow sensors (L/min) |
| `TS1-TS4` | Temperature sensors (°C) |
| `VS1` | Vibration sensor (mm/s) |
| `CE` | Cooling efficiency (%) |
| `CP` | Cooling power (kW) |
| `SE` | System efficiency (%) |
| `fault_score` | **Target:** Composite fault severity (continuous) |

---

## Ground Truth Files

Each dataset includes a `ground_truth.json` file containing clinically or domain-established threshold values. These serve as reference points for evaluating whether LGO-discovered thresholds align with expert knowledge.

**Format:**
```json
{
  "feature_name": {
    "values": [threshold1, threshold2],
    "source": "Reference guideline",
    "note": "Clinical interpretation"
  }
}
```

**Example (NHANES):**
```json
{
  "fasting_glucose": {
    "values": [100.0, 126.0],
    "source": "IDF / ADA",
    "note": "≥100 mg/dL (prediabetes), ≥126 mg/dL (diabetes)"
  }
}
```

---

## Data Access Requirements

| Dataset | Access | Requirements |
|---------|--------|--------------|
| ICU (MIMIC-IV) | Credentialed | PhysioNet credentialing + CITI training |
| eICU | Credentialed | PhysioNet credentialing + CITI training |
| NHANES | Public | None (CDC public data) |
| UCI datasets | Public | None |

**Note:** The CSV files in this repository are pre-extracted/processed. To regenerate from source data:

1. **MIMIC-IV:** Obtain access at [PhysioNet](https://physionet.org/content/mimiciv/), then run `mimic_extract_v7.py`
2. **eICU:** Obtain access at [PhysioNet](https://physionet.org/content/eicu-crd/), then run `eicu_extract_v2_1.py`
3. **NHANES:** Download XPT files from [CDC](https://wwwn.cdc.gov/nchs/nhanes/), then run `fm_XPT_toCSV_v4_3.py`

---

## Citation

If you use these datasets, please cite the original sources:

**MIMIC-IV:**
```bibtex
@article{johnson2023mimic,
  title={MIMIC-IV, a freely accessible electronic health record dataset},
  author={Johnson, Alistair EW and others},
  journal={Scientific Data},
  volume={10},
  pages={1},
  year={2023}
}
```

**eICU:**
```bibtex
@article{pollard2018eicu,
  title={The eICU Collaborative Research Database},
  author={Pollard, Tom J and others},
  journal={Scientific Data},
  volume={5},
  pages={180178},
  year={2018}
}
```

**NHANES:**
```bibtex
@misc{nhanes2017,
  title={National Health and Nutrition Examination Survey},
  author={{Centers for Disease Control and Prevention}},
  year={2017-2018},
  url={https://wwwn.cdc.gov/nchs/nhanes/}
}
```

**UCI Datasets:**
```bibtex
@misc{uci_ml_repository,
  author={Dua, Dheeru and Graff, Casey},
  title={{UCI} Machine Learning Repository},
  year={2017},
  url={http://archive.ics.uci.edu/ml}
}
```

---

## License

- **MIMIC-IV / eICU:** PhysioNet Credentialed Health Data License
- **NHANES:** Public domain (U.S. Government work)
- **UCI datasets:** CC BY 4.0 (see individual dataset pages)
- **Extraction scripts:** MIT License