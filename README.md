# 🏥 Diabetes Prediction Challenge - Playground Series S5E12

<div align="center">

![Python](https://img.shields.io/badge/Python-3.12-blue.svg)
![XGBoost](https://img.shields.io/badge/XGBoost-3.1.0-green.svg)
![CUDA](https://img.shields.io/badge/CUDA-Enabled-brightgreen.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

**Advanced Machine Learning Solution for Diabetes Risk Prediction**

[🔗 Kaggle Competition](https://www.kaggle.com/competitions/playground-series-s5e12) | [📝 Notebook (AutoBlend)](https://www.kaggle.com/code/rishabhkannaujiya/s5e12-xgb-lgbm-cat-drift-weights-auto-blend) | [📝 Notebook (Base)](https://www.kaggle.com/code/rishabhkannaujiya/s5e12-xgboost-drift-correction)

**🏆 Leaderboard Performance**
- **Public Score:** 0.70487
- **Private Score:** 0.70139
- **Final Ranking:** 388/4206 (Top 10%)

</div>

---

## 🎯 Overview

This repository contains a comprehensive solution for the **Kaggle Playground Series S5E12** competition, focusing on predicting diabetes diagnosis using advanced machine learning techniques. The solution leverages **XGBoost** with sophisticated drift correction strategies, parallel GPU training, and extensive feature engineering.

### 🏆 Competition Goal
Predict whether a patient has been diagnosed with diabetes based on various health metrics, demographic information, and lifestyle factors.

---

## ✨ Key Features

- 🚀 **GPU-Accelerated Training**: Dual GPU parallel processing for faster model training
- 📊 **Drift Analysis & Correction**: Comprehensive drift visualization and sample weighting
- 🔬 **Advanced Feature Engineering**: 
  - Medical risk ratios (LDL/HDL, Triglycerides/HDL)
  - Blood pressure metrics (Pulse Pressure, Mean Arterial Pressure)
  - Lifestyle interaction terms
  - Quantile-based binning
- 🎯 **Hyperparameter Optimization**: Optuna-tuned XGBoost parameters
- 🔄 **Stratified K-Fold CV**: Robust 5-fold cross-validation strategy
- 📈 **Ensemble Methods**: Multi-model blending (XGBoost, LightGBM, CatBoost)

---

## 🏥 Competition Details

**[Playground Series Season 5, Episode 12](https://www.kaggle.com/competitions/playground-series-s5e12)**

- **Type**: Binary Classification
- **Evaluation Metric**: Area Under ROC Curve (AUC)
- **Training Data**: 700,000 samples
- **Test Data**: 300,000 samples
- **Features**: 25 features including demographics, vitals, lab results, and lifestyle factors

### 📊 Dataset Features

<details>
<summary><b>Click to expand feature categories</b></summary>

#### Demographics & Socioeconomic
- Age, Gender, Ethnicity
- Education Level, Income Level
- Employment Status

#### Health Metrics
- BMI, Waist-to-Hip Ratio
- Systolic/Diastolic Blood Pressure
- Heart Rate

#### Laboratory Results
- Total Cholesterol, HDL, LDL
- Triglycerides

#### Lifestyle Factors
- Physical Activity (minutes/week)
- Sleep Hours, Screen Time
- Alcohol Consumption
- Smoking Status
- Diet Score

#### Medical History
- Family History of Diabetes
- Hypertension History
- Cardiovascular History

</details>

---

## 🧠 Approach

### 1️⃣ Data Analysis & Drift Detection

```python
# Drift visualization showing temporal distribution shifts
plt.plot(train_df['id'], 
         train_df['physical_activity_minutes_per_week'].rolling(window=5000).mean())
```

Our analysis revealed significant **temporal drift** in key features, particularly physical activity levels, which guided our sample weighting strategy.

### 2️⃣ Feature Engineering

```python
# Medical Risk Indicators
df["ldl_hdl_ratio"] = df["ldl_cholesterol"] / (df["hdl_cholesterol"] + 1e-6)
df["pulse_pressure"] = df["systolic_bp"] - df["diastolic_bp"]
df["metabolic_risk_score"] = df["bmi"] * df["ldl_hdl_ratio"]
```

### 3️⃣ Model Architecture

**XGBoost Classifier** with Optuna-optimized hyperparameters:

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `max_depth` | 5 | Control overfitting |
| `learning_rate` | 0.0132 | Fine-tuned convergence |
| `subsample` | 0.822 | Bootstrap sampling |
| `colsample_bytree` | 0.533 | Feature sampling |
| `reg_alpha` | 9.75 | L1 regularization |
| `reg_lambda` | 2.75 | L2 regularization |

### 4️⃣ Training Strategy

- **Stratified 5-Fold Cross-Validation**
- **Early Stopping** (50-200 rounds patience)
- **Parallel GPU Training** (2x NVIDIA Tesla T4)
- **Sample Weighting** for drift correction

---

## 📊 Results

### Competition Performance

| Metric | Score |
|--------|-------|
| **Public Leaderboard** | 0.70487 |
| **Private Leaderboard** | 0.70139 |
| **Final Ranking** | 388/4206 |
| **Percentile** | Top 10% |

### Cross-Validation Performance

| Metric | Score |
|--------|-------|
| **Mean Fold AUC** | 0.7276 |
| **OOF CV AUC** | 0.7276 |
| **Std Fold AUC** | 0.0008 |

---

## 📁 Repository Structure

```
.
├── README.md
├── LICENSE
├── s5e12-xgboost-drift-correction.ipynb         # Main solution notebook
├── s5e12-xgb-lgbm-cat-drift-weights-auto-blend.ipynb  # Ensemble solution
└── s5e12-xgb-lgbm-cat-drift-weights-auto-blend-Bin.ipynb  # Binning variant
```

---

## 🎓 Key Learnings

1. **Drift Detection**: Temporal drift in features significantly impacts model performance
2. **GPU Parallelization**: Dual GPU training reduces time by 50%
3. **Feature Engineering**: Medical domain knowledge improves model interpretability
4. **Ensemble Methods**: Multi-model blending provides marginal improvements
5. **Cross-Validation**: Stratified K-fold ensures robust performance estimates

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

<div align="center">

### ⭐ Star this repository if you found it helpful!

**Made with ❤️ for the Kaggle Community**

[Back to Top](#-diabetes-prediction-challenge---playground-series-s5e12)

</div>
