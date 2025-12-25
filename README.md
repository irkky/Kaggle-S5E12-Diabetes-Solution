# 🏥 Diabetes Prediction Challenge - Playground Series S5E12

<div align="center">

![Python](https://img.shields.io/badge/Python-3.12-blue.svg)
![XGBoost](https://img.shields.io/badge/XGBoost-3.1.0-green.svg)
![CUDA](https://img.shields.io/badge/CUDA-Enabled-brightgreen.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

**Advanced Machine Learning Solution for Diabetes Risk Prediction**

[🔗 Kaggle Competition](https://www.kaggle.com/competitions/playground-series-s5e12) | [📝 Notebook](https://www.kaggle.com/code/rishabhkannaujiya/s5e12-xgboost-drift-correction)

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
- 📈 **AUC Score**: ~0.7276 out-of-fold validation score

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
- **Early Stopping** (50 rounds patience)
- **Parallel GPU Training** (2x NVIDIA Tesla T4)
- **Sample Weighting** for drift correction

---

## 📊 Results

### Cross-Validation Performance

| Metric | Score |
|--------|-------|
| **Mean Fold AUC** | 0.7276 |
| **OOF CV AUC** | 0.7276 |
| **Std Fold AUC** | 0.0008 |

### Individual Fold Scores

```
Fold 1: 0.7282 AUC (GPU 0)
Fold 2: 0.7263 AUC (GPU 1)
Fold 3: 0.7273 AUC (GPU 0)
Fold 4: 0.7287 AUC (GPU 1)
Fold 5: 0.7277 AUC (GPU 0)
```

---

## 💡 Technical Highlights

### 1. Parallel GPU Training
```python
results = Parallel(n_jobs=2, backend="threading")(
    delayed(train_fold)(fold, tr, val, X, y, sample_weights, xgb_params)
    for fold, (tr, val) in enumerate(skf.split(X, y))
)
```
**Impact**: 2x training speedup using dual GPU architecture

### 2. Drift Correction Framework
```python
# Sample weighting based on drift analysis
sample_weights = calculate_drift_weights(train_df, test_df)
```
**Impact**: Improved model robustness to temporal distribution shifts

### 3. Advanced Feature Engineering
- **Medical Ratios**: LDL/HDL, Triglycerides/HDL
- **Pressure Metrics**: Pulse Pressure, MAP
- **Lifestyle Interactions**: Screen Time × Physical Activity
- **Risk Aggregations**: Metabolic Risk Score

### 4. Robust Validation Strategy
- Stratified sampling maintains target distribution
- 5-fold CV ensures reliable performance estimates
- Early stopping prevents overfitting

---

<div align="center">

### ⭐ Star this repository if you found it helpful!

**Made with ❤️ for the Kaggle Community**

[Back to Top](#-diabetes-prediction-challenge---playground-series-s5e12)

</div>
