# Bank Customer Multi-Class Classification  
*Machine Learning Project – Semester 3, 2024*  

---

## Overview  
This project applies supervised learning to predict **four customer classes (A, B, C, D)** from bank customer attributes. The goal is to benchmark tree-based models, apply systematic tuning, and report clear, comparable metrics that reflect balanced performance across all classes.

Dataset source (Kaggle):  
https://www.kaggle.com/datasets/gauravtopre/bank-customer-churn-dataset/data

---

## Key Insights  
- The task is **imbalanced** across classes, so macro metrics are emphasized.  
- **XGBoost** slightly outperforms **Random Forest** after tuning.  
- Tuning provides a consistent lift in **accuracy** and **macro F1** for both models.  
- Class D is the easiest class. Class B is the most difficult.  
- Further gains should target minority or hard classes rather than overall accuracy alone.

---

## Files in this Repository  
- **[`data/Train.csv`](./data/Train.csv)**  
  Cleaned training data used for modeling.  

- **[`src/indiv.ipynb`](./src/indiv.ipynb)**  
  Jupyter Notebook for EDA, preprocessing, model training, tuning, and evaluation.  

---

## Methods  
1. **Exploratory Data Analysis (EDA)**  
   - Checked missing values, duplicates, unique values, and outliers.  
   - Reviewed distributions of numerical features.  
   - Analyzed relationships between features and target classes.  

2. **Data Preprocessing**  
   - One-hot encoding for categorical variables.  
   - Scaling for numerical variables where useful.  
   - Stratified train–test split.  

3. **Modeling and Tuning**  
   - Baseline models: RandomForestClassifier, XGBoostClassifier.  
   - Hyperparameter search with GridSearchCV.  
   - Metrics: **Accuracy**, **Precision**, **Recall**, **F1-score** (macro and weighted).  
   - Final reporting on the test set.  

---

## Results  

### Random Forest  
- **Baseline**  
  - Accuracy: **0.4846**  
  - Macro Precision / Recall / F1: **0.48 / 0.48 / 0.48**  

- **After GridSearchCV**  
  - Accuracy: **0.5341**  
  - Macro Precision / Recall / F1: **0.52 / 0.52 / 0.52**  
  - Best parameters: `max_depth=10`, `n_estimators=200`, `max_features="sqrt"`, `min_samples_split=10`, `min_samples_leaf=2`  

---

### XGBoost  
- **Baseline**  
  - Accuracy: **0.5034**  
  - Macro Precision / Recall / F1: **0.49 / 0.49 / 0.49**  

- **After GridSearchCV**  
  - Accuracy: **0.5431**  
  - Macro Precision / Recall / F1: **0.53 / 0.53 / 0.53**  
  - Best parameters:  
    `learning_rate=0.1`, `max_depth=3`, `subsample=1.0`, `colsample_bytree=0.8`,  
    `min_child_weight=10`, `alpha=0.1`, `lambda=0.1`, `gamma=0.2`, `n_estimators=200`  

---

## How to Reproduce  

### Requirements  
- Python 3.8+  
- Packages: `pandas`, `numpy`, `scikit-learn`, `xgboost`, `matplotlib`, `seaborn`  

### Steps  
1. Clone the repository.  
2. Open `src/indiv.ipynb` in Jupyter.  
3. Run all cells to reproduce preprocessing, training, tuning, and evaluation.  

---

## Next Steps  
- Apply class weights or focal loss to improve hard classes.  
- Try LightGBM and CatBoost for gradient boosting comparisons.  
- Add repeated stratified cross-validation and calibration.  
- Engineer interaction features and domain aggregates.  

---

## Author  
Arieldhipta Tarliman  
BINUS University

Suggested repository name: **bank-customer-multiclass-classifier-2024**
