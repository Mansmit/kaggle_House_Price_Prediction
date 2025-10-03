# Housing Price Prediction - Machine Learning Project

This repository contains a Jupyter Notebook for predicting housing prices using multiple machine learning models, including Linear Regression, Random Forest, and XGBoost. The workflow demonstrates data preprocessing, feature engineering, model training, and evaluation on a real-world housing dataset.

---

## Features

- **Data Exploration & Preprocessing**
  - Loads the `train.csv` housing dataset.
  - Handles missing values:
    - Categorical columns: filled with mode.
    - Numerical columns: filled with mean.
  - Encodes categorical features using OneHotEncoder.

- **Feature Engineering**
  - Creates a high-dimensional feature set from one-hot encoding.
  - Scales features for model training using StandardScaler.

- **Modeling Approaches**
  - **Linear Regression** (scikit-learn)
  - **Random Forest Regressor** (scikit-learn)
  - **XGBoost Regressor**
    - Manual imputation workflow
    - Internal imputation workflow (demonstrates XGBoost's ability to handle NA values natively)

- **Model Evaluation**
  - R² Score (Accuracy)
  - Root Mean Squared Error (RMSE)

- **Comparison of Methods**
  - All models are compared on the same train/test split.
  - XGBoost with internal imputation achieves the highest accuracy.

---

## How to Use

1. **Requirements**
   - Python 3.x
   - Jupyter Notebook
   - pandas
   - numpy
   - scikit-learn
   - seaborn
   - matplotlib
   - xgboost

   Install dependencies:
   ```bash
   pip install pandas numpy scikit-learn seaborn matplotlib xgboost
   ```

2. **Run the Notebook**
   - Place `train.csv` (the housing dataset) in the same directory as the notebook.
   - Open `Housing_Price_Prediction_Ver2.ipynb` in Jupyter Notebook or JupyterLab.
   - Execute cells sequentially to reproduce results.

---

## Workflow Overview

1. **Import Libraries**
2. **Load Dataset**
3. **Data Inspection:** `.info()`, `.head()`, `.isnull().sum()`
4. **Missing Value Imputation**
5. **Feature Encoding**
6. **Train/Test Split**
7. **Feature Scaling**
8. **Model Training & Prediction**
   - Linear Regression
   - Random Forest
   - XGBoost (manual & internal imputation)
9. **Performance Metrics**
   - R² Score
   - RMSE

---

## Example Results

- **Linear Regression:**  
  Accuracy: 0.89  
  RMSE: ~28,477

- **Random Forest:**  
  Accuracy: 0.88  
  RMSE: ~29,857

- **XGBoost (manual imputation):**  
  Accuracy: 0.90  
  RMSE: ~27,554

- **XGBoost (internal imputation):**  
  Accuracy: 0.91  
  RMSE: ~26,245

---

## Notes

- The notebook demonstrates both manual and XGBoost-native approaches to missing value handling.
- One-hot encoding may drastically increase the number of features, which is handled efficiently by tree-based models.
- Each model's performance is evaluated on the same test set for fair comparison.

---

## License

This repository is released under the MIT License.

---

**For educational use. Adapt for your own machine learning projects!**