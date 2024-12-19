# Fuel Cell Performance Analysis and Prediction

This repository contains the analysis and prediction of fuel cell performance using advanced regression models. The project focuses on **Target 4** among multiple targets available in the dataset and aims to predict its behavior through extensive data preprocessing, analysis, and model evaluation.

---

## Project Overview

Fuel cells are an essential technology in clean energy generation, and understanding their performance metrics is vital for research and development. This project analyzes a dataset containing fuel cell performance data and applies various predictive models to accurately forecast **Target 4**. The steps include:

1. **Data Preprocessing**: Handling missing values, scaling features, and splitting the data into training and testing sets.
2. **Modeling**: Applying and evaluating multiple regression models, including advanced algorithms like XGBoost and LightGBM.
3. **Evaluation**: Comparing models using performance metrics such as RMSE and R² to identify the best-performing model.
4. **Results**: Highlighting key trends and insights from the analysis.

---

## Technologies Used

- **Programming Language**: Python
- **Libraries**: 
  - Data Analysis: `pandas`, `numpy`
  - Visualization: `matplotlib`, `seaborn`
  - Machine Learning: `scikit-learn`, `xgboost`, `lightgbm`

---

## Workflow

### 1. Data Preprocessing
- Removed missing values and handled anomalies.
- Scaled all features using `StandardScaler` for consistency.
- Split the dataset into **70% training** and **30% testing** sets.

### 2. Models Applied
The following models were implemented and evaluated:
- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor
- Gradient Boosting Regressor
- Support Vector Regressor (SVR)
- K-Nearest Neighbors (KNN)
- XGBoost Regressor
- LightGBM Regressor

### 3. Model Evaluation
- Performance metrics used:
  - **RMSE (Root Mean Squared Error)**
  - **R² Score**
- Identified the best model based on the lowest RMSE and highest R² score.

---

## Results

- **Best Model**: `Linear Regression`
- **Performance**:
  - RMSE: `2.2134119816076017`
  - R²: `-0.016989490810566776`
