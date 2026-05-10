# Auto Insurance Risk & Premium Prediction System

## Project Overview

This project builds a beginner-friendly auto insurance risk and premium prediction system. It uses structured insurance data, simple feature engineering, and standard machine learning models to predict claim risk and estimate expected premiums.

## Business Problem

Insurance companies need to identify high-risk customers and price premiums fairly. This project demonstrates how to use data science to support underwriting, risk segmentation, and premium pricing decisions.

## Dataset Description

The dataset contains customer and policy information such as age, sex, BMI, smoking status, region, vehicle age, annual mileage, accident history, credit score, and driving experience. The dataset also includes a target label for claim risk and an estimated insurance premium.

## Tools and Technologies Used

- Python
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- Jupyter Notebook

## Project Workflow

1. Load or generate the insurance dataset
2. Clean and preprocess data
3. Create synthetic auto insurance features
4. Encode categorical features
5. Train classification and regression models
6. Evaluate model performance
7. Save results and visualizations

## Models Used

- Logistic Regression
- Random Forest Classifier
- Linear Regression
- Random Forest Regressor

## Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R² score

## Key Business Insights

- High-risk customers often have accident history and smoker status.
- Premium estimates increase with vehicle age and credit risk.
- Underwriting decisions can use customer segmentation by risk score and premium level.
- Visual analysis helps identify groups with higher expected claim costs.

## How to Run the Project

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the main script:
   ```bash
   python main.py
   ```
3. View model reports in `outputs/classification_report.txt` and `outputs/regression_results.txt`.
4. Open charts in `outputs/charts/`.
5. Explore the notebook in `notebooks/auto_insurance_analysis.ipynb`.

## Future Improvements

- Add more real-world auto insurance features like policy type and vehicle make.
- Use cross-validation and hyperparameter tuning for stronger model performance.
- Deploy the pipeline with a web interface or API for underwriting teams.
- Include more detailed explainability and model interpretation features.
