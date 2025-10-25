#Bank Customer Churn Prediction



A comprehensive machine learning pipeline and Flask web application that predicts whether a customer will exit (churn) from a bank based on individual and account features. The project is situated at the intersection of predictive analytics and business intelligence, addressing a common problem faced by financial institutions.

Project Topic


Project Theme: Predictive Modeling for Customer Churn
Business Impact:
Churn prediction helps banks proactively identify customers who are at risk of leaving and enables targeted retention strategies. Accurate churn forecasts can improve customer management, reduce lost revenue, and optimize marketing efforts.​

Motivation
Churn is a critical KPI for banks: retaining customers reduces acquisition costs and improves lifetime value.​
Data-driven prediction utilizes features such as demographics (age, gender, geography) and account activity (balance, products, salary, credit score, etc.), enabling actionable insights.

Automating churn prediction with ML helps scale decision-making and exposes hidden patterns in large, complex datasets.

Features
Automated ingestion, preprocessing, and transformation of raw bank data (bank.csv)
Comparison of multiple ML algorithms (Logistic Regression, SVC, KNN, Decision Tree, Random Forest, Gradient Boosting, XGBoost); best model is chosen via accuracy and precision​
Model evaluation metrics: accuracy, precision, recall, F1 score (all tracked in notebook and code)​
User-friendly web interface (Flask + Bootstrap) for manual customer entry and real-time prediction,modular pipeline enables flexible retraining, new data ingestion, and future integration


Quickstart
Clone the repository

Install dependencies
bash
pip install -r requirements.txt

Run ML pipeline scripts

bash
python data_ingestion.py
python data_transformation.py
python model_trainer.py
Launch the web app

bash
python app.py
Then visit http://localhost:5000 in a browser.

Usage
Web UI lets you enter new customer data and predicts churn risk instantly.

Modular scripts allow for retraining with updated bank data and parameter tuning.​

Data & Analysis
Dataset features: CreditScore, Geography, Gender, Age, Tenure, Balance, NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary, Exited.

EDA notebook investigates feature distributions and target patterns.

Notebook and code contain extensive results on model performances and precision/accuracy scores.

Model Pipeline Overview
data_ingestion.py: Loads CSV, removes irrelevant columns, splits train/test.

data_transformation.py: Handles missing values, encodes categorical data, standardizes numerics.

model_trainer.py: Trains and evaluates several classifiers, saves best model based on metrics.

predict_pipeline.py: Interfaces with Flask app for prediction using saved model and preprocessing logic.

Dependencies
python>=3.7

numpy, pandas, scikit-learn, matplotlib, seaborn

xgboost, lightgbm, catboost

flask, joblib, dill


