📌 Project Overview

This project focuses on predicting customer churn in the telecom industry using machine learning techniques. Customer churn refers to when customers stop using a company’s services, and predicting it helps businesses take proactive retention measures.

The notebook performs:

Data exploration and visualization
Data preprocessing and feature engineering
Training multiple machine learning models
Model evaluation and comparison
📂 Dataset

The dataset used:

File: telecom_customer_churn_prediction_dataset.csv
Contains customer demographics, account information, and service usage details.
Key Features:
Gender, SeniorCitizen, Partner, Dependents
Tenure, Monthly Charges, Total Charges
Services (Internet, Phone, etc.)
Target Variable: Churn (Yes/No)
⚙️ Technologies Used
Python
Libraries:
pandas, numpy (data handling)
matplotlib, seaborn (visualization)
scikit-learn (ML models & preprocessing)
xgboost (advanced boosting model)
🔍 Project Workflow
1. Data Loading
Imported dataset using pandas
Checked structure and sample records
2. Exploratory Data Analysis (EDA)
Summary statistics (describe, info)
Distribution plots and count plots
Churn analysis across categories (gender, senior citizens, etc.)
3. Data Preprocessing
Handling categorical variables
Encoding labels
Feature scaling using StandardScaler
Train-test split
4. Model Training

Multiple supervised learning models were trained:

Logistic Regression
Decision Tree
Random Forest
K-Nearest Neighbors (KNN)
Naive Bayes
Support Vector Machine (SVM)
XGBoost
📈 Model Evaluation

Each model was evaluated using:

Accuracy Score
Confusion Matrix
Classification Report (Precision, Recall, F1-score)

📊 Visualizations

The notebook includes:

Histograms and count plots
Stacked bar charts
Heatmaps for confusion matrices

These help in understanding customer behavior and model performance.

🚀 How to Run
Clone the repository:
git clone https://github.com/your-username/your-repo-name.git
Install dependencies:
pip install pandas numpy matplotlib seaborn scikit-learn xgboost
Run the notebook:
jupyter notebook Telecom_customer.ipynb


Author:
Aesha Patel
