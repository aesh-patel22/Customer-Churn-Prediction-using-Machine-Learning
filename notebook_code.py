# CELL 0
# --------------------
# IMPORTS
# --------------------

# numerical and tabular data handling:
import pandas as pd
import numpy as np

# plotting / visualization:
import matplotlib.pyplot as plt
import seaborn as sns

# sklearn utilities: splitting, scaling, metrics, models, Classifications:
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix,mean_squared_error,r2_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from xgboost import XGBClassifier

# CELL 1
# --------------------
# DATA LOADING
# --------------------

# Read CSV file into pandas DataFrame:

df = pd.read_csv("/telecom_customer_churn_prediction_dataset.csv")

# Show first 5 rows to verify columns and sample values:

df.head()

# CELL 2
# --------------------
# INITIAL EXPLORATION
# --------------------

# Get summary statistics for numeric columns (count, mean, std, min, max, percentiles):

df.describe()

# CELL 3
# Show column data types and non-null counts:

df.info()

# CELL 4
# Example filter: count rows where gender is 'Female' and Churn == Yes:
# This demonstrates how to query and verify categorical labels before mapping:

print(df.query("gender=='Female' and Churn=='Yes'").count())

# CELL 5
plt.figure(figsize=(6, 4))
sns.histplot(x="Churn", data=df, hue="Churn", palette=['skyblue', 'lightcoral'])
plt.show()

# CELL 6
df['gender'].unique()

# CELL 7
# Create a cross-tabulation of 'gender' and 'Churn'
churn_by_gender = pd.crosstab(df['gender'], df['Churn'])

# Plotting the stacked bar chart
churn_by_gender.plot(kind='bar', stacked=True, figsize=(8, 6), color=['skyblue', 'lightcoral'])
plt.title('Churn Distribution by Gender')
plt.xlabel('Gender')
plt.ylabel('Number of Customers')
plt.xticks(rotation=0) # Keep x-axis labels horizontal
plt.legend(title='Churn', labels=['No', 'Yes']) # Rename legend labels for clarity
plt.tight_layout()
plt.show()

# CELL 8


# CELL 9
df['SeniorCitizen'].unique()

# CELL 10
df["SeniorCitizen"] = df["SeniorCitizen"].map({1: "Yes", 0: "No"})

# CELL 11
print(df['SeniorCitizen'].value_counts())

# CELL 12
# Plotting the count plot
plt.figure(figsize=(6, 4))
sns.countplot(x="SeniorCitizen", data=df, hue="SeniorCitizen", palette=['skyblue', 'lightcoral'], legend=False)
plt.xlabel("Senior Citizen")
plt.ylabel('Number of Customers')
plt.title("Senior Citizen Distribution")
plt.show()

# CELL 13
df['Partner'].unique()

# CELL 14
print(df['Partner'].value_counts())

# CELL 15
# Plotting the count plot
plt.figure(figsize=(6, 4))
sns.countplot(x="Partner", data=df, hue="Partner", palette=['skyblue', 'lightcoral'], legend=False)
plt.xlabel("Partner")
plt.ylabel('Number of Customers')
plt.title("Partner Distribution")
plt.show()

# CELL 16
df['Dependents'].unique()

# CELL 17
print(df['Dependents'].value_counts())

# CELL 18
# Plotting the count plot
plt.figure(figsize=(6, 4))
sns.countplot(x="Dependents", data=df, hue="Dependents", palette=['skyblue', 'lightcoral'], legend=False)
plt.xlabel("Dependents")
plt.ylabel('Number of Customers')
plt.title("Dependents Distribution")
plt.show()

# CELL 19
df['tenure'].unique()

# CELL 20
# Plotting the histogram
plt.figure(figsize=(8, 6))
sns.histplot(x='tenure', data=df, kde=True, bins=30, color='skyblue')
plt.title('Distribution of Customer Tenure')
plt.xlabel('Tenure (Months)')
plt.ylabel('Number of Customers')
plt.show()

# CELL 21
df['PhoneService'].unique()

# CELL 22
print(df['PhoneService'].value_counts())

# CELL 23
# Plotting the count plot
plt.figure(figsize=(6, 4))
sns.countplot(x="PhoneService", data=df, hue="PhoneService", palette=['skyblue', 'lightcoral'], legend=False)
plt.xlabel("PhoneService")
plt.ylabel('Number of Customers')
plt.title("PhoneService Distribution")
plt.show()

# CELL 24
df['MultipleLines'].unique()

# CELL 25
print(df['MultipleLines'].value_counts())

# CELL 26
# Create a cross-tabulation of 'MultipleLines' and 'Churn'
multiple_lines_churn = pd.crosstab(df['MultipleLines'], df['Churn'])

# Plotting the stacked bar chart
multiple_lines_churn.plot(kind='bar', stacked=True, figsize=(10, 6), color=['skyblue', 'lightcoral'])
plt.title('Churn Distribution by Multiple Lines Type')
plt.xlabel('Multiple Lines')
plt.ylabel('Number of Customers')
plt.xticks(rotation=0)
plt.legend(title='Churn', labels=['No', 'Yes']) # Rename legend labels for clarity
plt.tight_layout()
plt.show()

# CELL 27
df['InternetService'].unique()

# CELL 28
print(df['InternetService'].value_counts())

# CELL 29
# Create a cross-tabulation of 'InternetService' and 'Churn'
internet_service_churn = pd.crosstab(df['InternetService'], df['Churn'])

# Plotting the stacked bar chart
internet_service_churn.plot(kind='bar', stacked=True, figsize=(10, 6), color=['skyblue', 'lightcoral'])
plt.title('Churn Distribution by Internet Service Type')
plt.xlabel('Internet Service')
plt.ylabel('Number of Customers')
plt.xticks(rotation=0)
plt.legend(title='Churn', labels=['No', 'Yes']) # Rename legend labels for clarity
plt.tight_layout()
plt.show()

# CELL 30
df['OnlineSecurity'].unique()

# CELL 31
print(df['OnlineSecurity'].value_counts())

# CELL 32
# Create a cross-tabulation of 'OnlineSecurity' and 'Churn'
Online_Security_churn = pd.crosstab(df['OnlineSecurity'], df['Churn'])

# Plotting the stacked bar chart
Online_Security_churn.plot(kind='bar', stacked=True, figsize=(10, 6), color=['skyblue', 'lightcoral'])
plt.title('Churn Distribution by Online Security Type')
plt.xlabel('Online Security')
plt.ylabel('Number of Customers')
plt.xticks(rotation=0)
plt.legend(title='Churn', labels=['No', 'Yes']) # Rename legend labels for clarity
plt.tight_layout()
plt.show()

# CELL 33
df['OnlineBackup'].unique()

# CELL 34
print(df['OnlineBackup'].value_counts())

# CELL 35
# Create a cross-tabulation of 'OnlineBackup' and 'Churn'
Online_Backup_churn = pd.crosstab(df['OnlineBackup'], df['Churn'])

# Plotting the stacked bar chart
Online_Backup_churn.plot(kind='bar', stacked=True, figsize=(10, 6), color=['skyblue', 'lightcoral'])
plt.title('Churn Distribution by Online Backup Type')
plt.xlabel('Online Backup')
plt.ylabel('Number of Customers')
plt.xticks(rotation=0)
plt.legend(title='Churn', labels=['No', 'Yes']) # Rename legend labels for clarity
plt.tight_layout()
plt.show()

# CELL 36
df['DeviceProtection'].unique()

# CELL 37
print(df['DeviceProtection'].value_counts())

# CELL 38
# Create a cross-tabulation of 'DeviceProtection' and 'Churn'
Device_Protection_churn = pd.crosstab(df['DeviceProtection'], df['Churn'])

# Plotting the stacked bar chart
Device_Protection_churn .plot(kind='bar', stacked=True, figsize=(10, 6), color=['skyblue', 'lightcoral'])
plt.title('Churn Distribution by Device Protection Type')
plt.xlabel(' Device Protection')
plt.ylabel('Number of Customers')
plt.xticks(rotation=0)
plt.legend(title='Churn', labels=['No', 'Yes']) # Rename legend labels for clarity
plt.tight_layout()
plt.show()

# CELL 39
df['TechSupport'].unique()

# CELL 40
print(df['TechSupport'].value_counts())

# CELL 41
# Create a cross-tabulation of 'TechSupport' and 'Churn'
Tech_Support_churn = pd.crosstab(df['TechSupport'], df['Churn'])

# Plotting the stacked bar chart
Tech_Support_churn .plot(kind='bar', stacked=True, figsize=(10, 6), color=['skyblue', 'lightcoral'])
plt.title('Churn Distribution by Tech Support Type')
plt.xlabel('Tech Support')
plt.ylabel('Number of Customers')
plt.xticks(rotation=0)
plt.legend(title='Churn', labels=['No', 'Yes']) # Rename legend labels for clarity
plt.tight_layout()
plt.show()

# CELL 42
df['StreamingTV'].unique()

# CELL 43
print(df['StreamingTV'].value_counts())

# CELL 44
# Create a cross-tabulation of 'StreamingTV' and 'Churn'
Streaming_TV_churn = pd.crosstab(df['StreamingTV'], df['Churn'])

# Plotting the stacked bar chart
Streaming_TV_churn .plot(kind='bar', stacked=True, figsize=(10, 6), color=['skyblue', 'lightcoral'])
plt.title('Churn Distribution by Streaming TV Type')
plt.xlabel('Streaming TV')
plt.ylabel('Number of Customers')
plt.xticks(rotation=0)
plt.legend(title='Churn', labels=['No', 'Yes']) # Rename legend labels for clarity
plt.tight_layout()
plt.show()

# CELL 45
df['StreamingMovies'].unique()

# CELL 46
print(df['StreamingMovies'].value_counts())

# CELL 47
# Create a cross-tabulation of 'StreamingMovies' and 'Churn'
Streaming_Movies_churn = pd.crosstab(df['StreamingMovies'], df['Churn'])

# Plotting the stacked bar chart
Streaming_Movies_churn .plot(kind='bar', stacked=True, figsize=(10, 6), color=['skyblue', 'lightcoral'])
plt.title('Churn Distribution by Streaming Movies Type')
plt.xlabel('Streaming Movies')
plt.ylabel('Number of Customers')
plt.xticks(rotation=0)
plt.legend(title='Churn', labels=['No', 'Yes']) # Rename legend labels for clarity
plt.tight_layout()
plt.show()

# CELL 48
df['Contract'].unique()

# CELL 49
print(df['Contract'].value_counts())

# CELL 50
# Plotting the pie chart
plt.figure(figsize=(10, 6))
df['Contract'].value_counts().plot.pie(autopct='%1.1f%%', startangle=90, colors=['lightblue', 'lightgreen', 'lightcoral']) #autopct is used to display percentage per slice
plt.title('Contract Type Distribution')
plt.ylabel('')
plt.tight_layout()
plt.show()

# CELL 51
df['PaperlessBilling'].unique()

# CELL 52
print(df['PaperlessBilling'].value_counts())

# CELL 53
# Plotting the count plot
plt.figure(figsize=(6, 4))
sns.countplot(x="PaperlessBilling", data=df, hue="PaperlessBilling", palette=['skyblue', 'lightcoral'], legend=False)
plt.xlabel(" Paperless Billing")
plt.ylabel('Number of Customers')
plt.title("Paperless Billing Distribution")
plt.show()

# CELL 54
df['PaymentMethod'].unique()

# CELL 55
print(df['PaymentMethod'].value_counts())

# CELL 56
# Plotting the pie chart
plt.figure(figsize=(10, 6))
df['PaymentMethod'].value_counts().plot.pie(autopct='%1.1f%%', startangle=90, colors=['lightblue', 'lightgreen', 'lightcoral','lavender'])#autopct is used to display percentage per slice
plt.title('Payment Method Type Distribution')
plt.ylabel('')
plt.tight_layout()
plt.show()

# CELL 57
df['MonthlyCharges'].unique()

# CELL 58
print(df['MonthlyCharges'].value_counts())

# CELL 59

#Plotting the histogram
plt.figure(figsize=(8, 6))
sns.histplot(x='MonthlyCharges', data=df, kde=True, bins=30, color='skyblue')
plt.title('Distribution of Monthly Charges')
plt.xlabel('Monthly Charges')
plt.ylabel('Number of Customers')
plt.show()

# CELL 60
df['TotalCharges'].unique()

# CELL 61
print(df['TotalCharges'].value_counts())

# CELL 62
#Plotting the histogram
plt.figure(figsize=(8, 6))
sns.histplot(x='TotalCharges', data=df, kde=True, bins=70, color='lightblue')
plt.title('Distribution of Total Charges')
plt.xlabel('Total Charges')
plt.ylabel('Number of Customers')
plt.show()

# CELL 63
df['Churn'].unique()

# CELL 64
print(df['Churn'].value_counts())

# CELL 65
# Plotting the count plot
plt.figure(figsize=(6, 4))
sns.countplot(x="Churn", data=df, hue="Churn", palette=['skyblue', 'lightcoral'], legend=False)
plt.xlabel(" Churn")
plt.ylabel('Number of Customers')
plt.title("Churn Distribution")
plt.show()

# CELL 66
# --------------------
# CATEGORICAL MAPPING (TEXT -> NUMERIC)
# --------------------

for col in df.select_dtypes(include=['object']).columns:
  if col != 'Churn':
    df[col] = LabelEncoder().fit_transform(df[col])


# CELL 67


# CELL 68
df['Churn'] = LabelEncoder().fit_transform(df['Churn'])

# CELL 69
df.head()

# CELL 70
# --------------------
# PREPARE FEATURES AND TARGET
# --------------------

# Separate features (X) and target (y):

# drop target column -> features for modeling
X = df.drop('Churn',axis=1)

# keep target column separately
y = df['Churn']

# CELL 71
# Split into train/test sets (80/20) with fixed random_state for reproducibility:

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

# CELL 72
# Standardize training data (mean = 0, std = 1) using training statistics
# Apply the same scaling to test data without refitting

std = StandardScaler()
X_train = std.fit_transform(X_train)
X_test = std.transform(X_test)

# CELL 73
# --------------------
# SUPERVISED MODELING
# --------------------

# 1) Logistic Regression:
# - Appropriate for binary classification; outputs class probabilities and labels.

# fit on training set
model_linear = LogisticRegression()
model_linear.fit(X_train,y_train)

y_pred = model_linear.predict(X_test)   # predicted labels on test set
print(y_pred)


# CELL 74
# compute accuracy

acc_log = accuracy_score(y_test, y_pred)
print(f"Accuracy-score: {acc_log}")

# CELL 75
# Compute regression metrics to evaluate baseline behavior:

print(f"Mean-Squared-Error: {mean_squared_error(y_test, y_pred)}")
print(f"R2-Score: {r2_score(y_test, y_pred)}")

# CELL 76
print(confusion_matrix(y_test, y_pred))

# CELL 77
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d")

# CELL 78
# Classification Report:

print(classification_report(y_test, y_pred))

# CELL 79
# 2) Decision Tree:
# - Train a Decision Tree classifier using Gini index on training data

decision_tree = DecisionTreeClassifier(criterion="gini", random_state=42).fit(X_train, y_train)
decision_tree

# CELL 80
y_pred = decision_tree.predict(X_test)
y_pred

# CELL 81
acc_dt = accuracy_score(y_test, y_pred)
print(f"Accuracy-score: {acc_dt}")

# CELL 82
print(confusion_matrix(y_test, y_pred))

# CELL 83
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d")

# CELL 84
print(classification_report(y_test, y_pred))

# CELL 85
# 3) Random Forest:
# - Train a Random Forest using the same max depth as the Decision Tree for fair comparison

random_forest = RandomForestClassifier(max_depth=decision_tree.tree_.max_depth, random_state=42).fit(X_train, y_train)
random_forest

# CELL 86
y_pred = random_forest.predict(X_test)
y_pred

# CELL 87
acc_rf = accuracy_score(y_test, y_pred)
print(f"Accuracy-score: {acc_rf}")

# CELL 88
print(confusion_matrix(y_test, y_pred))

# CELL 89
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d")

# CELL 90
print(classification_report(y_test, y_pred))

# CELL 91
# 4) K-Nearest Neighbors (KNN):
# - Train a KNN classifier with k = 5 using training data

kneighbors = KNeighborsClassifier(n_neighbors=5).fit(X_train, y_train)
kneighbors

# CELL 92
y_pred = kneighbors.predict(X_test)
y_pred

# CELL 93
acc_kn = accuracy_score(y_test, y_pred)
print(f"Accuracy-score: {acc_kn}")

# CELL 94
print(confusion_matrix(y_test, y_pred))

# CELL 95
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d")

# CELL 96
print(classification_report(y_test, y_pred))

# CELL 97
# 5) Naive Bayes:
# - Train a Gaussian Naive Bayes classifier using training data

naive_bayes = GaussianNB().fit(X_train, y_train)
naive_bayes

# CELL 98
y_pred = naive_bayes.predict(X_test)
y_pred

# CELL 99
acc_nb = accuracy_score(y_test, y_pred)
print(f"Accuracy-score: {acc_nb}")

# CELL 100
print(confusion_matrix(y_test, y_pred))

# CELL 101
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d")

# CELL 102
print(classification_report(y_test, y_pred))

# CELL 103
# 6) Support Vector Machine (SVM):
# - Train a linear kernel SVM classifier using training data

SVC_classifier = SVC(kernel="linear", random_state=42).fit(X_train, y_train)
SVC_classifier

# CELL 104
y_pred = SVC_classifier.predict(X_test)
y_pred

# CELL 105
acc_svc = accuracy_score(y_test, y_pred)
print(f"Accuracy-score: {acc_svc}")

# CELL 106
print(confusion_matrix(y_test, y_pred))

# CELL 107
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d")

# CELL 108
print(classification_report(y_test, y_pred))

# CELL 109
# 7) XGBoost:
# - Train an XGBoost classifier using training data and log loss as evaluation metric

xgboost = XGBClassifier(eval_metric="logloss",random_state=42).fit(X_train, y_train)
xgboost

# CELL 110
y_pred = xgboost.predict(X_test)
y_pred

# CELL 111
acc_xgb = accuracy_score(y_test, y_pred)
print(f"Accuracy-score: {acc_xgb}")

# CELL 112
print(confusion_matrix(y_test, y_pred))

# CELL 113
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d")

# CELL 114
print(classification_report(y_test, y_pred))

# CELL 115
# Store accuracy scores of all trained models for comparison

accuracies = {
    "Logistic Regression": acc_log,
    "Decision Tree": acc_dt,
    "Random Forest": acc_rf,
    "K-Nearest Neighbors": acc_kn,
    "Naive Bayes": acc_nb,
    "Support Vector Machine": acc_svc,
    "XGBoost": acc_xgb

}

# CELL 116
# Find the model with the highest accuracy and store its name and score

best_model_name = max(accuracies, key=accuracies.get)
best_accuracy = accuracies[best_model_name]

# CELL 117
# Display the best performing model and its accuracy as final conclusion

print("----- Conclusion -----")
print(f"Best supervised model by test accuracy: {best_model_name} (accuracy = {best_accuracy:.4f})")

# CELL 118


