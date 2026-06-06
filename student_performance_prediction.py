# Student Performance Prediction using Multiple Linear Regression
# KodBud AIML Internship - Task 3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# --------------------------------------------------
# Create Sample Dataset
# --------------------------------------------------

data = {
    'Study_Hours': [2, 3, 4, 5, 6, 7, 8, 1, 9, 10, 4, 5, 6, 7, 8],
    'Attendance': [60, 65, 70, 75, 80, 85, 90, 55, 95, 98, 72, 78, 82, 88, 92],
    'Previous_Score': [50, 55, 60, 65, 70, 75, 80, 45, 85, 90, 62, 68, 74, 79, 84],
    'Assignments': [40, 50, 60, 65, 70, 75, 80, 35, 85, 90, 55, 60, 70, 78, 82],
    'Final_Grade': [52, 58, 63, 68, 73, 78, 84, 48, 88, 94, 66, 71, 76, 82, 87]
}

df = pd.DataFrame(data)

print("\nDataset Preview:\n")
print(df.head())

# --------------------------------------------------
# Data Visualization
# --------------------------------------------------

plt.figure(figsize=(8, 5))
sns.heatmap(df.corr(), annot=True, cmap='Blues')
plt.title("Correlation Matrix")
plt.show()

# --------------------------------------------------
# Features and Target
# --------------------------------------------------

X = df[['Study_Hours', 'Attendance', 'Previous_Score', 'Assignments']]
y = df['Final_Grade']

# --------------------------------------------------
# Train-Test Split
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# --------------------------------------------------
# Train Multiple Linear Regression Model
# --------------------------------------------------

model = LinearRegression()
model.fit(X_train, y_train)

# --------------------------------------------------
# Predictions
# --------------------------------------------------

y_pred = model.predict(X_test)

# --------------------------------------------------
# Model Evaluation
# --------------------------------------------------

print("\nModel Evaluation Results")
print("-" * 30)

print("R2 Score:", round(r2_score(y_test, y_pred), 4))
print("Mean Absolute Error:", round(mean_absolute_error(y_test, y_pred), 4))
print("Mean Squared Error:", round(mean_squared_error(y_test, y_pred), 4))

# --------------------------------------------------
# Factors Affecting Performance
# --------------------------------------------------

importance = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': model.coef_
})

importance = importance.sort_values(
    by='Coefficient',
    ascending=False
)

print("\nFactors Affecting Student Performance")
print("-" * 40)
print(importance)

# --------------------------------------------------
# Feature Importance Visualization
# --------------------------------------------------

plt.figure(figsize=(8, 5))
sns.barplot(
    x='Coefficient',
    y='Feature',
    data=importance
)

plt.title("Feature Importance")
plt.xlabel("Impact on Final Grade")
plt.ylabel("Features")
plt.show()

# --------------------------------------------------
# Actual vs Predicted
# --------------------------------------------------

comparison = pd.DataFrame({
    'Actual Grade': y_test.values,
    'Predicted Grade': np.round(y_pred, 2)
})

print("\nActual vs Predicted Grades")
print(comparison)

plt.figure(figsize=(7, 5))
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Grade")
plt.ylabel("Predicted Grade")
plt.title("Actual vs Predicted Grades")
plt.show()

# --------------------------------------------------
# Predict New Student
# --------------------------------------------------

new_student = [[8, 90, 85, 80]]

predicted_grade = model.predict(new_student)

print("\nPredicted Grade for New Student:")
print(f"{predicted_grade[0]:.2f}")
