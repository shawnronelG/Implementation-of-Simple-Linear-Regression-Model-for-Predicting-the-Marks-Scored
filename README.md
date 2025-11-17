# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
Step-by-step explanation (student friendly)
1) Imports
2) Load dataset
3) Prepare input X and target Y
4) Train/test split
5) Create and train the linear model
6) Predict on test set
7) Plots
8) Evaluation metrics
9) Predict new samples

## Program:
```
# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
# 1) Load dataset
df = pd.read_csv('exp_2_dataset_student_scores.csv')   # CSV should have two columns, e.g. "Hours","Scores"
print("First 5 rows:\n", df.head(), "\n")
print("Last 5 rows:\n", df.tail(), "\n")
# 2) Prepare input (X) and output (Y)
# Assume CSV columns: Hours (feature) and Scores (target)
X = df.iloc[:, :-1].values   # all rows, all columns except last -> shape (n_samples, 1)
Y = df.iloc[:, -1].values    # all rows, last column -> shape (n_samples,)
print("X (features):", X.flatten())
print("Y (targets):", Y)
# 3) Split data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=1/3, random_state=0)
print("\nTraining samples:", len(X_train), " Testing samples:", len(X_test))
# 4) Create and train the model
regressor = LinearRegression()
regressor.fit(X_train, Y_train)   # fit on training data
# 5) Predict on the test set
Y_pred = regressor.predict(X_test)
print("\nPredicted values:", np.round(Y_pred, 2))
print("Actual values   :", Y_test)
# 6) Plot training results
plt.figure(figsize=(6,4))
plt.scatter(X_train, Y_train, color="orange", label="Training data")
plt.plot(X_train, regressor.predict(X_train), color="red", label="Fitted line")
plt.title("Hours vs Scores (Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.legend()
plt.grid(True)
plt.show()
# 7) Plot testing results (use X_test sorted for a nicer line)
order = np.argsort(X_test.flatten())
X_test_sorted = X_test.flatten()[order]
Y_test_sorted = Y_test[order]
Y_pred_sorted = Y_pred[order]
plt.figure(figsize=(6,4))
plt.scatter(X_test, Y_test, color="blue", label="Test data")
plt.plot(X_test_sorted, Y_pred_sorted, color="green", label="Predictions")
plt.title("Hours vs Scores (Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.legend()
plt.grid(True)
plt.show()
# 8) Evaluation metrics
mae = mean_absolute_error(Y_test, Y_pred)
mse = mean_squared_error(Y_test, Y_pred)
rmse = np.sqrt(mse)
print("\nMean Absolute Error (MAE):", mae)
print("Mean Squared Error (MSE):", mse)
print("Root Mean Squared Error (RMSE):", rmse)
# 9) Example: predict for new students
new_hours = np.array([[2.5], [8.0]])   # shape must be (n_samples, 1)
pred_new = regressor.predict(new_hours)
print("\nPredictions for new hours", new_hours.flatten(), "=>", np.round(pred_new,2))
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: G. Shawn Ronel
RegisterNumber: 25005544 
*/
```

## Output:
<img width="197" height="373" alt="image" src="https://github.com/user-attachments/assets/ccc5dd89-3030-49b6-95d7-c3358888af4c" />

<img width="781" height="47" alt="image" src="https://github.com/user-attachments/assets/5256958b-ee39-46c5-a21a-d6bf4639c12d" />


<img width="591" height="72" alt="image" src="https://github.com/user-attachments/assets/3bff3946-68da-4e0b-84a3-916279bdeffc" />


<img width="540" height="391" alt="image" src="https://github.com/user-attachments/assets/9181448d-f68e-49f3-b3ff-9049768d80f9" />


<img width="532" height="391" alt="image" src="https://github.com/user-attachments/assets/604d85d5-82a2-4d37-b088-1d5616bb6088" />






## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
