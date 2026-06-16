import os
import urllib.request
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# 1. Download diabetes dataset if not present
DIABETES_CSV = 'diabetes.csv'
DIABETES_URL = 'https://raw.githubusercontent.com/npradaschnor/Pima-Indians-Diabetes-Dataset/master/diabetes.csv'

if not os.path.exists(DIABETES_CSV):
    print("Downloading diabetes dataset...")
    urllib.request.urlretrieve(DIABETES_URL, DIABETES_CSV)
    print("Downloaded diabetes.csv successfully.")

# 2. Train Diabetes Model (Random Forest + Missing Value Imputation)
print("\n--- Training Diabetes Model ---")
diabetes_df = pd.read_csv(DIABETES_CSV)

# Replace impossible 0 values with NaN for imputation
zero_features = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
diabetes_imputed = diabetes_df.copy()
for col in zero_features:
    diabetes_imputed[col] = diabetes_imputed[col].replace(0, np.nan)

# Impute missing values with group medians (by Outcome)
for col in zero_features:
    diabetes_imputed[col] = diabetes_imputed.groupby('Outcome')[col].transform(lambda x: x.fillna(x.median()))

diabetes_features = [
    'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
    'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'
]

X_diabetes = diabetes_imputed[diabetes_features]
y_diabetes = diabetes_imputed['Outcome']

X_train_d, X_test_d, y_train_d, y_test_d = train_test_split(X_diabetes, y_diabetes, test_size=0.2, random_state=23)

# Fit scaler on imputed training features
diabetes_scaler = StandardScaler()
X_train_d_scaled = diabetes_scaler.fit_transform(X_train_d)
X_test_d_scaled = diabetes_scaler.transform(X_test_d)

# Use Random Forest Classifier for higher accuracy
diabetes_model = RandomForestClassifier(random_state=42)
diabetes_model.fit(X_train_d_scaled, y_train_d)

d_pred = diabetes_model.predict(X_test_d_scaled)
d_acc = accuracy_score(y_test_d, d_pred)
print(f"Diabetes Model Accuracy: {d_acc * 100:.2f}%")

# Save Diabetes model and scaler
with open('diabetes_model.pkl', 'wb') as f:
    pickle.dump(diabetes_model, f)
with open('diabetes_scaler.pkl', 'wb') as f:
    pickle.dump(diabetes_scaler, f)
print("Saved diabetes_model.pkl and diabetes_scaler.pkl")
