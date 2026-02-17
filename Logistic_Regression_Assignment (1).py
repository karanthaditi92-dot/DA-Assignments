# app.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    classification_report
)

# -------------------------------
# App Title
# -------------------------------
st.title("Logistic Regression Model Deployment - Diabetes Dataset")

# -------------------------------
# 1. Data Exploration
# -------------------------------

@st.cache_data
def load_data():
    df = pd.read_csv("diabetes.csv")  # Change path if needed
    return df

df = load_data()

st.subheader("Dataset Overview")
st.write("Shape of dataset:", df.shape)
st.dataframe(df.head())

st.subheader("Data Types")
st.write(df.dtypes)

st.subheader("Summary Statistics")
st.write(df.describe())

# -------------------------------
# Visualizations
# -------------------------------
st.subheader("Feature Distributions")

numeric_cols = df.select_dtypes(include=np.number).columns

for col in numeric_cols:
    fig, ax = plt.subplots()
    df[col].hist(bins=20, ax=ax)
    ax.set_title(f"Histogram of {col}")
    st.pyplot(fig)

st.subheader("Correlation Matrix")
fig, ax = plt.subplots(figsize=(8, 6))
corr = df.corr()
cax = ax.matshow(corr)
plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
plt.yticks(range(len(corr.columns)), corr.columns)
fig.colorbar(cax)
st.pyplot(fig)

# -------------------------------
# 2. Data Preprocessing
# -------------------------------

st.subheader("Data Preprocessing")

# Handle missing values (replace 0s in certain columns with median)
cols_with_zero = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

for col in cols_with_zero:
    df[col] = df[col].replace(0, np.nan)
    df[col].fillna(df[col].median(), inplace=True)

st.write("Missing values after imputation:")
st.write(df.isnull().sum())

# -------------------------------
# Feature & Target Split
# -------------------------------
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# -------------------------------
# 3. Model Building
# -------------------------------

model = LogisticRegression()
model.fit(X_train, y_train)

# -------------------------------
# 4. Model Evaluation
# -------------------------------

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_prob)

st.subheader("Model Evaluation")

st.write("Accuracy:", accuracy)
st.write("Precision:", precision)
st.write("Recall:", recall)
st.write("F1 Score:", f1)
st.write("ROC-AUC Score:", roc_auc)

st.subheader("Classification Report")
st.text(classification_report(y_test, y_pred))

# ROC Curve
st.subheader("ROC Curve")
fpr, tpr, _ = roc_curve(y_test, y_prob)

fig, ax = plt.subplots()
ax.plot(fpr, tpr)
ax.plot([0, 1], [0, 1], linestyle='--')
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curve")
st.pyplot(fig)

# -------------------------------
# 5. Interpretation
# -------------------------------

st.subheader("Model Coefficients Interpretation")

coeff_df = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.coef_[0]
})

st.dataframe(coeff_df)

st.write("""
Positive coefficients increase the probability of diabetes.
Negative coefficients decrease the probability.
Larger magnitude indicates stronger influence.
""")

# -------------------------------
# 6. Deployment - User Input
# -------------------------------

st.subheader("Predict Diabetes")

pregnancies = st.number_input("Pregnancies", 0, 20, 1)
glucose = st.number_input("Glucose", 0, 200, 120)
bp = st.number_input("Blood Pressure", 0, 140, 70)
skin = st.number_input("Skin Thickness", 0, 100, 20)
insulin = st.number_input("Insulin", 0, 900, 80)
bmi = st.number_input("BMI", 0.0, 70.0, 25.0)
dpf = st.number_input("Diabetes Pedigree Function", 0.0, 3.0, 0.5)
age = st.number_input("Age", 10, 100, 30)

if st.button("Predict"):

    input_data = np.array([[pregnancies, glucose, bp, skin,
                            insulin, bmi, dpf, age]])

    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    if prediction == 1:
        st.error(f"High Risk of Diabetes (Probability: {probability:.2f})")
    else:
        st.success(f"Low Risk of Diabetes (Probability: {probability:.2f})")

