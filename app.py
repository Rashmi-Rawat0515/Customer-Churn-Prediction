import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, confusion_matrix

#PAGE CONFIG
st.set_page_config(page_title="Customer Churn Predictor", layout="wide")

#LOAD AND TRAIN MODEL
def train_model():
    df = pd.read_csv("churn.csv")

    df = df.drop("customerID",axis=1)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
    cat_cols = df.select_dtypes(include='object').columns.tolist()
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    df = df.astype(int)

    x = df.drop('Churn', axis=1)
    y = df['Churn']

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    smote = SMOTE(random_state=42)
    x_train_bal, y_train_bal = smote.fit_resample(x_train, y_train)

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train_bal)
    x_test_scaled = scaler.transform(x_test)

    model = LogisticRegression(random_state=42)
    model.fit(x_train_scaled, y_train_bal)
    
    auc = roc_auc_score(y_test, model.predict_proba(x_test_scaled)[:,1])

    return model, auc, scaler, x.columns.tolist(), x_test_scaled, y_test

model, auc, scaler, feature_names, x_test_scaled, y_test = train_model()

#TITLE
st.header("Customer Churn Predictor")
st.markdown("Predict which customers are likely to leave -- and why.")

st.divider()

#METRIC CARD
col1, col2, col3 = st.columns(3)
col1.metric("Model","Logistic Regression")
col2.metric("ROC_AUC Score", round(auc, 3))
col3.metric("Training Technique","SMOTE")

st.divider()

#PREDICTION FORM
st.subheader("Predict Churn for a Customer")
st.markdown("Adjust the sliders to match a customer's profile:")

col_a, col_b, col_c = st.columns(3)

with col_a:
    tenure = st.slider("Tenure (months)", 0, 72, 12)
    monthly_charges = st.slider("Monthly Charges ($)", 18, 120, 65)

with col_b:
    total_charges = st.slider("Total Charges ($)", 0, 9000, 1500)
    senior_citizen = st.selectbox("Senior Citizen", [0, 1])

with col_c:
    paperless_billing = st.selectbox("Paperless Billing", [0, 1])
    tech_support = st.selectbox("Has Tech Support", [0, 1])

#build input with all features with default 0
input_dict = {col: 0 for col in feature_names}
input_dict['tenure'] = tenure
input_dict['MonthlyCharges'] = monthly_charges
input_dict['TotalCharges'] = total_charges
input_dict['SeniorCitizen'] = senior_citizen

if paperless_billing:
    input_dict["PaperlessBilling_Yes"] = 1

if tech_support:
    if "TechSupport_Yes" in input_dict:
        input_dict['TechSupport_Yes'] = 1

input_df = pd.DataFrame([input_dict])
input_scaled = scaler.transform(input_df)
churn_proba = float(model.predict_proba(input_scaled)[0][1])
churn_pred = "Likely to Churn!!" if churn_proba > 0.5 else "Likely to stay"

st.divider()

 # SHOW PREDICTION
st.subheader("Prediction Result")
col_x,col_y = st.columns(2)

col_x.metric("Churn Probability",f"{churn_proba*100:.1f}%")
col_y.metric("Prediction", churn_pred)

st.divider()

# RISK BAR
fig, ax = plt.subplots(figsize = (6,1))
ax.barh(['Risk'],[churn_proba], color= "red" if churn_proba>0.5 else "green")
ax.barh(["Risk"], [1-churn_proba], left=[churn_proba], color ="lightgrey")
ax.set_xlim(0,1)
ax.axvline(0.5, color="black", linestyle="--", linewidth=1)
ax.set_xlabel("Churn Probability")
plt.tight_layout()
st.pyplot(fig)

st.divider()

# CONFUSION MATRIX
st.subheader("Model Performance")
y_pred = model.predict(x_test_scaled)
cm = confusion_matrix(y_test, y_pred)
fig1, ax1 = plt.subplots(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["No Churn","Churn"],
            yticklabels=["No Churn","Churn"], ax=ax1)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
ax.set_title("Confusion Matrix")
st.pyplot(fig1)

st.divider()

#SHOW RAW DATA
if st.checkbox("Show Raw Data"):
    raw = pd.read_csv("churn.csv")
    st.dataframe(raw.head(50))