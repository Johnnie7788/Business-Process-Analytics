#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt

st.set_page_config(page_title="Receivables Risk Predictor (RRP)", layout="wide")

st.title("📊 Receivables Risk Predictor (RRP)")
st.markdown("Predict and classify receivables into High, Medium, or Low risk tiers to support better collections strategy.")

uploaded_file = st.file_uploader("Upload the `rrp_dataset.csv` file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.subheader("🔍 Data Preview")
    st.dataframe(df.head())

    # Encode categorical features
    df = df.dropna(subset=["Risk_Tier"])
    target_map = {"High": 2, "Medium": 1, "Low": 0}
    df["Risk_Tier_Num"] = df["Risk_Tier"].map(target_map)

    cat_cols = ["Payment_History_Pattern", "Customer_Segment", "Region", "Preferred_Contact_Channel"]
    ohe = OneHotEncoder(sparse_output=False, drop='first')
    ohe_matrix = ohe.fit_transform(df[cat_cols])
    ohe_df = pd.DataFrame(ohe_matrix, columns=ohe.get_feature_names_out(cat_cols))

    X = pd.concat([
        df[["Invoice_Age", "Outstanding_Balance", "Credit_Score", "Contact_Attempts"]].reset_index(drop=True),
        ohe_df.reset_index(drop=True)
    ], axis=1)
    y = df["Risk_Tier_Num"].reset_index(drop=True)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    st.subheader("📈 Model Performance")
    st.code(classification_report(y_test, y_pred, target_names=["Low", "Medium", "High"]))

    st.subheader("📌 Feature Importance")
    importances = pd.Series(model.feature_importances_, index=X.columns)
    fig, ax = plt.subplots()
    importances.sort_values().plot(kind="barh", ax=ax, color='orange')
    ax.set_title("Feature Importance in Risk Prediction")
    st.pyplot(fig)

    st.subheader("📊 Risk Tier Distribution")
    tier_counts = df["Risk_Tier"].value_counts()
    st.bar_chart(tier_counts)

    display_columns = [
        "Invoice_ID", "Customer_ID", "Outstanding_Balance", "Credit_Score",
        "Payment_History_Pattern", "Invoice_Age", "Region", "Customer_Segment", "Contact_Attempts", "Preferred_Contact_Channel"
    ]

    st.subheader("📋 High-Risk Receivables")
    high_risk = df[df["Risk_Tier"] == "High"][display_columns]
    st.dataframe(high_risk)
    csv_high = high_risk.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download High-Risk Receivables", data=csv_high, file_name="high_risk_receivables.csv", mime="text/csv")

    st.subheader("📋 Medium-Risk Receivables")
    medium_risk = df[df["Risk_Tier"] == "Medium"][display_columns]
    st.dataframe(medium_risk)
    csv_medium = medium_risk.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download Medium-Risk Receivables", data=csv_medium, file_name="medium_risk_receivables.csv", mime="text/csv")

    st.subheader("📋 Low-Risk Receivables")
    low_risk = df[df["Risk_Tier"] == "Low"][display_columns]
    st.dataframe(low_risk)
    csv_low = low_risk.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download Low-Risk Receivables", data=csv_low, file_name="low_risk_receivables.csv", mime="text/csv")

else:
    st.info("Please upload `rrp_dataset.csv` to begin.")

