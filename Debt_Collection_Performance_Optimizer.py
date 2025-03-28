#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import OneHotEncoder

import base64
from io import BytesIO
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from datetime import datetime

st.set_page_config(page_title="Debt Collection Performance Optimizer", layout="wide")

st.title("📊 Debt Collection Performance Optimizer (DCPO)")
st.markdown("Analyze KPIs, flag risk segments, predict recovery, and drive strategic improvements in debt collection.")

uploaded_file = st.file_uploader("Upload the required `dcpo_dataset.csv` file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file, parse_dates=["Disbursement_Date", "Due_Date", "Paid_Date", "Recovery_Date"])

    def assign_risk(row):
        if row['Days_Past_Due'] > 90 or row['Outstanding_Balance'] > 10000:
            return 'High Risk'
        elif row['Days_Past_Due'] > 30 or row['Invoice_Amount'] > 5000:
            return 'Medium Risk'
        else:
            return 'Low Risk'
    df['Risk_Flag'] = df.apply(assign_risk, axis=1)

    st.subheader("📈 Key Performance Indicators")
    total_invoices = len(df)
    recovered = df[df['Collection_Status'] == 'Recovered']
    overdue = df[df['Paid_Date'] > df['Due_Date']]
    sla_breaches = df[df['Days_To_Collect'] > 30]

    recovery_rate = len(recovered) / total_invoices * 100
    overdue_rate = len(overdue) / total_invoices * 100
    sla_count = len(sla_breaches)

    col1, col2, col3 = st.columns(3)
    col1.metric("Recovery Rate", f"{recovery_rate:.1f} %")
    col2.metric("Overdue Rate", f"{overdue_rate:.1f} %")
    col3.metric("SLA Breaches (>30 days)", f"{sla_count}")

    st.subheader("📊 Risk Segment Performance")
    segment_summary = df.groupby("Risk_Flag").agg({
        "Invoice_Amount": "sum",
        "Recovered": "mean",
        "Days_To_Collect": "mean"
    }).rename(columns={
        "Invoice_Amount": "Total Invoiced (€)",
        "Recovered": "Recovery Rate (%)",
        "Days_To_Collect": "Avg Days to Collect"
    })
    segment_summary["Recovery Rate (%)"] *= 100
    st.dataframe(segment_summary.style.background_gradient(cmap='Reds'))

    st.subheader("🧠 Root-Cause Analysis: Correlation Heatmap")
    num_cols = df.select_dtypes(include=np.number)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(num_cols.corr(), annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
    st.pyplot(fig)

    st.subheader("📚 Recovery Prediction Model (Random Forest)")

    model_df = df.copy()
    model_df = model_df.dropna(subset=["Recovered", "Credit_Score", "Loan_Term", "Employment_Status", "Risk_Flag"])

    cat_cols = ["Employment_Status", "Risk_Flag", "Payment_History_Pattern", "Region", "Channel_of_Contact"]
    ohe = OneHotEncoder(sparse_output=False, drop='first')
    ohe_matrix = ohe.fit_transform(model_df[cat_cols])
    ohe_df = pd.DataFrame(ohe_matrix, columns=ohe.get_feature_names_out(cat_cols))

    X = pd.concat([
        model_df[["Days_To_Collect", "Invoice_Amount", "Credit_Score", "Loan_Term", "Days_Past_Due", "Contact_Attempts"]].reset_index(drop=True),
        ohe_df.reset_index(drop=True)
    ], axis=1)
    y = model_df["Recovered"].reset_index(drop=True)

    df_combined = pd.concat([X, y], axis=1)
    majority = df_combined[df_combined["Recovered"] == 0]
    minority = df_combined[df_combined["Recovered"] == 1]
    minority_upsampled = minority.sample(n=len(majority), replace=True, random_state=42)
    balanced = pd.concat([majority, minority_upsampled]).sample(frac=1, random_state=42)

    X_bal = balanced.drop("Recovered", axis=1)
    y_bal = balanced["Recovered"]

    X_train, X_test, y_train, y_test = train_test_split(X_bal, y_bal, test_size=0.3, random_state=42)
    clf = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    report = classification_report(y_test, y_pred, output_dict=False)
    st.text("Balanced Classification Report (Random Forest Model)")
    st.code(classification_report(y_test, y_pred), language='text')

    st.subheader("📌 Feature Importance (Random Forest)")
    importances = pd.Series(clf.feature_importances_, index=X_bal.columns)
    fig2, ax2 = plt.subplots()
    importances.sort_values().plot(kind='barh', color='teal', ax=ax2)
    ax2.set_title("Feature Importance in Recovery Prediction")
    st.pyplot(fig2)

    chart_buffer = BytesIO()
    fig2.savefig(chart_buffer, format='PNG')
    chart_buffer.seek(0)
    chart_image = ImageReader(chart_buffer)

    st.subheader("💡 Actionable Recommendations")
    if recovery_rate < 80:
        st.warning("Low recovery rate detected. Consider segment-specific strategies and automated outreach for overdue accounts.")
    if overdue_rate > 20:
        st.warning("High overdue rate. Reassess payment terms and client engagement workflow.")
    if sla_count > 0:
        st.warning(f"{sla_count} accounts breached SLA. Evaluate workflow bottlenecks and agent performance metrics.")

    st.subheader("📄 Generate PDF Report")
    if st.button("📥 Download KPI + Model Summary Report (PDF)"):
        buffer = BytesIO()
        c = canvas.Canvas(buffer, pagesize=A4)
        textobject = c.beginText(40, 800)

        textobject.setFont("Helvetica-Bold", 14)
        textobject.textLine("Debt Collection Performance Summary Report")
        textobject.setFont("Helvetica", 12)
        textobject.textLine(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M')}")

        textobject.moveCursor(0, 20)
        textobject.textLine("KEY PERFORMANCE INDICATORS")
        textobject.textLine(f"- Recovery Rate: {recovery_rate:.1f}%")
        textobject.textLine(f"- Overdue Rate: {overdue_rate:.1f}%")
        textobject.textLine(f"- SLA Breaches: {sla_count}")

        textobject.moveCursor(0, 20)
        textobject.textLine("ACTIONABLE RECOMMENDATIONS")
        if recovery_rate < 80:
            textobject.textLine("• Improve recovery strategy with segment-specific follow-ups.")
        if overdue_rate > 20:
            textobject.textLine("• Reassess payment terms and client engagement workflows.")
        if sla_count > 0:
            textobject.textLine("• Address bottlenecks causing SLA breaches.")

        textobject.moveCursor(0, 20)
        textobject.textLine("MODEL PERFORMANCE")
        for line in classification_report(y_test, y_pred).splitlines():
            textobject.textLine(line)

        c.drawText(textobject)
        c.drawImage(chart_image, 40, 100, width=500, preserveAspectRatio=True, mask='auto')
        c.showPage()
        c.save()
        buffer.seek(0)

        st.download_button(
            label="📄 Click to Download Report",
            data=buffer,
            file_name="dcpo_report.pdf",
            mime="application/pdf"
        )

else:
    st.info("Please upload `dcpo_dataset.csv` to begin.")

