#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from fpdf import FPDF
import base64
from datetime import datetime

st.set_page_config(page_title="Collections Root Cause Intelligence (CRCI)", layout="wide")

st.title("🧠 Collections Root Cause Intelligence (CRCI)")
st.markdown("Uncover process bottlenecks and root causes of unresolved debt collection challenges.")

uploaded_file = st.file_uploader("📤 Upload the `crci_dataset.csv` file", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.subheader("📄 Data Preview")
    st.dataframe(df.head())

    st.subheader("📊 SLA Breach Analysis")
    sla_counts = df['SLA_Breached'].value_counts()
    st.bar_chart(sla_counts)

    st.subheader("🔍 Contact Result Breakdown")
    contact_counts = df['Contact_Result'].value_counts()
    fig1, ax1 = plt.subplots()
    contact_counts.plot(kind='bar', color='skyblue', ax=ax1)
    ax1.set_title("Contact Outcomes")
    ax1.set_ylabel("Count")
    st.pyplot(fig1)

    st.subheader("⏱️ Delay Reasons by Resolution Status")
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    sns.countplot(data=df, x='Delay_Reason_Code', hue='Resolution_Status', ax=ax2)
    ax2.set_title("Resolution Status vs. Delay Reasons")
    ax2.tick_params(axis='x', rotation=45)
    st.pyplot(fig2)

    st.subheader("📉 Agent Performance Overview")
    agent_stats = df.groupby('Agent_ID')['Resolution_Status'].value_counts().unstack().fillna(0)
    agent_stats["Total_Handled"] = agent_stats.sum(axis=1)
    agent_stats["Recovery_Rate"] = (agent_stats.get("Recovered", 0) / agent_stats["Total_Handled"] * 100).round(1)
    agent_stats["Unresolved_Rate"] = ((agent_stats.get("Escalated", 0) + agent_stats.get("Unrecovered", 0)) / agent_stats["Total_Handled"] * 100).round(1)
    st.dataframe(agent_stats)

    st.subheader("📊 Agent Recovery Rate Chart")
    fig3, ax3 = plt.subplots()
    agent_stats["Recovery_Rate"].sort_values().plot(kind='barh', color='green', ax=ax3)
    ax3.set_title("Agent Recovery Rates")
    ax3.set_xlabel("Recovery Rate (%)")
    st.pyplot(fig3)

    st.subheader("⚠️ Underperforming Agents")
    underperformers = agent_stats[agent_stats["Recovery_Rate"] < 60]
    if not underperformers.empty:
        st.error("Agents with recovery rate below 60%:")
        st.dataframe(underperformers)
    else:
        st.success("All agents have a recovery rate of 60% or above.")

    st.subheader("📌 Actionable Root Cause Insights")
    insights = []
    if sla_counts.get("Yes", 0) > sla_counts.get("No", 0):
        warning_text = "High number of SLA breaches. Investigate delay reasons and agent workload."
        st.warning(warning_text)
        insights.append(warning_text)

    top_reason = df['Delay_Reason_Code'].value_counts().idxmax()
    insight1 = f"Most common delay reason: {top_reason} - Recommend root-cause investigation."
    st.info(insight1)
    insights.append(insight1)

    failed_contact = df[df['Contact_Result'] != 'Responded']['Contact_Result'].value_counts()
    if not failed_contact.empty:
        most_common_failure = failed_contact.idxmax()
        insight2 = f"Most frequent failed contact outcome: {most_common_failure} - Suggest strategy review for this contact result."
        st.info(insight2)
        insights.append(insight2)

    # PDF Generation
    class PDF(FPDF):
        def header(self):
            self.set_font('Arial', 'B', 12)
            self.cell(0, 10, 'Collections Root Cause Intelligence (CRCI) Report', 0, 1, 'C')
            self.set_font('Arial', '', 10)
            self.cell(0, 10, f'Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M")}', 0, 1, 'C')
            self.ln(5)

        def chapter_title(self, title):
            self.set_font('Arial', 'B', 11)
            self.cell(0, 10, title, 0, 1, 'L')
            self.ln(2)

        def chapter_body(self, body):
            self.set_font('Arial', '', 10)
            self.multi_cell(0, 10, body)
            self.ln()

    if st.button("📥 Download PDF Report"):
        pdf = PDF()
        pdf.add_page()
        pdf.chapter_title("Root Cause Insights")
        for insight in insights:
            # Strip non-latin characters to avoid UnicodeEncodeError
            safe_insight = insight.replace('•', '-').replace('—', '-').encode('latin-1', 'ignore').decode('latin-1')
            pdf.chapter_body("- " + safe_insight)

        pdf_output = "CRCI_Report.pdf"
        pdf.output(pdf_output)

        with open(pdf_output, "rb") as f:
            base64_pdf = base64.b64encode(f.read()).decode('utf-8')
            pdf_display = f'<a href="data:application/pdf;base64,{base64_pdf}" download="CRCI_Report.pdf">📄 Click to download your PDF report</a>'
            st.markdown(pdf_display, unsafe_allow_html=True)

else:
    st.info("Please upload the CRCI dataset to begin analysis.")

