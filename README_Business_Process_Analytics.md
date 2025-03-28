
# 📊 Business Process Analytics

**Turning operational inefficiencies into actionable intelligence.**

This repository contains a suite of intelligent, data-driven Streamlit applications built to address real-world challenges in receivables management, debt collection optimization, and root cause analysis. Designed specifically for Business Process Analysts, this toolkit offers powerful insights, automated recommendations, and performance diagnostics across critical operational workflows.

---

## 🧠 Included Applications

### 1. **Debt Collection Performance Optimizer (DCPO)**
A full performance dashboard for tracking and optimizing debt recovery operations.
- KPI monitoring: recovery rate, overdue rate, SLA breaches
- ML model for predicting recovery outcomes
- Automated strategic recommendations
- Segment performance and correlation heatmap

📁 Folder: `/DCPO`

---

### 2. **Receivables Risk Predictor (RRP)**
An AI-powered classifier that segments receivables into High, Medium, and Low risk.
- Behavior-driven scoring model
- Real-time risk tier classification
- Prioritization for collections team
- Feature importance and performance reporting

📁 Folder: `/RRP`

---

### 3. **Collections Root Cause Intelligence (CRCI)**
A deep-dive root cause analyzer to uncover bottlenecks and inefficiencies in collections workflows.
- SLA breach detection and delay reason breakdown
- Contact strategy effectiveness by channel/result
- Agent performance dashboard
- Root-cause alerts + PDF summary report generation

📁 Folder: `/CRCI`

---

## 🛠️ Tech Stack
- **Streamlit** – UI and app framework
- **Python (pandas, matplotlib, seaborn, scikit-learn)** – Data processing and ML
- **FPDF** – PDF reporting

---

## 📦 Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/Business-Process-Analytics.git
cd Business-Process-Analytics
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run any app:
```bash
streamlit run DCPO/app.py
```

Replace `DCPO` with `RRP` or `CRCI` to launch other apps.

---

## 📄 Datasets
Each app comes with a sample dataset inside its respective folder (`dcpo_dataset.csv`, `rrp_dataset.csv`, etc.). You can also upload your own.

---

## 📈 Use Case Alignment
This suite was designed to support roles involving:
- Business Process Analysis
- Receivables Management
- Debt Collection Strategy
- Operational Efficiency & Root Cause Diagnostics

---

## ✍️ Author
**John Johnson Ogbidi**  

---

## 📌 License
This project is released under the MIT License.

---

> **“Analytics is not about numbers. It’s about knowing what to fix, what to prioritize, and what to stop doing.”**
