# 📦 Vendor Invoice Intelligence Portal

An end-to-end Machine Learning project to **predict freight costs** and **identify risky vendor invoices** using a Streamlit dashboard.

---

## 📑 Table of Contents

- Project Overview  
- Business Objective  
- Data Sources  
- Exploratory Data Analysis  
- Models Used  
- Evaluation Metrics  
- Application   

---

## 🚀 Project Overview

This project builds a machine learning system to:

- Predict freight cost for vendor invoices  
- Detect anomalies in invoice data  
- Flag invoices requiring manual approval  

It combines data processing, feature engineering, machine learning, and UI deployment into one pipeline.

---

## 🎯 Business Objective

Businesses often face:

- Incorrect freight cost estimation  
- Fraudulent or abnormal invoices  
- Delayed approval workflows  

This system helps to:

- Reduce financial leakage  
- Automate invoice validation  
- Improve cost forecasting  

---

## 🗄 Data Sources

Data is stored in a **SQLite database (`inventory.db`)**.

### Tables:
- `purchases` → item-level purchase data  
- `vendor_invoice` → invoice-level data  

### Key Features:
- Quantity  
- Dollars  
- Freight  
- Receiving and PO dates  

---

## 📊 Exploratory Data Analysis

EDA focused on:

- Distribution of invoice amounts  
- Freight vs invoice value  
- Receiving delays  
- Outlier detection  

### Insights:
- Large differences between invoice and purchase values indicate risk  
- High receiving delays often signal anomalies  

---

## 🤖 Models Used

### Invoice Risk Prediction
- Random Forest Classifier  
- Pipeline: StandardScaler + Model  
- Output: 0 (Safe), 1 (Flagged)

### Freight Prediction
- Based on learned patterns from invoice features  

---

## 📏 Evaluation Metrics

- Accuracy  
- Precision  
- Recall  
- F1-score  

Focus is on **recall**, since missing risky invoices is more costly than false positives.

---

## 🌐 Application

A Streamlit dashboard provides:

### 🚚 Freight Cost Prediction
- Input: Quantity, Invoice Dollars  
- Output: Estimated freight value  

### 🚨 Invoice Risk Prediction
- Input: Invoice and purchase features  
- Output:
  - Risk Flag (Safe / Manual Approval)
  - Risk Score (probability)

---
