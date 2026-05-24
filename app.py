import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px  # fixed import

from inference.predict_freight import predict as predict_freight
from inference.predict_invoice_flag import predict as predict_invoice_flag

# page config
st.set_page_config(
    page_title="Vendor Invoice Intelligence Portal",
    page_icon="📦",
    layout="wide"
)

# header
st.markdown("""
# 📦 Vendor Invoice Intelligence Portal
### AI-Driven Freight Cost Prediction & Invoice Risk Flagging

This internal analytics portal leverages machine learning to:
- Forecast freight costs accurately
- Detect risky or abnormal vendor invoices
- Reduce financial leakage and manual workload
""")

st.divider()

# sidebar
st.sidebar.title("🔍 Model Selection")

selected_model = st.sidebar.radio(
    "Choose Prediction Module",
    [
        "Freight Cost Prediction",
        "Invoice Manual Approval Flag"
    ]
)

st.sidebar.markdown("""
---
**Business Impact**
- 📉 Improved cost forecasting
- 🧾 Reduce invoice fraud & anomalies
- ⚙️ Faster finance operations
""")

# ---------------- FREIGHT ----------------
if selected_model == "Freight Cost Prediction":

    st.subheader("🚚 Freight Cost Prediction")

    st.markdown("""
    **Objective:**
    Predict freight cost using invoice quantity and dollar value  
    to improve budgeting and vendor cost control.
    """)

    with st.form("freight_form"):
        col1, col2 = st.columns(2)

        with col1:
            quantity = st.number_input("📦 Quantity", min_value=1, value=1200)

        with col2:
            dollars = st.number_input("💲 Invoice Dollars", min_value=1.0, value=18500.0)

        submit_freight = st.form_submit_button("🔮 Predict Freight Cost")

    if submit_freight:
        input_data = {
            "invoice_quantity": [quantity],
            "invoice_dollars": [dollars],
            "Freight": [0],  # dummy (pipeline expects it)
            "total_item_quantity": [quantity],
            "total_item_dollars": [dollars],
            "avg_receiving_delay": [0]
        }

        result = predict_freight(input_data)

        st.success("Prediction completed")

        st.metric(
            label="📊 Estimated Freight Cost",
            value=f"{result['risk_score'][0]*100:.2f}"
        )

# ---------------- FLAG ----------------
else:

    st.subheader("🚨 Invoice Manual Approval Prediction")

    st.markdown("""
    **Objective:**
    Identify invoices that require manual approval based on  
    abnormal pricing, freight cost, or delivery delays.
    """)

    with st.form("invoice_flag_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            invoice_quantity = st.number_input("Invoice Quantity", min_value=1, value=50)
            freight = st.number_input("Freight Cost", min_value=0.0, value=200.0)

        with col2:
            invoice_dollars = st.number_input("Invoice Dollars", min_value=1.0, value=5000.0)
            total_item_quantity = st.number_input("Total Item Quantity", min_value=1, value=48)

        with col3:
            total_item_dollars = st.number_input("Total Item Dollars", min_value=1.0, value=4800.0)
            avg_receiving_delay = st.number_input("Avg Receiving Delay", min_value=0, value=5)

        submit_flag = st.form_submit_button("🧠 Evaluate Invoice Risk")

    if submit_flag:
        input_data = {
            "invoice_quantity": [invoice_quantity],
            "invoice_dollars": [invoice_dollars],
            "Freight": [freight],
            "total_item_quantity": [total_item_quantity],
            "total_item_dollars": [total_item_dollars],
            "avg_receiving_delay": [avg_receiving_delay]
        }

        result = predict_invoice_flag(input_data)

        is_flagged = bool(result["prediction"][0])

        if is_flagged:
            st.error("🚨 Invoice requires MANUAL APPROVAL")
        else:
            st.success("✅ Invoice is SAFE for Auto-Approval")

        st.metric(
            "Risk Score",
            f"{result['risk_score'][0]:.2f}"
        )