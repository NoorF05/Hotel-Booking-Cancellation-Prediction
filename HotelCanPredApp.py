# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 19:22:02 2026

@author: Noor
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ---------------- CONFIG ---------------- #
st.set_page_config(page_title="Hotel Dashboard", layout="wide", page_icon="🏨")

# ---------------- CSS ---------------- #
st.markdown("""
<style>
.card {
    background-color: #161b22;
    padding: 20px;
    border-radius: 12px;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.3);
    text-align: center;
}
.metric {
    font-size: 28px;
    font-weight: bold;
}
.label {
    font-size: 14px;
    color: #8b949e;
}
.section {
    padding: 15px 0px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- LOAD ---------------- #
model = joblib.load("HotelCan_Model.pkl")
columns = joblib.load("columns.pkl")

# ---------------- HEADER ---------------- #
st.title("🏨 Hotel Cancellation Predictor")
st.caption("Smart system to assess booking risk")

# ================= INPUT SECTION ================= #
st.markdown("## 🧾 Booking Details")

col1, col2, col3 = st.columns(3)

# -------- GUEST -------- #
with col1:
    st.markdown("### 👥 Guest Details")
    no_of_adults = st.number_input("Adults", 1, 10, 2)
    no_of_children = st.number_input("Children", 0, 5, 0)

    st.markdown("### 🚗 Preferences")
    parking = st.toggle("Parking Required")
    repeated_guest = st.toggle("Repeated Guest")

    required_car_parking_space = int(parking)
    repeated_guest = int(repeated_guest)

# -------- STAY -------- #
with col2:
    st.markdown("### 🛏 Stay Details")
    no_of_weekend_nights = st.number_input("Weekend Nights", 0, 10, 1)
    no_of_week_nights = st.number_input("Week Nights", 0, 15, 2)

    st.markdown("### 📅 Timing")
    arrival_date_input = st.date_input("Arrival Date")

    arrival_year = arrival_date_input.year
    arrival_month = arrival_date_input.month
    arrival_date = arrival_date_input.day

# -------- HISTORY + TYPE -------- #
with col3:
    st.markdown("### 📊 Booking History")
    no_of_previous_cancellations = st.number_input("Prev Cancellations", 0, 10, 0)
    no_of_previous_bookings_not_canceled = st.number_input("Successful Bookings", 0, 20, 0)

    no_of_special_requests = st.number_input("Special Requests", 0, 5, 1)

    st.markdown("### 🏷 Booking Type")

    room_map = {f"Room_Type {i}": i-1 for i in range(1,8)}
    room_choice = st.selectbox("Room Type", list(room_map.keys()))
    room_type_reserved = room_map[room_choice]

    meal_map = {
        "Meal Plan 1": 0,
        "Meal Plan 2": 1,
        "Meal Plan 3": 2,
        "Not Selected": 3
    }
    meal_choice = st.selectbox("Meal Plan", list(meal_map.keys()))
    type_of_meal_plan = meal_map[meal_choice]

    market_map = {
        "Online": 0,
        "Offline": 1,
        "Corporate": 2,
        "Aviation": 3,
        "Complementary": 4
    }
    market_choice = st.selectbox("Market Segment", list(market_map.keys()))
    market_segment_type = market_map[market_choice]

# -------- PRICING -------- #
st.markdown("### 💰 Pricing & Lead Time")

col4, col5 = st.columns(2)
with col4:
    lead_time = st.slider("Lead Time (days)", 0, 365, 30)
with col5:
    price = st.number_input("Avg Price", 0, 1000, 120)

# ================= PREP ================= #
lead_time_log = np.log1p(lead_time)
price_log = np.log1p(price)

input_data = pd.DataFrame({
    'no_of_adults': [no_of_adults],
    'no_of_children': [no_of_children],
    'no_of_weekend_nights': [no_of_weekend_nights],
    'no_of_week_nights': [no_of_week_nights],
    'required_car_parking_space': [required_car_parking_space],
    'arrival_year': [arrival_year],
    'arrival_month': [arrival_month],
    'arrival_date': [arrival_date],
    'repeated_guest': [repeated_guest],
    'no_of_previous_cancellations': [no_of_previous_cancellations],
    'no_of_previous_bookings_not_canceled': [no_of_previous_bookings_not_canceled],
    'no_of_special_requests': [no_of_special_requests],
    'room_type_reserved': [room_type_reserved],
    'type_of_meal_plan': [type_of_meal_plan],
    'market_segment_type': [market_segment_type],
    'lead_time_log': [lead_time_log],
    'price_log': [price_log]
})

input_data = input_data[columns]

# ================= BUTTON ================= #
st.divider()
predict = st.button("🚀 Analyze Booking", use_container_width=True)

# ================= DASHBOARD OUTPUT ================= #
if predict:
    with st.spinner("Running AI analysis..."):
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1]

    st.markdown("## 📊 Prediction Dashboard")

    # Risk logic
    if probability < 0.4:
        risk = "Low"
        color = "#2ea043"
    elif probability < 0.7:
        risk = "Medium"
        color = "#d29922"
    else:
        risk = "High"
        color = "#f85149"

    result = "Cancel ❌" if prediction == 1 else "Safe ✅"

    # -------- KPI ROW -------- #
    colA, colB, colC = st.columns(3)

    with colA:
        st.markdown(f"""
        <div class="card">
            <div class="label">Cancellation Probability</div>
            <div class="metric">{probability*100:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)

    with colB:
        st.markdown(f"""
        <div class="card">
            <div class="label">Risk Level</div>
            <div class="metric" style="color:{color};">{risk}</div>
        </div>
        """, unsafe_allow_html=True)

    with colC:
        st.markdown(f"""
        <div class="card">
            <div class="label">Final Prediction</div>
            <div class="metric">{result}</div>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # -------- VISUAL -------- #
    st.subheader("📈 Risk Visualization")
    st.progress(float(probability))

    # -------- INSIGHT -------- #
    st.subheader("🧠 Insight")

    if probability > 0.7:
        st.error("High-risk booking → Consider prepayment or reconfirmation.")
    elif probability > 0.4:
        st.warning("Moderate risk → Keep monitoring.")
    else:
        st.success("Low-risk booking → Likely to show up.")