# Hotel Booking Cancellation Predictor

This project is a **Streamlit app** that predicts whether a hotel booking is likely to be cancelled. The model uses guest details, booking history, room type, market segment, and other features to provide a cancellation risk estimate.

---

## 📝 Project Overview

- Predict hotel booking cancellations to help hotels manage bookings and resources efficiently.
- Uses a machine learning model trained on historical booking data.
- Includes a **dashboard-style Streamlit interface** for easy input and results visualization.

---

## 🔍 Data Exploration & Feature Engineering

- Conducted exploratory data analysis (EDA) in a Jupyter Notebook (`HotelCancelPred_Final.ipynb`).
- Investigated trends in cancellations based on:
  - Guest type (adults/children)
  - Booking history (previous cancellations, repeated guests)
  - Stay details (weekend/weekday nights, lead time)
  - Special requests
- Feature engineering included:
  - Log transformation of `lead_time` and `average price per room`
  - Encoding categorical variables (room type, meal plan, market segment)

---

## 🤖 Model Development

- Compared multiple models in the notebook:
  - Logistic Regression
  - Random Forest
  - SVM
  - XGBoost
  - Decision Tree
- Selected the best-performing model based on accuracy, precision, recall, and ROC-AUC.
- The trained model is saved as a compressed `.pkl` file for use in the app.

---

## Check out the live app 
https://hotel-booking-cancellation-prediction-6qe3wgza9u9qslwe8kodxj.streamlit.app/
