import streamlit as st
import joblib
import pandas as pd
import datetime

# Load model and vectorizer
model = joblib.load("models/xgb_model.joblib")
vectorizer = joblib.load("models/tfidf_vectorizer.joblib")

# ---------- PAGE CONFIG ----------
st.set_page_config(
    page_title="IT Ticket Predictor",
    page_icon="🛠️",
    layout="centered"
)

# ---------- HEADER ----------
st.markdown("<h1 style='text-align: center; color: #2C3E50;'>🛠️ IT Ticket Resolution Time Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: gray;'>Predict how long it will take to resolve IT support tickets using ML.</p>", unsafe_allow_html=True)
st.markdown("---")

# ---------- INPUT FORM ----------
with st.form("predict_form"):
    st.subheader("📋 Ticket Details")

    col1, col2 = st.columns(2)

    with col1:
        ticket_id = st.text_input("🎫 Ticket ID", "TCKT0001")
        priority = st.selectbox("⚠️ Priority", ["High", "Medium", "Low"])
        category = st.selectbox("📂 Category", ["Hardware", "Software", "Network"])
    
    with col2:
        department = st.selectbox("🏢 Department", ["IT", "HR", "Finance"])
        date = st.date_input("📅 Created Date", datetime.date.today())
        time = st.time_input("⏰ Created Time", datetime.datetime.now().time())

    description = st.text_area("📝 Description", "Outlook not working properly")

    submitted = st.form_submit_button("🔮 Predict Resolution Time")

# ---------- PREDICTION ----------
if submitted:
    created_time = datetime.datetime.combine(date, time)

    df = pd.DataFrame([{
        "Ticket ID": ticket_id,
        "Created Time": created_time,
        "Priority": priority,
        "Category": category,
        "Department": department,
        "Description": description
    }])

    # Feature engineering
    df["Description Length"] = df["Description"].apply(lambda x: len(str(x)))
    df["Created Hour"] = pd.to_datetime(df["Created Time"]).dt.hour
    df["Created DayOfWeek"] = pd.to_datetime(df["Created Time"]).dt.dayofweek

    # Encode categorical variables
    mapping = {
        "Priority": {"Low": 0, "Medium": 1, "High": 2},
        "Category": {"Hardware": 0, "Network": 1, "Software": 2},
        "Department": {"Finance": 0, "HR": 1, "IT": 2}
    }

    for col in ["Priority", "Category", "Department"]:
        df[col] = df[col].map(mapping[col])

    tfidf_features = vectorizer.transform(df["Description"]).toarray()
    tfidf_df = pd.DataFrame(tfidf_features, columns=vectorizer.get_feature_names_out())
    df = pd.concat([df.reset_index(drop=True), tfidf_df], axis=1)

    X = df[["Priority", "Category", "Department", "Description Length", "Created Hour", "Created DayOfWeek"] + list(tfidf_df.columns)]

    prediction = model.predict(X)[0]

    st.markdown("---")
    st.success("✅ Prediction Complete!")

    st.metric(label="🕒 Estimated Resolution Time", value=f"{round(float(prediction), 2)} hours")
    st.balloons()

# ---------- FOOTER ----------
with st.expander("ℹ️ About this App"):
    st.markdown("""
        - Built with **Streamlit**
        - ML model: **XGBoost Regressor**
        - Features used: Priority, Category, Department, TF-IDF on description, etc.
        - Author: Samruddhi
    """)
