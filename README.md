
# 🛠️ IT Support Ticket Resolution Time Prediction System

This project predicts the resolution time (in hours) for incoming IT support tickets using historical data and machine learning, helping IT service teams improve SLA compliance, resource allocation, and triage efficiency.

---

## 🚀 Features

- 📊 ML-powered prediction using **XGBoost Regressor**
- 📝 Text-based ticket description processing via **TF-IDF**
- 🔍 Feature engineering: priority, category, department, and timestamp patterns
- 🌐 REST API for real-time predictions using **Flask**
- 🎛️ Interactive UI using **Streamlit** (with Light/Dark mode)
- ☁️ Ready for cloud deployment (AWS Lambda, Streamlit Cloud)

---

## 🎯 Objective

IT service teams often face unpredictable resolution times, leading to SLA violations and poor user satisfaction. This system uses past ticket data to **predict the expected resolution time** as soon as a new ticket is created.

---

## 📁 Project Structure

```
ticket-resolution-predictor/
│
├── data/
│   └── tickets.csv                  # Sample training data
│
├── models/
│   ├── xgb_model.joblib             # Trained ML model
│   └── tfidf_vectorizer.joblib      # TF-IDF vectorizer
│
├── utils/
│   └── preprocessing.py             # Data cleaning & feature engineering
│
├── main.py                          # Model training and evaluation
├── app.py                           # Flask API
├── app_streamlit.py                 # Streamlit UI app
├── requirements.txt                 # Dependencies
└── README.md                        # You're here!
```

---

## 🧪 Tech Stack

- **Python**, **Pandas**, **Scikit-learn**, **XGBoost**
- **TF-IDF (NLP)** for processing ticket descriptions
- **Flask** for REST API
- **Streamlit** for interactive web UI
- **Joblib** for model serialization
- **AWS Lambda + S3** (optional deployment)
- **Postman** for API testing

---

## 📊 Model Performance

| Metric               | Value       |
|----------------------|-------------|
| MAE (Mean Abs Error) | ~0.8 hours  |
| RMSE                 | ~1.3 hours  |
| R² Score             | 0.87        |

> Best performance achieved with `XGBoostRegressor` after hyperparameter tuning.

---

## 🧠 How It Works

1. **Input**: Priority, category, department, created time, and ticket description
2. **Text Vectorization**: Description is transformed using TF-IDF
3. **Feature Engineering**: Adds useful features like description length, hour of creation, day of week, etc.
4. **Prediction**: ML model predicts resolution time in hours

---

## ⚙️ Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/samruddhi-08/ticket-resolution-predictor.git
cd ticket-resolution-predictor
```

### 2. Create and Activate Virtual Environment

```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Train the Model (optional if using saved one)

```bash
python main.py
```

---

## 💡 Run the Apps

### 🌐 1. Flask API

```bash
uvicorn app:app --reload
```

**POST request to `/predict`:**

```json
{
  "Ticket ID": "T123",
  "Created Time": "2025-06-23 10:00:00",
  "Priority": "High",
  "Category": "Software",
  "Department": "IT",
  "Description": "Outlook is not working and emails are delayed."
}
```

### 🖥️ 2. Streamlit UI

```bash
streamlit run app_streamlit.py
```

---

## 📈 Example Use Case

- **New ticket arrives**
- User enters ticket details in UI or POSTs to API
- System predicts resolution time instantly (e.g., 4.6 hours)
- Team triages or escalates based on predicted SLA risk

---

## 📦 Dependencies

```
streamlit
pandas
scikit-learn
xgboost
joblib
uvicorn
fastapi
```

---

## 📌 Author

**Samruddhi Deore**  
Data Engineer | Data Scientist 
[LinkedIn](https://www.linkedin.com/in/samruddhi-deore-7700281ba/) | [GitHub](https://github.com/samruddhi-08)




