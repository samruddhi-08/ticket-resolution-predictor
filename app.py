import joblib
from flask import Flask, request, jsonify
import pandas as pd

# Load trained model and vectorizer
model = joblib.load("models/xgb_model.joblib")
vectorizer = joblib.load("models/tfidf_vectorizer.joblib")

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        input_df = pd.DataFrame([data])

        input_df['Created Time'] = pd.to_datetime(input_df['Created Time'])
        input_df['Description Length'] = input_df['Description'].apply(lambda x: len(str(x)))
        input_df['Created Hour'] = input_df['Created Time'].dt.hour
        input_df['Created DayOfWeek'] = input_df['Created Time'].dt.dayofweek

        priority_map = {'Low': 0, 'Medium': 1, 'High': 2, 'Urgent': 3}
        category_map = {'Hardware': 0, 'Software': 1, 'Network': 2, 'Access Request': 3, 'Bug Report': 4}
        department_map = {'IT': 0, 'HR': 1, 'Finance': 2, 'Engineering': 3, 'Support': 4}

        input_df['Priority'] = input_df['Priority'].map(priority_map)
        input_df['Category'] = input_df['Category'].map(category_map)
        input_df['Department'] = input_df['Department'].map(department_map)

        # ✅ Reuse the same vectorizer from training
        tfidf_matrix = vectorizer.transform(input_df['Description'].fillna(''))
        tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=vectorizer.get_feature_names_out())
        input_df = pd.concat([input_df.reset_index(drop=True), tfidf_df.reset_index(drop=True)], axis=1)

        # Drop unused columns
        input_df.drop(columns=['Ticket ID', 'Created Time', 'Resolved Time', 'Resolution Time', 'Description'], inplace=True, errors='ignore')

        prediction = model.predict(input_df)[0]
        return jsonify({'predicted_resolution_time_hours': float(round(prediction, 2))})

    except Exception as e:
        return jsonify({'error': str(e)})


if __name__ == '__main__':
    app.run(debug=True)