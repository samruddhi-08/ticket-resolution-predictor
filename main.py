import os
os.makedirs("models", exist_ok=True)
from utils.preprocessing import load_data_from_csv, clean_and_engineer_data, tfidf_transform, get_features_and_target
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
import joblib

# 1. Load and preprocess data
df = load_data_from_csv("data/tickets.csv")
df = clean_and_engineer_data(df)
df = tfidf_transform(df, fit=True)
df.fillna(0, inplace=True)  # Add this line

X, y = get_features_and_target(df)

# 2. Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Train models
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(),
    "XGBoost": XGBRegressor(verbosity=0)
}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred) ** 0.5
    r2 = r2_score(y_test, y_pred)
    
    print(f"\n🧠 Model: {name}")
    print(f"MAE:  {mae:.2f} hours")
    print(f"RMSE: {rmse:.2f} hours")
    print(f"R² Score: {r2:.2f}")

# 4. Save best model (e.g., XGBoost)
best_model = models["XGBoost"]
joblib.dump(best_model, "models/xgb_model.joblib")
print("\n✅ Best model saved to models/xgb_model.joblib")
