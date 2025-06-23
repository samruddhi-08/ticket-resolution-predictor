import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import joblib

def load_data_from_csv(path):
    df = pd.read_csv(path)
    return df

def clean_and_engineer_data(df):
    # Drop rows with missing essential values
    df.dropna(subset=['Created Time', 'Resolved Time', 'Priority', 'Category', 'Department', 'Description'], inplace=True)

    # Calculate resolution time in hours
    df['Created Time'] = pd.to_datetime(df['Created Time'])
    df['Resolved Time'] = pd.to_datetime(df['Resolved Time'])
    df['Resolution Time'] = (df['Resolved Time'] - df['Created Time']).dt.total_seconds() / 3600

    # Remove outliers
    df = df[df['Resolution Time'] < df['Resolution Time'].quantile(0.95)].copy()


    # Encode categorical variables
    for col in ['Priority', 'Category', 'Department']:
        le = LabelEncoder()
        df.loc[:, col] = pd.Series(le.fit_transform(df[col])).astype(int)




    # Create new features
    df.loc[:, 'Description Length'] = df['Description'].apply(lambda x: len(str(x)))
    df.loc[:, 'Created Hour'] = df['Created Time'].dt.hour
    df.loc[:, 'Created DayOfWeek'] = df['Created Time'].dt.dayofweek


    return df

def tfidf_transform(df, max_features=100, vectorizer=None, fit=True):
    if vectorizer is None and fit:
        tfidf = TfidfVectorizer(max_features=max_features, stop_words='english')
        tfidf_matrix = tfidf.fit_transform(df['Description'].fillna(''))
        joblib.dump(tfidf, 'models/tfidf_vectorizer.joblib')  # ✅ Save the trained vectorizer
    else:
        tfidf = joblib.load('models/tfidf_vectorizer.joblib')
        tfidf_matrix = tfidf.transform(df['Description'].fillna(''))

    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf.get_feature_names_out())
    df = df.reset_index(drop=True)
    df_tfidf = pd.concat([df, tfidf_df], axis=1)
    return df_tfidf


def get_features_and_target(df):
    X = df.drop(columns=['Ticket ID', 'Created Time', 'Resolved Time', 'Resolution Time', 'Description'])
    y = df['Resolution Time']
    return X, y
