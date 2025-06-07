import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OrdinalEncoder

def preprocess_data(df):
    features = df.drop(columns=["salary", "salary_currency", "salary_in_usd"])
    target = df["salary_in_usd"]

    cat_cols = features.select_dtypes(include="object").columns
    encoder = OrdinalEncoder()
    features[cat_cols] = encoder.fit_transform(features[cat_cols])

    return features, target, encoder, cat_cols

def train_model(X, y):
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def encode_input(df, encoder, cat_cols):
    df_encoded = df.copy()
    df_encoded[cat_cols] = encoder.transform(df_encoded[cat_cols])
    return df_encoded
