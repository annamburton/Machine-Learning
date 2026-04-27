##1. Import 

import streamlit as st
import pandas as pd
import numpy as np
import torch

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from xgboost import XGBClassifier
from pytorch_tabnet.tab_model import TabNetClassifier

##2. Page setup

st.set_page_config(page_title="Restaurant Closure Predictor", layout="wide")

st.title("🍽️ Hybrid Restaurant Closure Prediction App")
st.write("Model = XGBoost + TabNet + GNN")

##3. Load Dataset



@st.cache_data
def load_data():
    return pd.read_csv("model_dataset.csv")

df = load_data()

##4. Basic Checks

print(df.columns.tolist())
print(df["target"].value_counts(dropna=False))
print(df["target"].value_counts(normalize=True, dropna=False))

##5. Define features and target

target_col = "target"

features = [
    "city","state","postal_code","RestaurantsPriceRange2",
    "primary_category","hours_open_per_week","days_open",
    "nearest_dist"
]

drop_cols = [
    'num_neighbors_1km', 'meal_options_count', 'has_nightlife',
    'music_options_count', 'is_bar_style', 'is_takeout_friendly',
    'has_breakfast', 'open_early', 'same_cuisine_neighbors',
    'business_stars', 'open_late', 'ambience_score',
    'has_seafood', 'has_fast_food', 'longitude',
    'is_date_spot', 'avg_neighbor_rating', 'has_italian',
    'is_full_service', 'parking_options_count', 'best_nights_count',
    'NoiseLevel', 'location_cluster'
]

df = df.drop(columns=drop_cols, errors="ignore")

features = [col for col in features if col not in drop_cols]

df_model = df[features + [target_col]].dropna()

X = df_model[features]
y = df_model[target_col]


##6. Handle missing values

# Drop highly missing columns (>70% missing)
missing_ratio = X.isnull().mean()
high_missing = missing_ratio[missing_ratio > 0.7].index

print("Dropping high-missing columns:", list(high_missing))

X = X.drop(columns=high_missing)

# Fill remaining numeric columns
for col in X.columns:
    if pd.api.types.is_numeric_dtype(X[col]):
        X[col] = X[col].fillna(X[col].median())
    else:
        X[col] = X[col].fillna("Unknown")

print("Remaining missing:", X.isnull().sum().sum())

##7. Preprocessing

categorical_features = ["city", "state", "primary_category","postal_code"]
numeric_features = [col for col in features if col not in categorical_features]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ]
)

##8. Train-Test Split and Modeling

@st.cache_resource
def train_models():
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    X_train_p = preprocessor.fit_transform(X_train)
    X_test_p = preprocessor.transform(X_test)

    X_train_p = X_train_p.toarray()
    X_test_p = X_test_p.toarray()

    # XGBoost
    xgb = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        eval_metric="logloss"
    )
    xgb.fit(X_train_p, y_train)

    # TabNet
    tabnet = TabNetClassifier()
    tabnet.fit(X_train_p, y_train)

    # GNN (simple neural net version)
    class GNN(torch.nn.Module):
        def __init__(self, input_dim):
            super().__init__()
            self.fc1 = torch.nn.Linear(input_dim, 64)
            self.fc2 = torch.nn.Linear(64, 32)
            self.out = torch.nn.Linear(32, 1)

        def forward(self, x):
            x = torch.relu(self.fc1(x))
            x = torch.relu(self.fc2(x))
            return self.out(x)

    gnn = GNN(X_train_p.shape[1])
    optimizer = torch.optim.Adam(gnn.parameters(), lr=0.001)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    X_tensor = torch.tensor(X_train_p, dtype=torch.float32)
    y_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)

    for epoch in range(10):
        optimizer.zero_grad()
        outputs = gnn(X_tensor)
        loss = loss_fn(outputs, y_tensor)
        loss.backward()
        optimizer.step()

    return xgb, tabnet, gnn, preprocessor

xgb_model, tabnet_model, gnn_model, preprocessor = train_models()

## 9. Input Features

st.sidebar.header("Input Features")

def select(col):
    return st.sidebar.selectbox(col, sorted(df[col].dropna().unique()))

def num(col):
    return st.sidebar.number_input(col, float(df[col].median()))

def slider(col):
    return st.sidebar.slider(col, float(df[col].min()), float(df[col].max()), float(df[col].median()))

input_data = {
    "city": select("city"),
    "state": select("state"),
    "postal_code": select("postal_code"),
    "primary_category": select("primary_category"),
    "RestaurantsPriceRange2": slider("RestaurantsPriceRange2"),
    "hours_open_per_week": num("hours_open_per_week"),
    "days_open": num("days_open"),
    "nearest_dist": num("nearest_dist")


}

input_df = pd.DataFrame([input_data])

## 10. Prediction Function

if st.button("Predict"):

    X_input = preprocessor.transform(input_df).toarray()

    # XGBoost
    xgb_prob = xgb_model.predict_proba(X_input)[0][1]

    # TabNet
    tabnet_prob = tabnet_model.predict_proba(X_input)[0][1]

    # GNN
    with torch.no_grad():
        gnn_tensor = torch.tensor(X_input, dtype=torch.float32)
        gnn_out = gnn_model(gnn_tensor)
        gnn_prob = torch.sigmoid(gnn_out).item()

    # Hybrid
    final_prob = 0.4*xgb_prob + 0.3*tabnet_prob + 0.3*gnn_prob
    prediction = 1 if final_prob >= 0.5 else 0

    st.subheader("Results")

    st.write("XGBoost:", round(xgb_prob, 3))
    st.write("TabNet:", round(tabnet_prob, 3))
    st.write("GNN:", round(gnn_prob, 3))

    st.metric("Final Probability", f"{final_prob:.2%}")

    if prediction == 1:
        st.success("✅ Likely to stay open")
    else:
        st.error("⚠️ Likely to close")