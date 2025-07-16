# --- feature_selection.py ---
from sklearn.ensemble import RandomForestRegressor
import pandas as pd

def select_top_features(X, y, top_n=30):
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X, y)
    importances = rf.feature_importances_
    feature_importance = pd.Series(importances, index=X.columns)
    top_features = feature_importance.sort_values(ascending=False).head(top_n).index.tolist()
    return X[top_features], top_features