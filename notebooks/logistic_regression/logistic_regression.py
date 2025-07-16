#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve, f1_score
from sklearn.decomposition import PCA


# In[3]:


df = pd.read_csv("C:/Users/sivas/OneDrive/Documents/GitHub/WorkflowForDataScience/datasets/model_input_flight_delay.csv")


# In[4]:


df.head()


# In[6]:


# Example target column (already binary)
target = 'DELAY_STATUS'

# Choose relevant predictors
features = [
    'AIRLINE', 'ORIGIN', 'DEST',
    'CRS_DEP_TIME', 'CRS_ARR_TIME',
    'CRS_ELAPSED_TIME', 'DISTANCE',
    'FL_DATE_day_name', 'FL_DATE_is_weekend'
]

df_model = df[features + [target]].dropna()


# In[14]:


df_model.head()


# In[7]:


X = pd.get_dummies(df_model[features], drop_first=True)  # One-hot encoding
y = df_model[target]


# In[8]:


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


# In[9]:


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# In[ ]:


lr = LogisticRegression(max_iter=1000, solver='liblinear')
lr.fit(X_train_scaled, y_train)

y_pred = lr.predict(X_test_scaled)
y_proba = lr.predict_proba(X_test_scaled)[:, 1]

print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("ROC-AUC Score:", roc_auc_score(y_test, y_proba))

importance = pd.Series(lr.coef_[0], index=X.columns).sort_values(key=abs, ascending=False)
print(importance.head(10))


# In[ ]:


lr = LogisticRegression(class_weight='balanced', max_iter=1000, solver='liblinear')

lr.fit(X_train_scaled, y_train)

y_pred = lr.predict(X_test_scaled)
y_proba = lr.predict_proba(X_test_scaled)[:, 1]

print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("ROC-AUC Score:", roc_auc_score(y_test, y_proba))

importance = pd.Series(lr.coef_[0], index=X.columns).sort_values(key=abs, ascending=False)
print(importance.head(10))


# In[19]:


y_proba = lr.predict_proba(X_test_scaled)[:, 1]
precision, recall, thresholds = precision_recall_curve(y_test, y_proba)

# Try a new threshold (e.g., 0.3)
new_preds = (y_proba > 0.3).astype(int)
print("ROC-AUC Score:", roc_auc_score(y_test, new_preds))


# In[20]:


for solver in ['lbfgs', 'liblinear', 'saga']:
    model = LogisticRegression(solver=solver, class_weight='balanced', max_iter=1000)
    model.fit(X_train_scaled, y_train)
    score = model.score(X_test_scaled, y_test)
    print(f"{solver} accuracy: {score:.4f}")


# In[22]:


for solver in ['liblinear', 'lbfgs', 'saga']:
    model = LogisticRegression(solver=solver, class_weight='balanced', max_iter=1000)
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]
    print(f"{solver} - F1: {f1_score(y_test, y_pred):.4f}, ROC-AUC: {roc_auc_score(y_test, y_proba):.4f}")


# In[24]:


param_grid = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100],
    'penalty': ['l2'],  # 'l1' and 'elasticnet' require saga/liblinear
    'solver': ['lbfgs']
}

grid = GridSearchCV(LogisticRegression(class_weight='balanced', max_iter=1000), param_grid, scoring='roc_auc', cv=3)
grid.fit(X_train_scaled, y_train)
print("Best AUC:", grid.best_score_)
print("Best Params:", grid.best_params_)


# In[27]:


def time_of_day(hour):
    if 5 <= hour < 10:
        return 'morning'
    elif 10 <= hour < 15:
        return 'midday'
    elif 15 <= hour < 20:
        return 'evening'
    else:
        return 'late_night'


# In[ ]:


df['dep_hour'] = df['CRS_DEP_TIME'] // 100
df['dep_time_bucket'] = df['dep_hour'].apply(time_of_day)
top_airports = df['ORIGIN'].value_counts().head(10).index
df['is_busy_origin'] = df['ORIGIN'].isin(top_airports).astype(int)


# In[29]:


top_destinations = df['DEST'].value_counts().head(10).index
df['is_busy_destination'] = df['DEST'].isin(top_destinations).astype(int)


# In[31]:


# Example target column (already binary)
target = 'DELAY_STATUS'

# Choose relevant predictors
features = [
    'AIRLINE', 'ORIGIN', 'DEST',
    'CRS_DEP_TIME', 'CRS_ARR_TIME',
    'CRS_ELAPSED_TIME', 'DISTANCE',
    'FL_DATE_day_name', 'FL_DATE_is_weekend',
    'dep_hour',
    'is_busy_origin',
    'is_busy_destination',
    'dep_time_bucket'
]

df_model = df[features + [target]].dropna()


# In[32]:


X = pd.get_dummies(df_model[features], drop_first=True)  # One-hot encoding
y = df_model[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

lr = LogisticRegression(class_weight='balanced', max_iter=1000, solver="lbfgs", penalty="l2", C=0.01)
lr.fit(X_train_scaled, y_train)

y_pred = lr.predict(X_test_scaled)
y_proba = lr.predict_proba(X_test_scaled)[:, 1]

print(classification_report(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_proba))


# In[37]:


importance = pd.Series(lr.coef_[0], index=X.columns).sort_values(key=abs, ascending=False)
print(importance.head(10))


# In[40]:


minimal_cols = [
    'DELAY_STATUS',         # target
    'CRS_DEP_TIME',
    'DISTANCE',
    'CRS_ELAPSED_TIME',
    'FL_DATE_is_weekend',
    'AIRLINE'
    # optionally: 'DEST'
]

df_min = df[minimal_cols].dropna()
df_encoded = pd.get_dummies(df_min, columns=['AIRLINE'], drop_first=True)
X = df_encoded.drop('DELAY_STATUS', axis=1)
y = df_encoded['DELAY_STATUS']

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

lr = LogisticRegression(class_weight='balanced', max_iter=1000)
lr.fit(X_train_scaled, y_train)

y_pred = lr.predict(X_test_scaled)
y_proba = lr.predict_proba(X_test_scaled)[:, 1]

print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_proba))
importance = pd.Series(lr.coef_[0], index=X.columns).sort_values(key=abs, ascending=False)
print(importance.head(10))


# In[42]:


# Drop target + categorical features for PCA
numeric_cols = [
    'CRS_DEP_TIME', 'DISTANCE', 'CRS_ELAPSED_TIME',
    'DEP_DELAY', 'TAXI_OUT', 'TAXI_IN', 'AIR_TIME',
    'FL_DATE_is_weekend'
]

df_pca = df[numeric_cols + ['DELAY_STATUS']].dropna()
X_numeric = df_pca[numeric_cols]
y = df_pca['DELAY_STATUS']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_numeric)

# Try to retain 95% of variance
pca = PCA(n_components=0.95)
X_pca = pca.fit_transform(X_scaled)

print(f"PCA reduced from {X_scaled.shape[1]} to {X_pca.shape[1]} components")

# Optional: plot explained variance
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel("Number of Components")
plt.ylabel("Cumulative Explained Variance")
plt.grid(True)
plt.tight_layout()
plt.show()

X_train, X_test, y_train, y_test = train_test_split(X_pca, y, stratify=y, test_size=0.2, random_state=42)

lr = LogisticRegression(class_weight='balanced', max_iter=1000)
lr.fit(X_train, y_train)

y_pred = lr.predict(X_test)
y_proba = lr.predict_proba(X_test)[:, 1]

print(classification_report(y_test, y_pred))
print("ROC-AUC Score:", roc_auc_score(y_test, y_proba))


# In[44]:


print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))


# In[48]:


# Shape: [n_pca_components x n_original_features]
pca_components = pd.DataFrame(pca.components_, columns=numeric_cols)

# Logistic regression weights for each PCA component
lr_weights = lr.coef_[0]  # shape: [n_pca_components]

# Combine: weighted sum of component directions
feature_importance = np.dot(lr_weights, pca_components)

# Convert to Series
importance_series = pd.Series(feature_importance, index=numeric_cols).sort_values(key=abs, ascending=False)

print(importance_series.head(10))


# In[49]:


# Drop target + categorical features for PCA
numeric_cols = [
    'CRS_DEP_TIME', 'DISTANCE', 'CRS_ELAPSED_TIME',
    'TAXI_OUT', 'TAXI_IN', 'AIR_TIME',
    'FL_DATE_is_weekend'
]

df_pca = df[numeric_cols + ['DELAY_STATUS']].dropna()
X_numeric = df_pca[numeric_cols]
y = df_pca['DELAY_STATUS']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_numeric)

# Try to retain 95% of variance
pca = PCA(n_components=0.95)
X_pca = pca.fit_transform(X_scaled)

print(f"PCA reduced from {X_scaled.shape[1]} to {X_pca.shape[1]} components")

# Optional: plot explained variance
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel("Number of Components")
plt.ylabel("Cumulative Explained Variance")
plt.grid(True)
plt.tight_layout()
plt.show()

X_train, X_test, y_train, y_test = train_test_split(X_pca, y, stratify=y, test_size=0.2, random_state=42)

lr = LogisticRegression(class_weight='balanced', max_iter=1000)
lr.fit(X_train, y_train)

y_pred = lr.predict(X_test)
y_proba = lr.predict_proba(X_test)[:, 1]

print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print("ROC-AUC Score:", roc_auc_score(y_test, y_proba))


# In[50]:


# Shape: [n_pca_components x n_original_features]
pca_components = pd.DataFrame(pca.components_, columns=numeric_cols)

# Logistic regression weights for each PCA component
lr_weights = lr.coef_[0]  # shape: [n_pca_components]

# Combine: weighted sum of component directions
feature_importance = np.dot(lr_weights, pca_components)

# Convert to Series
importance_series = pd.Series(feature_importance, index=numeric_cols).sort_values(key=abs, ascending=False)

print(importance_series.head(10))


# In[51]:


# Drop target + categorical features for PCA
numeric_cols = [
    'CRS_DEP_TIME', 'DISTANCE', 'CRS_ELAPSED_TIME',
    'AIR_TIME',
    'FL_DATE_is_weekend'
]

df_pca = df[numeric_cols + ['DELAY_STATUS']].dropna()
X_numeric = df_pca[numeric_cols]
y = df_pca['DELAY_STATUS']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_numeric)

# Try to retain 95% of variance
pca = PCA(n_components=0.95)
X_pca = pca.fit_transform(X_scaled)

print(f"PCA reduced from {X_scaled.shape[1]} to {X_pca.shape[1]} components")

# Optional: plot explained variance
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel("Number of Components")
plt.ylabel("Cumulative Explained Variance")
plt.grid(True)
plt.tight_layout()
plt.show()

X_train, X_test, y_train, y_test = train_test_split(X_pca, y, stratify=y, test_size=0.2, random_state=42)

lr = LogisticRegression(class_weight='balanced', max_iter=1000)
lr.fit(X_train, y_train)

y_pred = lr.predict(X_test)
y_proba = lr.predict_proba(X_test)[:, 1]

print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print("ROC-AUC Score:", roc_auc_score(y_test, y_proba))

# Shape: [n_pca_components x n_original_features]
pca_components = pd.DataFrame(pca.components_, columns=numeric_cols)

# Logistic regression weights for each PCA component
lr_weights = lr.coef_[0]  # shape: [n_pca_components]

# Combine: weighted sum of component directions
feature_importance = np.dot(lr_weights, pca_components)

# Convert to Series
importance_series = pd.Series(feature_importance, index=numeric_cols).sort_values(key=abs, ascending=False)

print(importance_series.head(10))


# In[72]:


# Drop target + categorical features for PCA
features = [
    'CRS_DEP_TIME', 'DISTANCE', 'CRS_ELAPSED_TIME',
    'AIR_TIME',
    'FL_DATE_is_weekend', 'dep_hour','dep_time_bucket','is_busy_destination','FL_DATE_day_name'
]

df_subset = df[features + ['DELAY_STATUS']].dropna()

df_encoded = pd.get_dummies(df_subset, columns=['dep_time_bucket','FL_DATE_day_name'], drop_first=True)

X = df_encoded.drop('DELAY_STATUS', axis=1)
y = df_encoded['DELAY_STATUS']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Try to retain 95% of variance
pca = PCA(n_components=0.95)
X_pca = pca.fit_transform(X_scaled)

print(f"PCA reduced from {X_scaled.shape[1]} to {X_pca.shape[1]} components")

# Optional: plot explained variance
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel("Number of Components")
plt.ylabel("Cumulative Explained Variance")
plt.grid(True)
plt.tight_layout()
plt.show()

X_train, X_test, y_train, y_test = train_test_split(X_pca, y, stratify=y, test_size=0.2, random_state=42)

lr = LogisticRegression(class_weight='balanced', max_iter=1000)
lr.fit(X_train, y_train)

y_pred = lr.predict(X_test)
y_proba = lr.predict_proba(X_test)[:, 1]

print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print("ROC-AUC Score:", roc_auc_score(y_test, y_proba))

pca_importance = pd.Series(np.dot(lr.coef_[0], pca.components_), index=X.columns).sort_values(key=abs, ascending=False)
print(pca_importance.head(10))


# In[ ]:




