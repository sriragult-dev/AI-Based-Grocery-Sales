"""
AI Based Grocery Sales Prediction and Forecasting
Model: Random Forest Regressor (only)
Target: Item_Outlet_Sales
Dataset: K_class_sales_clean.csv

Run:  python grocery_sales_rf.py
Put the CSV in the same folder as this file.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # saves plots as images (no popup window needed)
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

RANDOM_STATE = 42
CSV_FILE = "K_class_sales_clean.csv"
CURRENT_YEAR = 2026


# --------------------------------------------------------------------
# STEP 1: LOAD DATA
# --------------------------------------------------------------------
print("\n=== STEP 1: LOAD DATA ===")
df = pd.read_csv(CSV_FILE)
print("Shape:", df.shape)
print(df.head())
print("\nMissing values:\n", df.isnull().sum())
print("\nSummary:\n", df.describe())


# --------------------------------------------------------------------
# STEP 2: CLEANING
# --------------------------------------------------------------------
print("\n=== STEP 2: CLEANING ===")

# Fix messy fat-content labels (only changes them if they exist)
if "Item_Fat_Content" in df.columns:
    df["Item_Fat_Content"] = df["Item_Fat_Content"].replace(
        {"LF": "Low Fat", "low fat": "Low Fat", "reg": "Regular"}
    )
    print(df["Item_Fat_Content"].value_counts())

# Fill any leftover missing values (safe even if there are none)
for col in df.columns:
    if df[col].isnull().any():
        if df[col].dtype == "object":
            df[col] = df[col].fillna(df[col].mode()[0])
        else:
            df[col] = df[col].fillna(df[col].median())

df = df.drop_duplicates()
print("Shape after cleaning:", df.shape)


# --------------------------------------------------------------------
# STEP 3: EDA (plots saved as PNG files)
# --------------------------------------------------------------------
print("\n=== STEP 3: EDA ===")

plt.figure(figsize=(7, 4))
sns.histplot(df["Item_Outlet_Sales"], kde=True)
plt.title("Distribution of Item_Outlet_Sales")
plt.tight_layout(); plt.savefig("eda_sales_distribution.png", dpi=120); plt.close()

plt.figure(figsize=(7, 4))
sns.scatterplot(x="Item_MRP", y="Item_Outlet_Sales", data=df, alpha=0.4)
plt.title("Item_MRP vs Sales")
plt.tight_layout(); plt.savefig("eda_mrp_vs_sales.png", dpi=120); plt.close()

plt.figure(figsize=(9, 4))
sns.boxplot(x="Outlet_Type", y="Item_Outlet_Sales", data=df)
plt.xticks(rotation=15)
plt.title("Sales by Outlet Type")
plt.tight_layout(); plt.savefig("eda_outlet_type_sales.png", dpi=120); plt.close()

plt.figure(figsize=(8, 6))
sns.heatmap(df.select_dtypes("number").corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.tight_layout(); plt.savefig("eda_correlation.png", dpi=120); plt.close()

print("EDA plots saved (eda_*.png)")


# --------------------------------------------------------------------
# STEP 4: FEATURE ENGINEERING
# --------------------------------------------------------------------
print("\n=== STEP 4: FEATURE ENGINEERING ===")

df["Outlet_Age"] = CURRENT_YEAR - df["Outlet_Establishment_Year"]
df["Item_Category"] = df["Item_Identifier"].str[:2]   # FD / DR / NC

# NOTE: Profit is NOT used as a feature (it is derived from sales -> data leakage)
drop_cols = ["Item_Outlet_Sales", "Profit", "Item_Identifier",
             "Outlet_Establishment_Year"]
X = df.drop(columns=[c for c in drop_cols if c in df.columns])
y = df["Item_Outlet_Sales"]

cat_cols = X.select_dtypes(include="object").columns.tolist()
num_cols = X.select_dtypes(include="number").columns.tolist()
print("Categorical:", cat_cols)
print("Numerical  :", num_cols)


# --------------------------------------------------------------------
# STEP 5: TRAIN / TEST SPLIT
# --------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE
)
print("\nTrain:", X_train.shape, " Test:", X_test.shape)


# --------------------------------------------------------------------
# STEP 6: PIPELINE (preprocessing + Random Forest)
# --------------------------------------------------------------------
pre = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ("num", "passthrough", num_cols),
])

rf_pipe = Pipeline([
    ("prep", pre),
    ("model", RandomForestRegressor(n_estimators=300,
                                    random_state=RANDOM_STATE,
                                    n_jobs=-1)),
])


def evaluate(model, name):
    tr_pred = model.predict(X_train)
    te_pred = model.predict(X_test)
    print(f"\n--- {name} ---")
    print(f"Train R2 : {r2_score(y_train, tr_pred):.4f}")
    print(f"Test  R2 : {r2_score(y_test, te_pred):.4f}")
    print(f"Test MAE : {mean_absolute_error(y_test, te_pred):.2f}")
    print(f"Test RMSE: {np.sqrt(mean_squared_error(y_test, te_pred)):.2f}")
    return te_pred


# --------------------------------------------------------------------
# STEP 7: BASELINE RANDOM FOREST
# --------------------------------------------------------------------
print("\n=== STEP 7: BASELINE RANDOM FOREST ===")
rf_pipe.fit(X_train, y_train)
evaluate(rf_pipe, "Baseline Random Forest")


# --------------------------------------------------------------------
# STEP 8: HYPERPARAMETER TUNING
# --------------------------------------------------------------------
print("\n=== STEP 8: TUNING (this may take a few minutes) ===")
param_grid = {
    "model__n_estimators":     [200, 300, 500],
    "model__max_depth":        [8, 12, 16, None],
    "model__min_samples_leaf": [3, 5, 10, 20],
    "model__max_features":     ["sqrt", 0.5, 1.0],
}

search = RandomizedSearchCV(
    rf_pipe, param_distributions=param_grid, n_iter=20, cv=5,
    scoring="neg_root_mean_squared_error",
    random_state=RANDOM_STATE, n_jobs=-1, verbose=1,
)
search.fit(X_train, y_train)
best = search.best_estimator_
print("\nBest parameters:", search.best_params_)

pred = evaluate(best, "Tuned Random Forest")


# --------------------------------------------------------------------
# STEP 9: FEATURE IMPORTANCE
# --------------------------------------------------------------------
print("\n=== STEP 9: FEATURE IMPORTANCE ===")
names = best.named_steps["prep"].get_feature_names_out()
imp = pd.Series(best.named_steps["model"].feature_importances_, index=names)
top = imp.sort_values(ascending=False).head(15)
print(top)

plt.figure(figsize=(9, 6))
top.sort_values().plot(kind="barh")
plt.title("Top 15 Feature Importances (Random Forest)")
plt.tight_layout(); plt.savefig("feature_importance.png", dpi=120); plt.close()


# --------------------------------------------------------------------
# STEP 10: ACTUAL vs PREDICTED
# --------------------------------------------------------------------
plt.figure(figsize=(6, 6))
plt.scatter(y_test, pred, alpha=0.4)
lims = [y.min(), y.max()]
plt.plot(lims, lims, "r--")
plt.xlabel("Actual Sales"); plt.ylabel("Predicted Sales")
plt.title("Actual vs Predicted")
plt.tight_layout(); plt.savefig("actual_vs_predicted.png", dpi=120); plt.close()

# Residuals
plt.figure(figsize=(7, 4))
sns.histplot(y_test - pred, kde=True)
plt.title("Residuals (Actual - Predicted)")
plt.tight_layout(); plt.savefig("residuals.png", dpi=120); plt.close()
print("Plots saved: feature_importance.png, actual_vs_predicted.png, residuals.png")


# --------------------------------------------------------------------
# STEP 11: FORECASTING / WHAT-IF SCENARIOS
# --------------------------------------------------------------------
print("\n=== STEP 11: WHAT-IF FORECASTING ===")

# Build a sample row using the most common values from your own data
# (so category names always match your file)
sample = {}
for c in cat_cols:
    sample[c] = X[c].mode()[0]
for c in num_cols:
    sample[c] = float(X[c].median())

# Scenario A: how does predicted sales change with price (MRP)?
mrp_values = np.linspace(X["Item_MRP"].min(), X["Item_MRP"].max(), 10)
rows = []
for m in mrp_values:
    r = sample.copy(); r["Item_MRP"] = m
    rows.append(r)
scen_mrp = pd.DataFrame(rows)[X.columns]
scen_mrp["Predicted_Sales"] = best.predict(scen_mrp)
print("\nScenario A: Price (MRP) vs Predicted Sales")
print(scen_mrp[["Item_MRP", "Predicted_Sales"]].round(2))

plt.figure(figsize=(7, 4))
plt.plot(scen_mrp["Item_MRP"], scen_mrp["Predicted_Sales"], marker="o")
plt.xlabel("Item_MRP"); plt.ylabel("Predicted Sales")
plt.title("What-if: Price vs Predicted Sales")
plt.tight_layout(); plt.savefig("scenario_mrp.png", dpi=120); plt.close()

# Scenario B: how does predicted sales change with outlet type?
rows = []
for t in X["Outlet_Type"].unique():
    r = sample.copy(); r["Outlet_Type"] = t
    rows.append(r)
scen_out = pd.DataFrame(rows)[X.columns]
scen_out["Predicted_Sales"] = best.predict(scen_out)
print("\nScenario B: Outlet Type vs Predicted Sales")
print(scen_out[["Outlet_Type", "Predicted_Sales"]].round(2))

# Scenario C: your own custom prediction (edit the values below)
custom = sample.copy()
custom["Item_MRP"] = 150.0
# custom["Outlet_Type"] = "Supermarket Type1"   # uncomment / change if needed
custom_df = pd.DataFrame([custom])[X.columns]
print("\nCustom prediction:", round(best.predict(custom_df)[0], 2))


# --------------------------------------------------------------------
# STEP 12: SAVE MODEL
# --------------------------------------------------------------------
joblib.dump(best, "rf_sales_model.pkl")
print("\nModel saved as rf_sales_model.pkl")
print("Load later with: model = joblib.load('rf_sales_model.pkl')")
print("\nDONE.")
