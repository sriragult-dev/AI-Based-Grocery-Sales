"""
AI Based Grocery Sales Prediction & Forecasting  -  Streamlit frontend
Model: Random Forest Regressor (only)

Run locally:   streamlit run app.py
Needs in same folder: K_class_sales_clean.csv
"""

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

CSV_FILE = "K_class_sales_clean.csv"
CURRENT_YEAR = 2026
SEED = 42
DROP_COLS = ["Item_Outlet_Sales", "Profit", "Item_Identifier",
             "Outlet_Establishment_Year"]   # Profit dropped on purpose (leakage)

st.set_page_config(page_title="Grocery Sales Predictor", page_icon="🛒", layout="wide")


# ------------------------------------------------------------------
# DATA
# ------------------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv(CSV_FILE)
    df["Item_Fat_Content"] = df["Item_Fat_Content"].replace(
        {"LF": "Low Fat", "low fat": "Low Fat", "reg": "Regular"})
    for col in df.columns:
        if df[col].isnull().any():
            if df[col].dtype == "object":
                df[col] = df[col].fillna(df[col].mode()[0])
            else:
                df[col] = df[col].fillna(df[col].median())
    df = df.drop_duplicates()
    df["Outlet_Age"] = CURRENT_YEAR - df["Outlet_Establishment_Year"]
    df["Item_Category"] = df["Item_Identifier"].str[:2]
    return df


# ------------------------------------------------------------------
# MODEL (trained once, then cached)
# ------------------------------------------------------------------
@st.cache_resource(show_spinner="Training the Random Forest model (first load only)...")
def train_model():
    df = load_data()
    X = df.drop(columns=[c for c in DROP_COLS if c in df.columns])
    y = df["Item_Outlet_Sales"]
    cat_cols = X.select_dtypes(include="object").columns.tolist()
    num_cols = X.select_dtypes(include="number").columns.tolist()

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=SEED)

    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ("num", "passthrough", num_cols),
    ])
    # Tip: replace these with the best params printed by grocery_sales_rf.py
    rf = RandomForestRegressor(n_estimators=300, max_depth=12, min_samples_leaf=5,
                               max_features=0.5, random_state=SEED, n_jobs=-1)
    pipe = Pipeline([("prep", pre), ("model", rf)])
    pipe.fit(X_tr, y_tr)

    pred = pipe.predict(X_te)
    metrics = {
        "R2": r2_score(y_te, pred),
        "MAE": mean_absolute_error(y_te, pred),
        "RMSE": float(np.sqrt(mean_squared_error(y_te, pred))),
        "Train R2": r2_score(y_tr, pipe.predict(X_tr)),
    }
    results = pd.DataFrame({"Actual": y_te.values, "Predicted": pred})

    # feature importance, grouped back to original column names
    names = pipe.named_steps["prep"].get_feature_names_out()
    imp = pd.Series(rf.feature_importances_, index=names)
    grouped = {}
    for col in cat_cols:
        grouped[col] = imp[[n for n in names if n.startswith(f"cat__{col}_")]].sum()
    for col in num_cols:
        grouped[col] = imp[f"num__{col}"]
    importance = pd.Series(grouped).sort_values(ascending=False)

    return pipe, list(X.columns), metrics, results, importance


df = load_data()
model, FEATURES, metrics, results, importance = train_model()

# lookups so the user only picks an outlet and the rest fills automatically
outlet_info = (df.groupby("Outlet_Identifier")
                 [["Outlet_Size", "Outlet_Location_Type", "Outlet_Type", "Outlet_Age"]]
                 .first())
item_cat_map = (df.groupby("Item_Type")["Item_Category"]
                  .agg(lambda s: s.mode()[0]).to_dict())


def make_row(item_type, fat, weight, visibility, mrp, outlet_id):
    o = outlet_info.loc[outlet_id]
    row = {
        "Item_Weight": weight,
        "Item_Fat_Content": fat,
        "Item_Visibility": visibility,
        "Item_Type": item_type,
        "Item_MRP": mrp,
        "Outlet_Identifier": outlet_id,
        "Outlet_Size": o["Outlet_Size"],
        "Outlet_Location_Type": o["Outlet_Location_Type"],
        "Outlet_Type": o["Outlet_Type"],
        "Outlet_Age": o["Outlet_Age"],
        "Item_Category": item_cat_map[item_type],
    }
    return pd.DataFrame([row])[FEATURES]


def predict_many(rows):
    return model.predict(pd.concat(rows, ignore_index=True))


# ------------------------------------------------------------------
# SIDEBAR  (inputs shared by all tabs)
# ------------------------------------------------------------------
st.sidebar.title("🛒 Product & Outlet")
item_type = st.sidebar.selectbox("Item type", sorted(df["Item_Type"].unique()))
fat = st.sidebar.selectbox("Fat content", sorted(df["Item_Fat_Content"].unique()))
weight = st.sidebar.number_input("Item weight (kg)", 0.0, 50.0,
                                 float(round(df["Item_Weight"].median(), 2)), 0.1)
visibility = st.sidebar.slider("Item visibility", 0.0, float(df["Item_Visibility"].max()),
                               float(df["Item_Visibility"].median()), 0.005,
                               help="Share of display area the item gets in the store")
mrp = st.sidebar.slider("Item MRP (price)", float(df["Item_MRP"].min()),
                        float(df["Item_MRP"].max()), float(df["Item_MRP"].median()), 1.0)
outlet_id = st.sidebar.selectbox("Outlet", list(outlet_info.index))
o = outlet_info.loc[outlet_id]
st.sidebar.caption(f"{o['Outlet_Type']} · {o['Outlet_Size']} · "
                   f"{o['Outlet_Location_Type']} · {int(o['Outlet_Age'])} yrs old")

# ------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------
st.title("AI Based Grocery Sales Prediction & Forecasting")
st.caption("Random Forest model trained on item and outlet data. "
           "Change the inputs in the sidebar and see the prediction update.")

tab1, tab2, tab3, tab4 = st.tabs(
    ["🔮 Predict", "📈 What-if Forecast", "🎯 Model Performance", "📊 Data Explorer"])

# ---------------- TAB 1: PREDICT ----------------
with tab1:
    row = make_row(item_type, fat, weight, visibility, mrp, outlet_id)
    point = float(model.predict(row)[0])

    # spread across the individual trees gives a rough confidence range
    Xt = model.named_steps["prep"].transform(row)
    tree_preds = np.array([t.predict(Xt)[0]
                           for t in model.named_steps["model"].estimators_])
    low, high = np.percentile(tree_preds, [10, 90])

    c1, c2, c3 = st.columns(3)
    c1.metric("Predicted sales", f"₹ {point:,.0f}")
    c2.metric("Likely range (10th–90th pct)", f"₹ {low:,.0f} – {high:,.0f}")
    avg = df.loc[df["Item_Type"] == item_type, "Item_Outlet_Sales"].mean()
    c3.metric(f"Avg for {item_type}", f"₹ {avg:,.0f}", f"{point - avg:+,.0f} vs avg")

    st.subheader("Spread of predictions across the 300 trees")
    hist, edges = np.histogram(tree_preds, bins=25)
    st.bar_chart(pd.DataFrame({"Trees": hist},
                              index=[f"{e:,.0f}" for e in edges[:-1]]))

# ---------------- TAB 2: WHAT-IF ----------------
with tab2:
    mode = st.radio("What do you want to forecast?",
                    ["Price (MRP) sweep", "Compare outlets", "Compare item types"],
                    horizontal=True)

    if mode == "Price (MRP) sweep":
        mrps = np.linspace(df["Item_MRP"].min(), df["Item_MRP"].max(), 30)
        rows = [make_row(item_type, fat, weight, visibility, m, outlet_id) for m in mrps]
        out = pd.DataFrame({"Item_MRP": mrps, "Predicted_Sales": predict_many(rows)})
        st.line_chart(out, x="Item_MRP", y="Predicted_Sales")

    elif mode == "Compare outlets":
        rows = [make_row(item_type, fat, weight, visibility, mrp, oid)
                for oid in outlet_info.index]
        out = pd.DataFrame({"Outlet": outlet_info.index,
                            "Outlet_Type": outlet_info["Outlet_Type"].values,
                            "Predicted_Sales": predict_many(rows)})
        st.bar_chart(out, x="Outlet", y="Predicted_Sales")

    else:
        types = sorted(df["Item_Type"].unique())
        rows = [make_row(t, fat, weight, visibility, mrp, outlet_id) for t in types]
        out = pd.DataFrame({"Item_Type": types, "Predicted_Sales": predict_many(rows)})
        st.bar_chart(out, x="Item_Type", y="Predicted_Sales")

    st.dataframe(out.round(2), use_container_width=True)
    st.download_button("Download this table (CSV)", out.to_csv(index=False),
                       "forecast.csv", "text/csv")

# ---------------- TAB 3: PERFORMANCE ----------------
with tab3:
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Test R²", f"{metrics['R2']:.3f}")
    m2.metric("Test MAE", f"{metrics['MAE']:.0f}")
    m3.metric("Test RMSE", f"{metrics['RMSE']:.0f}")
    m4.metric("Train R²", f"{metrics['Train R2']:.3f}")

    left, right = st.columns(2)
    with left:
        st.subheader("Actual vs Predicted (test set)")
        st.scatter_chart(results, x="Actual", y="Predicted")
    with right:
        st.subheader("Feature importance")
        st.bar_chart(importance)

# ---------------- TAB 4: DATA ----------------
with tab4:
    a, b = st.columns(2)
    with a:
        st.subheader("Average sales by outlet type")
        st.bar_chart(df.groupby("Outlet_Type")["Item_Outlet_Sales"].mean())
    with b:
        st.subheader("Average sales by item type")
        st.bar_chart(df.groupby("Item_Type")["Item_Outlet_Sales"].mean())
    st.subheader("Dataset preview")
    st.dataframe(df.head(100), use_container_width=True)
    st.caption(f"{len(df):,} rows · {df.shape[1]} columns")
