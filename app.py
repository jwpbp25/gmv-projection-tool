#!/usr/bin/env python
# coding: utf-8

# In[1]:


# app_local.py
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

st.set_page_config(page_title="GMV Projection Tool", layout="wide")
st.title("🎯 GMV Projection Tool (Local)")

# -----------------------------
# 1. Load data (from CSV or define inline)
# -----------------------------
# Option 1: Define inline
data = {
    "gmv": [1471.0, 20958.0, 42193.48, 75385.7, 138078.64, 200184.11, 94626.29, 131042.48, 155955.37, 142379.55, 181240.14, 206835.39, 882517.39, 260180.28, 198149.75],
    "makers": [8,24,33,42,46,53,51,57,65,68,74,85,88,90,102],
    "active_skus": [19,130,229,378,431,571,440,495,611,610,660,879,1451,1042,986],
    "gmv_per_active_sku": [77.42,161.22,184.25,199.43,320.37,350.59,215.06,264.73,255.25,233.41,274.61,235.31,608.21,249.69,200.96],
    "sku_maker": [2,5,7,9,9,11,9,9,9,9,9,10,16,12,10],
    "orders": [21,154,275,650,928,1598,998,1441,1287,1248,1450,1735,4770,1957,1675],
    "units": [24,239,434,955,1385,2331,1413,1839,1679,1636,1877,2389,6555,2599,2224],
    "discounted_skus": [0,0,0,305,316,539,251,327,529,359,535,708,1356,668,440]
}

df = pd.DataFrame(data)

st.write("Sample data:")
st.dataframe(df.head())

# -----------------------------
# 2. Features and Target
# -----------------------------
features = ["makers", "active_skus", "orders", "units", "sku_maker", "discounted_skus"]
target = "gmv"

X = df[features]
y = df[target]

# -----------------------------
# 3. Train model
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
score = rf.score(X_test, y_test)
st.write(f"📈 Model R² on test set: {score:.2f}")

# -----------------------------
# 4. Streamlit UI Inputs
# -----------------------------
st.sidebar.header("Simulation Inputs")

target_gmv = st.sidebar.number_input("Target GMV ($)", value=int(df["gmv"].sum()*1.5), step=50000)
makers = st.sidebar.slider("Number of Makers", int(df["makers"].min()), int(df["makers"].max()), int(df["makers"].mean()))
active_skus = st.sidebar.slider("Active SKUs", int(df["active_skus"].min()), int(df["active_skus"].max()), int(df["active_skus"].mean()))
orders = st.sidebar.slider("Orders", int(df["orders"].min()), int(df["orders"].max()), int(df["orders"].mean()))
units = st.sidebar.slider("Units", int(df["units"].min()), int(df["units"].max()), int(df["units"].mean()))
sku_maker = st.sidebar.slider("SKU per Maker", int(df["sku_maker"].min()), int(df["sku_maker"].max()), int(df["sku_maker"].mean()))
discounted_skus = st.sidebar.slider("Discounted SKUs", int(df["discounted_skus"].min()), int(df["discounted_skus"].max()), int(df["discounted_skus"].mean()))

input_features = pd.DataFrame([{
    "makers": makers,
    "active_skus": active_skus,
    "orders": orders,
    "units": units,
    "sku_maker": sku_maker,
    "discounted_skus": discounted_skus
}])

# -----------------------------
# 5. Prediction
# -----------------------------
predicted_gmv = rf.predict(input_features)[0]

st.subheader("🔮 Projection Results")
st.metric("Predicted GMV", f"${predicted_gmv:,.0f}")
st.write(f"🎯 Target GMV: ${target_gmv:,.0f}")

if predicted_gmv >= target_gmv:
    st.success("✅ On track to hit the target!")
else:
    st.warning("⚠️ Below target. Adjust inputs or add more SKUs/makers.")

# -----------------------------
# 6. Feature Importance
# -----------------------------
st.subheader("📊 Feature Importance")
fi = pd.DataFrame({
    "feature": features,
    "importance": rf.feature_importances_
}).sort_values("importance", ascending=False)

st.bar_chart(fi.set_index("feature"))

