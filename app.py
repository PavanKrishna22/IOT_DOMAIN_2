import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import VotingRegressor, RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Load datasets
foot_traffic = pd.read_csv("grocery_store_datasets/realistic_foot_traffic_dataset.csv")
billing_info = pd.read_csv("grocery_store_datasets/updated_billing_info_dataset_realistic.csv")
product_info = pd.read_csv("grocery_store_datasets/product_info_dataset.csv")
sales_data = pd.read_csv("grocery_store_datasets/sales_dataset.csv")

# Preprocess dates
sales_data['Date'] = pd.to_datetime(sales_data['Date'])
foot_traffic['Date'] = pd.to_datetime(foot_traffic['Date'])
billing_info['Date'] = pd.to_datetime(billing_info['Date'])

# Page Title
st.title("🛒 Smart Grocery Store Sales Analysis & Forecasting Dashboard")
st.markdown("""
This dashboard visualizes 3-month sales and store activity data.
It includes product sales insights, foot traffic analysis, and store trends.
An ML model predicts future sales using hybrid regression techniques.
Optimize your operations using real-time sales intelligence.
""")

# -------- CARD 1: Product Sales Over Time --------
st.markdown("---")
st.markdown("### 📊 Product Sales Over Time")
selected_item = st.selectbox("Select a product:", product_info['Product_Name'])
item_sales = sales_data[sales_data['Product_Name'] == selected_item]
st.line_chart(item_sales.set_index('Date')['Units_Sold'])

# -------- CARD 2: Product Contribution Pie Chart --------
st.markdown("---")
st.markdown("### 🥧 Product Contribution to Total Sales")
product_sales = sales_data.groupby('Product_Name')['Total_Sales_Amount'].sum()
fig, ax = plt.subplots()
ax.pie(product_sales, labels=product_sales.index, autopct='%1.1f%%')
st.pyplot(fig)

# -------- CARD 3: Total Store Sales Trend --------
st.markdown("---")
st.markdown("### 📈 Total Store Sales Trend")
daily_sales = sales_data.groupby('Date')['Total_Sales_Amount'].sum()
st.line_chart(daily_sales)

# -------- CARD 4: Foot Traffic Pattern --------
st.markdown("---")
st.markdown("### 🚶 Foot Traffic Pattern (Average per Hour)")
foot_traffic['Hour'] = pd.to_datetime(foot_traffic['Timestamp'], format='%H:%M').dt.hour
avg_hourly_traffic = foot_traffic.groupby('Hour')['Foot_Count'].mean()
st.line_chart(avg_hourly_traffic)

# -------- CARD 5: Average Sales per Product --------
st.markdown("---")
st.markdown("### 🛍️ Average Sales per Product")
avg_sales_per_product = sales_data.groupby('Product_Name')['Units_Sold'].mean()
st.bar_chart(avg_sales_per_product)

# -------- CARD 6: Sales vs Foot Traffic --------
st.markdown("---")
st.markdown("### 🔁 Sales vs Foot Traffic Comparison")
daily_foot_traffic = foot_traffic.groupby('Date')['Foot_Count'].sum()
df_compare = pd.DataFrame({
    'Sales': daily_sales,
    'Foot Traffic': daily_foot_traffic
}).sort_index()
df_compare.fillna(method='ffill', inplace=True)
st.line_chart(df_compare)

# -------- CARD 7: ML Sales Prediction --------
st.markdown("---")
st.markdown("### 🔮 Predicting Sales for Next Month Using ML")

# ML preprocessing
billing_info['Transaction_Count'] = 1
daily_billing = billing_info.groupby('Date').agg({
    'Total_Amount': 'sum',
    'Transaction_Count': 'count'
}).reset_index()
daily_billing.columns = ['Date', 'Total_Daily_Bill_Amount', 'Total_Transactions']

daily_features = sales_data.groupby('Date')['Total_Sales_Amount'].sum().reset_index()
daily_features = daily_features.merge(daily_foot_traffic.reset_index(), on='Date', how='left')
daily_features = daily_features.merge(daily_billing, on='Date', how='left')
daily_features.fillna(method='ffill', inplace=True)

daily_features['Day'] = daily_features['Date'].dt.day
daily_features['Month'] = daily_features['Date'].dt.month
daily_features['Weekday'] = daily_features['Date'].dt.weekday

X = daily_features[['Day', 'Month', 'Weekday', 'Foot_Count', 'Total_Transactions']]
y = daily_features['Total_Sales_Amount']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

model = VotingRegressor([
    ('lr', LinearRegression()),
    ('rf', RandomForestRegressor()),
    ('gb', GradientBoostingRegressor())
])
model.fit(X_train, y_train)

# Predict next 30 days
last_avg_traffic = daily_features['Foot_Count'].iloc[-30:].mean()
last_avg_transactions = daily_features['Total_Transactions'].iloc[-30:].mean()
future_days = pd.date_range(start=daily_features['Date'].max() + pd.Timedelta(days=1), periods=30)
X_future = pd.DataFrame({
    'Day': future_days.day,
    'Month': future_days.month,
    'Weekday': future_days.weekday,
    'Foot_Count': [last_avg_traffic]*30,
    'Total_Transactions': [last_avg_transactions]*30
})
X_future_scaled = scaler.transform(X_future)
predicted_sales = model.predict(X_future_scaled)

# Show Results
total_pred_sales = predicted_sales.sum()
daily_avg_pred_sales = predicted_sales.mean()
st.success(f"✅ Predicted Total Sales for Next Month: ₹{int(total_pred_sales):,} | Daily Avg: ₹{int(daily_avg_pred_sales):,}")

# -------- CARD 8: Daily Predicted Sales Graph --------
st.markdown("### 📅 Predicted Daily Sales for Next Month")
predicted_df = pd.DataFrame({
    'Date': future_days,
    'Predicted_Sales': predicted_sales
})
st.line_chart(predicted_df.set_index('Date'))

csv = predicted_df.to_csv(index=False).encode('utf-8')
st.download_button("📄 Download Predicted Sales CSV", csv, "predicted_sales_next_month.csv", "text/csv")

# -------- CARD 9: Actual vs Predicted Sales --------
st.markdown("---")
st.markdown("### 📊 Actual vs Predicted Sales Comparison")
actual_sales = daily_sales[-90:].reset_index()
combo_df = pd.concat([
    pd.DataFrame({'Date': actual_sales['Date'], 'Sales': actual_sales['Total_Sales_Amount'], 'Type': 'Actual'}),
    pd.DataFrame({'Date': predicted_df['Date'], 'Sales': predicted_df['Predicted_Sales'], 'Type': 'Predicted'})
])
fig, ax = plt.subplots()
for label, df_ in combo_df.groupby('Type'):
    if label == 'Actual':
        ax.bar(df_['Date'], df_['Sales'], label='Actual Sales', alpha=0.7)
    else:
        ax.plot(df_['Date'], df_['Sales'], label='Predicted Sales', color='red', linewidth=2)
ax.set_title("Actual vs Predicted Sales")
ax.set_ylabel("Sales Amount")
ax.legend()
st.pyplot(fig)
