import streamlit as st
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt

# Title
st.title("📈 Air Quality Forecasting App")

# Sidebar options
st.sidebar.header("Settings")
forecast_period = st.sidebar.slider("Days to Forecast", min_value=7, max_value=90, value=30)

# File upload option
uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])

# Load data
if uploaded_file is not None:
    df = pd.read_csv('air_quality_clean.csv')
else:
    st.sidebar.info("Using default dataset: air_quality_clean.csv")
    df = pd.read_csv("air_quality_clean.csv")

# Show preview
st.write("### Data Preview")
st.dataframe(df.head())

# Ensure column names are correct
df = df.rename(columns={df.columns[0]: "ds", df.columns[1]: "y"})

# Train Prophet model
model = Prophet()
model.fit(df)

# Make future dataframe
future = model.make_future_dataframe(periods=forecast_period)
forecast = model.predict(future)

# Show forecast data
st.write("### Forecast Results")
st.dataframe(forecast.tail())

# Plot forecast
st.write("### Forecast Plot")
fig1 = model.plot(forecast)
st.pyplot(fig1)

# Plot components
st.write("### Forecast Components")
fig2 = model.plot_components(forecast)
st.pyplot(fig2)
