import streamlit as st
import pandas as pd
import joblib
import pickle
import requests
import zipfile
import matplotlib.pyplot as plt

st.title("⚡ Electricity Forecasting App")
st.write("Predict global active power usage based on historical electrical readings.")

# Load dataset
@st.cache_data
def load_data():
    with zipfile.ZipFile("electricity.zip") as z:
        with z.open("electricity.csv") as f:
            df = pd.read_csv(f, parse_dates=['Datetime'], index_col='Datetime')
    return df
df = load_data()

st.subheader("📊 Sample of the Dataset")
st.dataframe(df.head())

# Load trained model
def load_model():
    url = "https://huggingface.co/VineetSaini81/electricity-forecast-model/resolve/main/model.pkl"
    response = requests.get(url)
    if response.status_code == 200:
        model = pickle.loads(response.content)
        return model
    else:
        st.error(f"Failed to fetch model. Status code: {response.status_code}")
        return None

model = load_model()
# Feature selection
features = ['Global_reactive_power', 'Voltage', 'Global_intensity', 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']

st.sidebar.header("🧮 Input Features")

def user_input():
    input_data = {}
    for feature in features:
        val = st.sidebar.slider(f"{feature}", float(df[feature].min()), float(df[feature].max()), float(df[feature].mean()))
        input_data[feature] = val
    return pd.DataFrame([input_data])

input_df = user_input()

st.subheader("🔍 Model Input Preview")
st.write(input_df)

# Prediction
# For example, predict next 10 time steps
steps = 10
prediction = model.forecast(steps=steps)


st.subheader("📈 Forecasted Global Active Power")
st.success(f"Predicted Value: {prediction[0]:.3f} kilowatts")

# Plot historical data
st.subheader("📉 Historical Global Active Power")
st.line_chart(df['Global_active_power'].dropna().resample("D").mean())
