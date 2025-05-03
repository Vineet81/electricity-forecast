# Streamlit Link:
[Click here:](https://electricity-forecast-bepu39vfve3gmfkwbhpsz8.streamlit.app/)
# 🔌 Electricity Forecasting Web App (Streamlit + Hugging Face)

This is a simple Streamlit web application that performs **electricity consumption forecasting** using a machine learning model hosted on **Hugging Face Hub**.

---

## 🚀 Features

- Loads pre-trained ML model (`model.pkl`) directly from Hugging Face
- Makes real-time predictions based on user inputs
- Built using **Streamlit**, **Pickle**, and **Requests**
- Clean interface for quick forecasting

---

## 📦 Model

The trained model is hosted publicly on Hugging Face:
[🔗 model.pkl on Hugging Face](https://huggingface.co/VineetSaini81/electricity-forecast-model/resolve/main/model.pkl)

It was saved using:

```python
with open("model.pkl", "wb") as f:
    pickle.dump(model_fit, f)
## To run the app on Terminal :
pip install -r requirements.txt
streamlit run app.py
