import streamlit as st
import pandas as pd
import numpy as np
import pickle
import time
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# -------------------------
# Page Config
# -------------------------
st.set_page_config(page_title="AI Toxicity Detection", layout="wide")

# -------------------------
# Custom Animated CSS
# -------------------------
st.markdown("""
<style>
@keyframes gradient {
  0% {background-position: 0% 50%;}
  50% {background-position: 100% 50%;}
  100% {background-position: 0% 50%;}
}

.header {
    font-size: 42px;
    font-weight: bold;
    text-align: center;
    padding: 20px;
    color: white;
    background: linear-gradient(-45deg, #1f1c2c, #928DAB, #232526, #414345);
    background-size: 400% 400%;
    animation: gradient 8s ease infinite;
    border-radius: 12px;
}

.fade-in {
    animation: fadeIn 1.5s ease-in;
}

@keyframes fadeIn {
    from {opacity: 0;}
    to {opacity: 1;}
}

.result-toxic {
    background: linear-gradient(90deg, #ff4e50, #f00000);
    padding: 20px;
    border-radius: 10px;
    color: white;
    font-size: 20px;
    text-align: center;
}

.result-clean {
    background: linear-gradient(90deg, #11998e, #38ef7d);
    padding: 20px;
    border-radius: 10px;
    color: white;
    font-size: 20px;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# -------------------------
# Load Model & Tokenizer
# -------------------------
model = load_model("final_toxicity_lstm_model.keras")

with open("final_tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

MAX_LEN = 150

def preprocess_text(text):
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=MAX_LEN, padding='post')
    return padded

# -------------------------
# Load Dataset
# -------------------------
df = pd.read_csv("train.csv")
df['label'] = df[
    ['toxic','severe_toxic','obscene','threat','insult','identity_hate']
].max(axis=1)

# -------------------------
# Animated Header
# -------------------------
st.markdown('<div class="header">💬 AI Comment Toxicity Detection</div>', unsafe_allow_html=True)
st.markdown("---")

# -------------------------
# Sidebar
# -------------------------
menu = st.sidebar.radio(
    "Navigation",
    ["Real-Time Prediction", "Data Insights", "Model Performance", "Bulk CSV Prediction"]
)

# ==============================
# REAL-TIME PREDICTION
# ==============================
if menu == "Real-Time Prediction":

    st.subheader("🔍 Analyze Comment")

    user_input = st.text_area("Enter your comment:", height=120)

    if st.button("🚀 Predict Now"):

        if user_input.strip() == "":
            st.warning("Please enter a comment.")
        else:
            with st.spinner("Analyzing..."):
                time.sleep(1)
                processed = preprocess_text(user_input)
                prediction = model.predict(processed)[0][0]

            st.progress(float(prediction))

            if prediction >= 0.5:
                st.markdown(f'<div class="result-toxic fade-in">⚠ Toxic Comment<br>Confidence: {prediction:.2f}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="result-clean fade-in">✅ Non-Toxic Comment<br>Confidence: {1-prediction:.2f}</div>', unsafe_allow_html=True)

# ==============================
# DATA INSIGHTS
# ==============================
elif menu == "Data Insights":

    st.subheader("📊 Dataset Insights")

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Comments", df.shape[0])
    col2.metric("Toxic Comments", int(df['label'].sum()))
    col3.metric("Non-Toxic Comments", int(df.shape[0] - df['label'].sum()))

    fig, ax = plt.subplots()
    df['label'].value_counts().plot(kind='bar', ax=ax)
    ax.set_xticklabels(["Non-Toxic", "Toxic"], rotation=0)
    ax.set_ylabel("Count")
    st.pyplot(fig)

# ==============================
# MODEL PERFORMANCE
# ==============================
elif menu == "Model Performance":

    st.subheader("📈 Model Comparison")

    lstm_accuracy = 0.9613
    cnn_accuracy = 0.9571

    st.metric("LSTM Accuracy", lstm_accuracy)
    st.metric("CNN Accuracy", cnn_accuracy)

    fig, ax = plt.subplots()
    ax.bar(["LSTM", "CNN"], [lstm_accuracy, cnn_accuracy])
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy Comparison")
    st.pyplot(fig)

    st.success("🏆 LSTM Selected as Final Model")

# ==============================
# BULK CSV PREDICTION
# ==============================
elif menu == "Bulk CSV Prediction":

    st.subheader("📂 Bulk CSV Analysis")

    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

    if uploaded_file is not None:

        bulk_df = pd.read_csv(uploaded_file)

        if "comment_text" not in bulk_df.columns:
            st.error("CSV must contain 'comment_text' column.")
        else:
            sequences = tokenizer.texts_to_sequences(bulk_df["comment_text"])
            padded = pad_sequences(sequences, maxlen=MAX_LEN, padding='post')

            predictions = model.predict(padded)

            bulk_df["Toxic_Score"] = predictions
            bulk_df["Prediction"] = bulk_df["Toxic_Score"].apply(
                lambda x: "Toxic" if x >= 0.5 else "Non-Toxic"
            )

            st.dataframe(bulk_df)

            st.download_button(
                "⬇ Download Results",
                bulk_df.to_csv(index=False),
                file_name="bulk_predictions.csv",
                mime="text/csv"
            )
