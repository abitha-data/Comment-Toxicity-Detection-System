# 💬 AI Comment Toxicity Detection System

### 📌 Overview

This project builds a Deep Learning-based system to detect toxic comments in online platforms. The model classifies comments as Toxic or Non-Toxic and is deployed using Streamlit for real-time and bulk predictions.


## 🧠 Key Features

✔ Real-Time Toxicity Detection  
✔ Bulk CSV Moderation  
✔ Deep Learning Model Comparison (LSTM vs CNN)  
✔ Dataset Insights & Visualization  
✔ Glassmorphism UI with Dark/Light Mode  
✔ Streamlit Cloud Deployment Ready  

---

## 🛠 Tech Stack

| Category | Tools Used |
|-----------|------------|
| Programming | Python |
| NLP | Tokenization, Stopword Removal, Padding |
| Deep Learning | LSTM, CNN |
| Framework | TensorFlow / Keras |
| Deployment | Streamlit |
| Visualization | Matplotlib |
| Version Control | GitHub |

---

## 📂 Dataset

Dataset: **Jigsaw Toxic Comment Classification Dataset**

Original Labels:
- toxic  
- severe_toxic  
- obscene  
- threat  
- insult  
- identity_hate  

### Binary Target Creation

A unified binary label was created:


LSTM (Final Model) – 96.13% Accuracy

CNN – 95.71% Accuracy

LSTM performed better in contextual understanding and was selected for deployment.


