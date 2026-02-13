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


Toxic (1) → If any toxicity category = 1

Non-Toxic (0) → Otherwise


## 🔍 Project Workflow

### 1️⃣ Data Exploration
- Dataset shape & structure analysis
- Missing value check
- Class imbalance analysis
- Sample toxic vs non-toxic inspection

### 2️⃣ Text Preprocessing
- Lowercasing
- Special character removal
- Stopword removal
- Tokenization
- Sequence padding

### 3️⃣ Model Development

Two architectures were implemented and compared:

#### 🔹 LSTM Model
- Embedding Layer
- LSTM Layer
- Dropout
- Dense (Sigmoid Output)

#### 🔹 CNN Model
- Embedding Layer
- Conv1D
- Global Max Pooling
- Dense (Sigmoid Output)

## 📊 Model Performance

| Model | Accuracy |
|--------|----------|
| LSTM | **96.13%** |
| CNN | 95.71% |

📌 LSTM achieved better contextual understanding and was selected for deployment.


## 💾 Model Saving

Final Model: 
 
      final_toxicity_lstm_model.keras

Tokenizer:

      final_tokenizer.pkl


     
### ▶️ How to Run the Project

1️⃣ Install dependencies:

      pip install -r requirements.txt

2️⃣ Run Streamlit App:

      streamlit run app.py


 ### 📁 Project Structure

          AI-Toxicity-Detection/
             │
             ├── app.py
             ├── final_toxicity_lstm_model.keras
             ├── final_tokenizer.pkl
             ├── test.csv
             ├── train.csv
             ├── requirements.txt
             └── README.md
