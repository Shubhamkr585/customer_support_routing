import streamlit as st
import pandas as pd
import pickle
import re, string
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer

# ---------------------- LOAD MODEL ---------------------- #
@st.cache_resource
def load_model():
    model = pickle.load(open("model/ticket_classifier.pkl", "rb"))
    tfidf = pickle.load(open("model/tfidf.pkl", "rb"))
    return model, tfidf

model, tfidf = load_model()

# ---------------------- CLEAN TEXT ---------------------- #
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|@\S+|#\S+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text.strip()

# ---------------------- PAGE CONFIG ---------------------- #
st.set_page_config(page_title="Customer Support Ticket Analyzer", page_icon="📊", layout="wide")
st.title("📊 Customer Support Ticket Routing Dashboard")
st.write("### Automatically classify and route support queries using NLP (Logistic Regression Model).")

# ---------------------- SIDEBAR ---------------------- #
st.sidebar.header("Navigation")
page = st.sidebar.radio("Go to:", ["Model Overview", "Evaluation", "Live Prediction"])

# ---------------------- PAGE 1: MODEL OVERVIEW ---------------------- #
if page == "Model Overview":
    st.subheader("📁 Dataset & Model Summary")

    df = pd.read_csv("../dataset/testing_output.csv")
    st.write("**Sample of test predictions (1000 rows)**")
    st.dataframe(df.head(10))

    st.info("✅ Model: Logistic Regression (TF-IDF)\n\n✅ Trained on 9000 samples, tested on 1000\n\n✅ Accuracy: ~88–90%")

# ---------------------- PAGE 2: EVALUATION ---------------------- #
elif page == "Evaluation":
    st.subheader("📈 Model Evaluation Metrics")

    try:
        cm = pd.read_csv("../dataset/confusion_matrix.png")  
    except:
        # Instead show pre-saved confusion matrix image
        st.image("../dataset/confusion_matrix.png", caption="Confusion Matrix", use_container_width=True)

    st.image("../dataset/class_scores.png", caption="Class-wise Precision, Recall, F1-Score", use_container_width=True)

    df_eval = pd.read_csv("../dataset/testing_output.csv")
    correct = (df_eval["True_Label"] == df_eval["Predicted_Label"]).sum()
    acc = round(correct / len(df_eval), 3)
    st.success(f"✅ Overall Model Accuracy on Test Data: {acc*100}%")

# ---------------------- PAGE 3: LIVE PREDICTION ---------------------- #
elif page == "Live Prediction":
    st.subheader("🤖 Try the Model Yourself")
    st.write("Type a customer support query below and see which department it belongs to!")

    text_input = st.text_area("Enter your query:", height=150, placeholder="e.g. My payment got deducted twice 😕")

    if st.button("Predict Department"):
        if not text_input.strip():
            st.warning("Please enter a message first!")
        else:
            clean = clean_text(text_input)
            vec = tfidf.transform([clean])
            pred = model.predict(vec)[0]

            colors = {"Billing": "#22c55e", "Technical": "#2563eb", "Account": "#eab308", "General": "#9333ea"}
            st.markdown(f"### 🎯 Predicted Department: <span style='color:{colors.get(pred, '#000')}'>{pred}</span>", unsafe_allow_html=True)

            st.balloons()
