import streamlit as st
import pickle
import re
import numpy as np
import requests
import pandas as pd

# -------------------------
# PAGE CONFIG
# -------------------------
st.set_page_config(
    page_title="Education Sentiment Analysis Dashboard",
    page_icon="🎓",
    layout="wide"
)

# -------------------------
# LOAD MODELS
# -------------------------
svm_model = pickle.load(open("sentiment_model.pkl", "rb"))
nb_model = pickle.load(open("naive_bayes_model.pkl", "rb"))
xgboost_model = pickle.load(open("xgboost_model.pkl", "rb"))

# -------------------------
# TEXT CLEANING
# -------------------------
def clean_text(text):
    text = str(text)  # Ensures no int error
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z ]", "", text)
    return text

# -------------------------
# FETCH DATA FROM API
# -------------------------
def fetch_reviews_from_api():
    url = "28176f9d-d0a2-4b5c-ac94-9efc1ec43bd1"

    try:
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            return data[:10]
        else:
            return None
    except:
        return None

# -------------------------
# SIDEBAR
# -------------------------
st.sidebar.title("📊 Model Selection")

selected_model = st.sidebar.selectbox(
    "Choose Model",
    ("SVM", "Naive Bayes", "XGBoost")
)

st.sidebar.markdown("---")

if st.sidebar.button("📥 Fetch Reviews"):
    api_data = fetch_reviews_from_api()

    if api_data:
        st.session_state["api_reviews"] = api_data
        st.sidebar.success("Data fetched successfully!")
    else:
        st.sidebar.error("Failed to fetch data")

st.sidebar.markdown("---")
st.sidebar.info(
    "Our model analyzes students’ feedback and automatically classifies it into positive, negative, or neutral sentiments."
)

# -------------------------
# SELECT MODEL
# -------------------------
if selected_model == "SVM":
    model = svm_model
elif selected_model == "Naive Bayes":
    model = nb_model
else:
    model = xgboost_model

print("MODEL TYPE:", type(model))

# -------------------------
# MAIN TITLE
# -------------------------
st.title("🎓 Education System Sentiment Analysis")
st.markdown(f"Currently using **{selected_model} Model**")

# -------------------------
# MANUAL INPUT SECTION
# -------------------------
st.subheader("📝 Manual Sentiment Prediction")

user_input = st.text_area("Enter Review", height=150)

if st.button("Analyze Sentiment"):

    if user_input and user_input.strip() != "":

        cleaned = clean_text(user_input)

        # DEBUG
        st.write("DEBUG TYPE:", type(cleaned))
        st.write("DEBUG VALUE:", cleaned)

        try:
            prediction = model.predict([cleaned])[0]

            if prediction == 1:
                st.success("😊 Positive Sentiment")
            elif prediction == 0:
                st.error("😠 Negative Sentiment")
            else:
                st.warning("😐 Neutral Sentiment")

            if hasattr(model, "predict_proba"):
                prob = np.max(model.predict_proba([cleaned])) * 100
                st.metric("Confidence Score", f"{prob:.2f}%")

        except Exception as e:
            st.error(f"Error: {e}")

    else:
        st.warning("Please enter some text.")

# -------------------------
# API DATA ANALYSIS SECTION
# -------------------------
if "api_reviews" in st.session_state:

    st.markdown("---")
    st.subheader("📊 API Reviews Sentiment Analysis")

    results = []

    for review in st.session_state["api_reviews"]:
        text = review.get("body", "")
        cleaned = clean_text(text)

        try:
            prediction = model.predict([cleaned])[0]

            if prediction == 1:
                sentiment = "Positive"
            elif prediction == 0:
                sentiment = "Negative"
            else:
                sentiment = "Neutral"

            results.append({
                "Review": text,
                "Sentiment": sentiment
            })

        except:
            continue

    df = pd.DataFrame(results)
    st.dataframe(df)

    sentiment_counts = df["Sentiment"].value_counts()
    st.bar_chart(sentiment_counts)

# -------------------------
# FOOTER
# -------------------------
st.markdown("---")
st.caption("Educational Sentiment Analysis Project. Built with ❤️ by Dipika.")