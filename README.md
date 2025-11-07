import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# -----------------------------------------
# PAGE CONFIG
# -----------------------------------------
st.set_page_config(page_title="Emotion Detection App", page_icon="😊", layout="centered")
st.title("😊 Emotion Detection from Text")
st.markdown("### Detect emotions (Happy, Sad, Angry, Fear, etc.) from your text using Machine Learning")

# -----------------------------------------
# LOAD DATASET
# -----------------------------------------
@st.cache_data
def load_data():
    # Load your local Kaggle dataset file
    df = pd.read_csv("train.txt", sep=";", header=None, names=["text", "emotion"])
    return df

df = load_data()

st.subheader("📊 Dataset Preview")
st.dataframe(df.head())

# -----------------------------------------
# TRAIN THE MODEL
# -----------------------------------------
st.info("Training model... please wait ⏳")

X_train, X_test, y_train, y_test = train_test_split(df["text"], df["emotion"], test_size=0.2, random_state=42)

vectorizer = TfidfVectorizer(max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

model = LogisticRegression(max_iter=200)
model.fit(X_train_vec, y_train)

# Evaluate accuracy
y_pred = model.predict(X_test_vec)
acc = accuracy_score(y_test, y_pred)
st.success(f"✅ Model trained successfully with **{acc*100:.2f}% accuracy**")

# -----------------------------------------
# PREDICT EMOTION FROM USER INPUT
# -----------------------------------------
st.subheader("🧠 Try it Yourself")

user_text = st.text_area("✍️ Enter a sentence to detect emotion:")

if st.button("🔍 Detect Emotion"):
    if user_text.strip() == "":
        st.warning("Please enter some text to analyze.")
    else:
        input_vec = vectorizer.transform([user_text])
        prediction = model.predict(input_vec)[0]

        emoji_map = {
            "happy": "😄",
            "sad": "😢",
            "angry": "😠",
            "fear": "😨",
            "surprise": "😮",
            "love": "❤️"
        }

        emoji = emoji_map.get(prediction.lower(), "🙂")
        st.success(f"**Predicted Emotion:** {prediction.upper()} {emoji}")

# -----------------------------------------
# MODEL PERFORMANCE
# -----------------------------------------
with st.expander("📈 Model Performance Report"):
    st.text(classification_report(y_test, y_pred))

# Emotion-Detection-from-Text
