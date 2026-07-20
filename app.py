import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# -----------------------------
# Load Model
# -----------------------------
MODEL_NAME = "papluca/xlm-roberta-base-language-detection"

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    return tokenizer, model

tokenizer, model = load_model()

# -----------------------------
# Page
# -----------------------------
st.set_page_config(
    page_title="Language Detection",
    page_icon="🌍",
    layout="centered"
)

st.title("🌍 Language Detection using Transformers")

st.write(
    "Detect the language of any text using the pretrained "
    "**XLM-RoBERTa Language Detection Model**."
)

# -----------------------------
# Input
# -----------------------------
text = st.text_area(
    "✍️ Enter your text",
    height=150,
    placeholder="Write any sentence here..."
)

# -----------------------------
# Button
# -----------------------------
if st.button("Detect Language"):

    if text.strip() == "":
        st.warning("Please enter some text.")
    else:

        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True
        )

        with torch.no_grad():
            outputs = model(**inputs)

        probs = torch.softmax(outputs.logits, dim=1)[0]

        id2label = model.config.id2label

        results = []

        for i, prob in enumerate(probs):
            results.append((id2label[i], float(prob)))

        results = sorted(results, key=lambda x: x[1], reverse=True)

        st.success("Prediction Completed ✅")

        for language, score in results[:5]:
            st.write(f"**{language}** : {score:.2%}")

# -----------------------------
# Examples
# -----------------------------
st.markdown("---")
st.subheader("Examples")

st.code("Hello, how are you?")
st.code("مرحبا كيف حالك؟")
st.code("Bonjour tout le monde")
st.code("Hola amigos")
st.code("Guten Morgen")
