import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# --------------------------------------------------
# Page Configuration
# --------------------------------------------------
st.set_page_config(
    page_title="🌍 AI Language Detector",
    page_icon="🌍",
    layout="centered"
)

# --------------------------------------------------
# Load Model
# --------------------------------------------------
MODEL_NAME = "papluca/xlm-roberta-base-language-detection"

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    return tokenizer, model

tokenizer, model = load_model()

# --------------------------------------------------
# Sidebar
# --------------------------------------------------
with st.sidebar:

    st.title("🌍 AI Language Detector")

    st.markdown("---")

    st.markdown("""
### 📌 About

This application detects the language of any text using the pretrained **XLM-RoBERTa** model from Hugging Face.

### 🛠 Technologies

- Python
- Streamlit
- PyTorch
- Transformers

### 👩‍💻 Developer

**Habiba Gamal**

### 🌐 Languages

Supports **100+ Languages**
""")

# --------------------------------------------------
# Main Page
# --------------------------------------------------

st.title("🌍 AI Language Detection")

st.write(
    """
Detect the language of any text using the powerful
**XLM-RoBERTa Language Detection Model**
from Hugging Face.
"""
)

# --------------------------------------------------
# Input
# --------------------------------------------------

text = st.text_area(
    "✍️ Enter Your Text",
    placeholder="Write any sentence here...",
    height=180
)

# --------------------------------------------------
# Text Statistics
# --------------------------------------------------

col1, col2 = st.columns(2)

with col1:
    st.metric("Words", len(text.split()))

with col2:
    st.metric("Characters", len(text))

# --------------------------------------------------
# Prediction
# --------------------------------------------------

if st.button("🚀 Detect Language", use_container_width=True):

    if text.strip() == "":
        st.warning("⚠ Please enter some text.")

    else:

        with st.spinner("Detecting language..."):

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
                results.append(
                    (id2label[i], float(prob))
                )

            results = sorted(
                results,
                key=lambda x: x[1],
                reverse=True
            )

        st.success("Prediction Completed Successfully ✅")

        best_language = results[0][0]
        best_score = results[0][1]

        st.subheader("🌟 Detected Language")

        st.success(
            f"""
Language : **{best_language}**

Confidence : **{best_score:.2%}**
"""
        )

        st.subheader("📊 Top Predictions")

        for language, score in results[:5]:

            st.write(f"**{language}**")

            st.progress(score)

            st.write(f"{score:.2%}")

# --------------------------------------------------
# Examples
# --------------------------------------------------

st.markdown("---")

st.subheader("💡 Try These Examples")

examples = [
    "Hello, how are you?",
    "مرحبا كيف حالك؟",
    "Bonjour tout le monde",
    "Hola amigos",
    "Guten Morgen",
    "Ciao a tutti",
    "こんにちは",
]

for example in examples:
    st.code(example)

# --------------------------------------------------
# Footer
# --------------------------------------------------

st.markdown("---")

st.caption(
    "Made with ❤️ using Streamlit • Hugging Face • Transformers • PyTorch"
)
