import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForMultipleChoice

# ---------------------------------------------------------------
# Change this to your Hugging Face Hub repo id from Step 2
# ---------------------------------------------------------------
REPO_ID = "Sandhya152004/Smart-MCQ-Solver-App"

MAX_LENGTH = 192
CHOICE_LABELS = ["A", "B", "C", "D", "E"]


@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(REPO_ID)
    model = AutoModelForMultipleChoice.from_pretrained(REPO_ID)
    model.eval()
    return model, tokenizer


def predict(model, tokenizer, prompt, options_dict):
    """options_dict: ordered dict-like {'A': '...', 'B': '...', ...}, 2-5 entries"""
    labels = list(options_dict.keys())
    candidates = list(options_dict.values())
    n = len(candidates)

    prompts = [prompt] * n
    enc = tokenizer(
        prompts,
        candidates,
        max_length=MAX_LENGTH,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

    # AutoModelForMultipleChoice expects (batch=1, num_choices, seq_len)
    input_ids = enc["input_ids"].unsqueeze(0)
    attention_mask = enc["attention_mask"].unsqueeze(0)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        probs = torch.softmax(outputs.logits, dim=1)[0].tolist()

    ranked = sorted(zip(labels, probs), key=lambda x: x[1], reverse=True)
    return ranked


# ---------------------------------------------------------------
# UI
# ---------------------------------------------------------------

st.set_page_config(page_title="Smart MCQ Solver (ELECTRA)", page_icon="⚡")

st.title("⚡ Smart MCQ Solver")
st.caption("Fine-tuned ELECTRA-base, hosted on Hugging Face Hub.")

with st.spinner("Loading model (first run may take a minute)..."):
    model, tokenizer = load_model()

st.subheader("Enter a question")
prompt = st.text_area("Question", placeholder="Type the question here...", height=100)

st.subheader("Enter the options")
col1, col2 = st.columns(2)
with col1:
    opt_a = st.text_input("A")
    opt_b = st.text_input("B")
    opt_c = st.text_input("C")
with col2:
    opt_d = st.text_input("D")
    opt_e = st.text_input("E")

if st.button("Predict answer", type="primary"):
    options = {"A": opt_a, "B": opt_b, "C": opt_c, "D": opt_d, "E": opt_e}
    options = {k: v for k, v in options.items() if v.strip() != ""}

    if not prompt.strip():
        st.warning("Please enter a question.")
    elif len(options) < 2:
        st.warning("Please enter at least two options.")
    else:
        ranked = predict(model, tokenizer, prompt, options)

        st.subheader("Prediction")
        top_choice, top_prob = ranked[0]
        st.success(f"**Top answer: {top_choice}** — {options[top_choice]}")

        st.write("Top 3 answers:")
        for choice, prob in ranked[:3]:
            st.write(f"**{choice}**: {options[choice]}")
            st.progress(min(prob, 1.0))
            st.caption(f"Confidence: {prob:.2%}")
