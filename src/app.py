import streamlit as st
from pathlib import Path

# Import your pipeline
from main import run_pipeline   # if you're inside src/ when running
# or, if running from project root:
# from src.main import run_pipeline

# 1. Build the same relative paths you did in main.py
BASE_DIR = Path(__file__).resolve().parent.parent
KB_DIR   = BASE_DIR / "knowledgebase"
file_paths = [
    str(KB_DIR / "model_S_owners_manual.pdf"),
    str(KB_DIR / "model_X_owners_manual.pdf"),
]

st.set_page_config(page_title="Customer Care Chatbot", layout="wide")
st.title("🔎 Customer Care Chatbot")

# 2. User input
query = st.text_input(
    "Your question:",
    "What to do if parking brake fault sign is on?"
)

# 3. Trigger the pipeline
if st.button("Ask"):
    with st.spinner("Thinking…"):
        answer, chunks = run_pipeline(file_paths, query)

    # 4. Show results
    st.subheader("Answer")
    st.write(answer)

    st.subheader("Context")
    for c in chunks:
        text = getattr(c, "page_content", c)
        st.write(text)
