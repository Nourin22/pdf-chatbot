# ui.py
import requests
import streamlit as st
from streamlit_chat import message  # for nicer chat bubbles
import os

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="📚 PDF RAG Chatbot",
    page_icon="🤖",
    layout="wide"
)

# --------------- HEADER ----------------
st.title("📚 PDF RAG Chatbot")
st.markdown(
    """
    <div style="color: gray; font-size: 15px; margin-bottom: 20px;">
    Upload a PDF, ask questions, summarize content, and explore insights — powered by Groq LLM & Chroma RAG.
    </div>
    """,
    unsafe_allow_html=True,
)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "session_id" not in st.session_state:
    st.session_state.session_id = None

# --------------- SIDEBAR ----------------
with st.sidebar:
    st.header("⚙️ Settings")
    st.markdown("Manage your document sessions here.")

    if st.button("🔄 Reset Session", type="secondary"):
        st.session_state.session_id = None
        st.session_state.messages = []
        st.success("Session reset successfully!")

# --------------- UPLOAD PDF ----------------
st.header("📤 Upload Document")

uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file:
    files = {"file": (uploaded_file.name, uploaded_file, "application/pdf")}
    resp = requests.post(f"{API_URL}/upload", files=files)
    if resp.status_code == 200:
        data = resp.json()
        st.session_state.session_id = data["session_id"]
        st.success("✅ PDF uploaded and indexed successfully!")
    else:
        st.error("❌ Upload failed.")

# --------------- MAIN AREA ----------------
if st.session_state.session_id:
    tabs = st.tabs(["💬 Chat", "📑 Summaries & TOC", "🔑 Keywords"])

    # ---- Chat Tab ----
    with tabs[0]:
        st.subheader("Chat with your PDF")

        user_input = st.text_input("Ask a question:", placeholder="Type your question here...")
        if st.button("Send", type="primary") and user_input:
            payload = {"question": user_input, "session_id": st.session_state.session_id}
            resp = requests.post(f"{API_URL}/chat", json=payload)

            if resp.status_code == 200:
                answer = resp.json()["answer"]
                st.session_state.messages.append(("user", user_input))
                st.session_state.messages.append(("assistant", answer))
            else:
                st.error("Error: " + resp.text)

        # Display chat history
        for i, (role, content) in enumerate(st.session_state.messages):
            is_user = role == "user"
            message(content, is_user=is_user, key=f"msg_{i}")

    # ---- Summaries & TOC Tab ----
    with tabs[1]:
        st.subheader("Summaries & Table of Contents")

        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("📝 Summarize Whole Document"):
                resp = requests.get(
                    f"{API_URL}/summarize_document",
                    params={"session_id": st.session_state.session_id}
                )
                if resp.status_code == 200:
                    st.info(resp.json()["summary"])
                else:
                    st.error("Error: " + resp.text)

        with col2:
            page_number = st.number_input("Enter page number", min_value=1, value=1)
            if st.button("📄 Pagewise Summary"):
                resp = requests.get(
                    f"{API_URL}/summarize_page",
                    params={
                        "session_id": st.session_state.session_id,
                        "page_number": page_number
                    }
                )
                if resp.status_code == 200:
                    summary = resp.json()["summary"]
                    st.write(f"Page {page_number} Summary:\n{summary}")
                else:
                    st.error("Error: " + resp.text)

        with col3:
            if st.button("📑 Generate TOC"):
                resp = requests.get(
                    f"{API_URL}/toc",
                    params={"session_id": st.session_state.session_id}
                )
                if resp.status_code == 200:
                    st.success(resp.json()["toc"])
                else:
                    st.error("Error: " + resp.text)

    # ---- Keywords Tab ----
    with tabs[2]:
        st.subheader("Extracted Keywords")
        if st.button("🔑 Extract Keywords"):
            resp = requests.get(
                f"{API_URL}/keywords",
                params={"session_id": st.session_state.session_id}
            )
            if resp.status_code == 200:
                keywords = resp.json()["keywords"]
                st.write(", ".join(keywords.split(",")))
            else:
                st.error("Error: " + resp.text)
