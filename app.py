import os
from typing import List
from dataclasses import dataclass
from dotenv import load_dotenv

# --- PDF & chunking ---
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- Embeddings & Vector DB ---
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma

# --- LLM client (Groq) ---
from groq import Groq

# -----------------------------
# Config
# -----------------------------
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 120
TOP_K = 4
CHROMA_DIR = "./chroma_pdf_index"
GROQ_MODEL = "llama-3.1-8b-instant"


@dataclass
class RetrievedContext:
    text: str
    source: str


SYSTEM_PROMPT = """You are a helpful assistant that answers strictly using the provided context.
If the answer is not in the context, say "I don't know from the provided document."
Be concise and cite short quotes when helpful.
"""


# -----------------------------
# Utility Functions
# -----------------------------
def _get_client():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is not set in environment.")
    return Groq(api_key=api_key)


def _safe_split_docs(pdf_path: str, chunk_size=1200, chunk_overlap=150):
    """Load a PDF and split into safe chunks."""
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, add_start_index=True
    )
    return splitter.split_documents(docs)


def summarize_text(text: str, instruction: str, max_tokens: int = 300) -> str:
    """Helper: summarize a given text with Groq."""
    client = _get_client()
    messages = [
        {"role": "system", "content": "You are a summarization assistant."},
        {"role": "user", "content": f"{instruction}\n\n{text}"},
    ]
    resp = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=messages,
        max_tokens=max_tokens,
        temperature=0.2,
    )
    return resp.choices[0].message.content.strip()


# -----------------------------
# Indexing & Retrieval
# -----------------------------
def load_or_build_index(pdf_path: str, persist_dir: str, force_rebuild: bool = False) -> Chroma:
    os.makedirs(persist_dir, exist_ok=True)
    embeddings = SentenceTransformerEmbeddings(model_name=EMBED_MODEL)

    has_existing = any(os.scandir(persist_dir))
    if has_existing and not force_rebuild:
        return Chroma(persist_directory=persist_dir, embedding_function=embeddings)

    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP, add_start_index=True
    )
    chunks = splitter.split_documents(docs)

    vectorstore = Chroma.from_documents(
        documents=chunks, embedding=embeddings, persist_directory=persist_dir
    )
    vectorstore.persist()
    return vectorstore


def retrieve_context(vectorstore: Chroma, query: str, k: int = TOP_K) -> List[RetrievedContext]:
    docs = vectorstore.similarity_search(query, k=k)
    out = []
    for d in docs:
        meta = d.metadata or {}
        src = meta.get("source", "PDF")
        out.append(RetrievedContext(text=d.page_content, source=src))
    return out


# -----------------------------
# Chat with Groq
# -----------------------------
def build_prompt(user_query: str, contexts: List[RetrievedContext]) -> List[dict]:
    context_text = "\n\n---\n\n".join([c.text for c in contexts])
    user_prompt = f"""Answer the question using ONLY the context below.

[CONTEXT]
{context_text}

[QUESTION]
{user_query}

Rules:
- If the context does not include the answer, say: "I don't know from the provided document."
- Keep answers precise. Quote short phrases from the context when appropriate.
"""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def chat_with_groq(messages: List[dict], temperature: float = 0.1, max_tokens: int = 800) -> str:
    client = _get_client()
    resp = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content.strip()


# -----------------------------
# Main Q&A Functions
# -----------------------------
def answer_question(pdf_path: str, question: str, persist_dir: str, force_rebuild: bool = False) -> str:
    vectorstore = load_or_build_index(pdf_path, persist_dir, force_rebuild=force_rebuild)
    contexts = retrieve_context(vectorstore, question, k=TOP_K)
    messages = build_prompt(question, contexts)
    return chat_with_groq(messages)


def summarize_page(pdf_path: str, page_number: int) -> str:
    loader = PyPDFLoader(pdf_path)
    pages = loader.load_and_split()
    if page_number > len(pages):
        return f"Page {page_number} is out of range. Total pages: {len(pages)}"
    page_text = pages[page_number - 1].page_content
    return summarize_text(page_text, instruction=f"Summarize page {page_number}", max_tokens=300)


def summarize_document(pdf_path: str) -> str:
    chunks = _safe_split_docs(pdf_path)
    if not chunks:
        return "Document is empty."

    grouped, buf, size = [], [], 0
    for c in chunks:
        size += len(c.page_content)
        buf.append(c.page_content)
        if size > 3000:
            grouped.append("\n".join(buf))
            buf, size = [], 0
    if buf:
        grouped.append("\n".join(buf))

    partials = [
        summarize_text(g, instruction=f"Summarize section {i+1}/{len(grouped)}", max_tokens=350)
        for i, g in enumerate(grouped)
    ]
    return summarize_text("\n".join(partials), instruction="Combine into final summary", max_tokens=600)


def extract_keywords(pdf_path: str) -> str:
    chunks = _safe_split_docs(pdf_path, chunk_size=1000)
    keywords = []
    for c in chunks:
        kws = summarize_text(
            c.page_content,
            instruction="Extract 5-7 important keywords (comma-separated):",
            max_tokens=60,
        )
        keywords.append(kws)
    return ", ".join(keywords)


def generate_toc(pdf_path: str) -> str:
    chunks = _safe_split_docs(pdf_path, chunk_size=1200)
    toc_parts = []
    for i, c in enumerate(chunks, 1):
        toc = summarize_text(
            c.page_content,
            instruction=f"Extract section headings/subheadings (chunk {i}/{len(chunks)}):",
            max_tokens=120,
        )
        toc_parts.append(toc)
    return summarize_text("\n".join(toc_parts), instruction="Merge into a clean Table of Contents:", max_tokens=400)
