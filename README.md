📘 PDF RAG Chatbot

A Python-based Retrieval-Augmented Generation (RAG) chatbot that allows users to upload PDF documents and ask questions. The chatbot extracts, indexes, and retrieves relevant content from the document, then uses a Groq LLM (LLaMA 3) to generate precise, context-aware answers.



🚀 Features

📄 Upload PDFs: Users can upload their own documents.

🔍 RAG-powered Q&A: Ask questions and get answers strictly based on document content.

📑 Summarization: Summarize per page or the entire document.

📝 Keyword Extraction: Extract important keywords from the PDF.

📖 Table of Contents Generator: Auto-generate a TOC-style overview.

💬 Chat Interface: User-friendly Streamlit UI with persistent chat history.

🐳 Docker Deployment: Fully containerized (FastAPI + Streamlit + ChromaDB).

🛠️ Tech Stack

Python: 3.11

Backend: FastAPI

Frontend: Streamlit

Vector DB: ChromaDB

Embeddings: SentenceTransformers (all-MiniLM-L6-v2)

LLM: Groq (LLaMA 3.1 8B Instant)

Deployment: Docker & Docker Compose

📂 Project Structure
.
├── app.py                # Core RAG logic
├── api.py                # FastAPI backend
├── ui.py                 # Streamlit frontend
├── requirements.txt      # Dependencies
├── Dockerfile.api        # Backend container
├── Dockerfile.ui         # Frontend container
├── docker-compose.yml    # Compose file
├── .env.example          # API key example file
└── README.md             # Documentation

⚙️ Setup Instructions
1️⃣ Clone Repository
git clone https://github.com/<your-username>/pdf-rag-chatbot.git
cd pdf-rag-chatbot

2️⃣ Create Virtual Environment 
python -m venv .venv
.venv\Scripts\activate   

3️⃣ Install Dependencies
pip install -r requirements.txt

4️⃣ Set Environment Variables

Create a .env file:

GROQ_API_KEY=your_groq_api_key_here

▶️ Running the Application
Option 1: Local Run

Run API:

uvicorn api:api --reload


Run UI:

streamlit run ui.py


Visit: http://localhost:8501


Option 2: Docker Run

Make sure Docker & Docker Compose are installed.

docker compose up --build
Access the application:

API → http://localhost:8000

UI → http://localhost:8501

Make sure your .env file contains GROQ_API_KEY=your_groq_api_key_here.

📡 API Documentation
1. Upload PDF

Endpoint: POST /upload

Payload: Multipart (PDF file)

Response:

{
  "session_id": "123e4567-e89b-12d3-a456-426614174000",
  "message": "PDF uploaded and indexed."
}

2. Chat with PDF

Endpoint: POST /chat

Payload:

{
  "session_id": "<session_id>",
  "question": "What is the main topic of the document?"
}


Response:

{
  "answer": "The document discusses..."
}

