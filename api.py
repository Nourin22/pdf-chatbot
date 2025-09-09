import os
import shutil
import uuid
from fastapi import FastAPI, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import app  # import your RAG + summarization logic

load_dotenv()
api = FastAPI()

api.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

class ChatRequest(BaseModel):
    session_id: str
    question: str

@api.post("/upload")
async def upload_pdf(file: UploadFile):
    session_id = str(uuid.uuid4())
    persist_dir = f"./chroma_sessions/{session_id}"
    os.makedirs(persist_dir, exist_ok=True)

    pdf_path = os.path.join(persist_dir, file.filename)
    with open(pdf_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    app.load_or_build_index(pdf_path, persist_dir, force_rebuild=True)
    return {"session_id": session_id, "pdf_path": pdf_path}

@api.post("/chat")
async def chat(req: ChatRequest):
    persist_dir = f"./chroma_sessions/{req.session_id}"
    if not os.path.exists(persist_dir):
        return {"error": "Invalid session. Upload a PDF first."}

    pdf_files = [f for f in os.listdir(persist_dir) if f.endswith(".pdf")]
    if not pdf_files:
        return {"error": "No PDF found."}

    pdf_path = os.path.join(persist_dir, pdf_files[0])
    answer = app.answer_question(pdf_path, req.question, persist_dir=persist_dir)
    return {"answer": answer}

@api.get("/summarize_page")
async def summarize_page(session_id: str, page_number: int):
    persist_dir = f"./chroma_sessions/{session_id}"
    pdf_files = [f for f in os.listdir(persist_dir) if f.endswith(".pdf")]
    pdf_path = os.path.join(persist_dir, pdf_files[0])
    summary = app.summarize_page(pdf_path, page_number)
    return {"page": page_number, "summary": summary}

@api.get("/summarize_document")
async def summarize_document(session_id: str):
    persist_dir = f"./chroma_sessions/{session_id}"
    pdf_files = [f for f in os.listdir(persist_dir) if f.endswith(".pdf")]
    pdf_path = os.path.join(persist_dir, pdf_files[0])
    summary = app.summarize_document(pdf_path)
    return {"summary": summary}

@api.get("/keywords")
async def keywords(session_id: str):
    persist_dir = f"./chroma_sessions/{session_id}"
    pdf_files = [f for f in os.listdir(persist_dir) if f.endswith(".pdf")]
    pdf_path = os.path.join(persist_dir, pdf_files[0])
    kws = app.extract_keywords(pdf_path)
    return {"keywords": kws}

@api.get("/toc")
async def toc(session_id: str):
    persist_dir = f"./chroma_sessions/{session_id}"
    pdf_files = [f for f in os.listdir(persist_dir) if f.endswith(".pdf")]
    pdf_path = os.path.join(persist_dir, pdf_files[0])
    toc = app.generate_toc(pdf_path)
    return {"toc": toc}
