from fastapi import FastAPI, UploadFile, File, Form, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os
from backend.ingest import ingest_document
from backend.query import query_document

app = FastAPI()

# In-memory session store: { session_id -> [{ filename, collection, chunks }] }
session_docs: dict = {}

@app.get("/health")
def health():
    return {"status": "ok"}

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="frontend"), name="static")

@app.get("/")
def root():
    return FileResponse("frontend/index.html")

@app.get("/documents")
def get_documents(session_id: str = Query(...)):
    return session_docs.get(session_id, [])

@app.post("/upload")
async def upload(file: UploadFile = File(...), session_id: str = Form(...)):
    os.makedirs("uploads", exist_ok=True)
    file_path = f"uploads/{file.filename}"
    with open(file_path, "wb") as f:
        f.write(await file.read())

    doc_info = ingest_document(file_path, session_id)

    if session_id not in session_docs:
        session_docs[session_id] = []

    # Avoid duplicates within the same session
    if not any(d["filename"] == doc_info["filename"] for d in session_docs[session_id]):
        session_docs[session_id].append(doc_info)

    return {"message": f"Successfully ingested {doc_info['filename']}", "documents": session_docs[session_id]}

@app.post("/chat")
async def chat(data: dict):
    question = data.get("question", "")
    history = data.get("history", [])
    collection = data.get("collection", "")
    if not collection:
        return {"answer": "Please select a document first."}
    answer = query_document(question, collection, history)
    return {"answer": answer}