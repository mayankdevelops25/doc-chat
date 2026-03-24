from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os
from backend.ingest import ingest_document, get_docs_index
from backend.query import query_document

app = FastAPI()

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
def get_documents():
    return get_docs_index()

@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    os.makedirs("uploads", exist_ok=True)
    file_path = f"uploads/{file.filename}"
    with open(file_path, "wb") as f:
        f.write(await file.read())
    result = ingest_document(file_path)
    return {"message": result, "documents": get_docs_index()}

@app.post("/chat")
async def chat(data: dict):
    question = data.get("question", "")
    history = data.get("history", [])
    collection = data.get("collection", "")
    if not collection:
        return {"answer": "Please select a document first."}
    answer = query_document(question, collection, history)
    return {"answer": answer}