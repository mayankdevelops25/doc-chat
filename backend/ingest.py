from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.embeddings import Embeddings
from dotenv import load_dotenv
import google.generativeai as genai
import os
import json

load_dotenv()

CHROMA_PATH = "chroma_db"
DOCS_INDEX = "uploaded_docs.json"

class GeminiEmbeddings(Embeddings):
    def embed_documents(self, texts):
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        result = genai.embed_content(
            model="models/gemini-embedding-001",
            content=texts
        )
        return result["embedding"] if isinstance(texts, str) else [r for r in result["embedding"]]

    def embed_query(self, text):
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        result = genai.embed_content(
            model="models/gemini-embedding-001",
            content=text
        )
        return result["embedding"]

def get_docs_index():
    if os.path.exists(DOCS_INDEX):
        with open(DOCS_INDEX, "r") as f:
            return json.load(f)
    return []

def save_docs_index(docs):
    with open(DOCS_INDEX, "w") as f:
        json.dump(docs, f)

def ingest_document(file_path: str):
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    filename = os.path.basename(file_path)
    collection_name = filename.replace(" ", "_").replace(".", "_")

    if file_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    else:
        from langchain_community.document_loaders import TextLoader
        loader = TextLoader(file_path)

    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(documents)

    embeddings = GeminiEmbeddings()

    Chroma.from_documents(
        chunks,
        embeddings,
        persist_directory=CHROMA_PATH,
        collection_name=collection_name
    )

    docs = get_docs_index()
    if not any(d["filename"] == filename for d in docs):
        docs.append({
            "filename": filename,
            "collection": collection_name,
            "chunks": len(chunks)
        })
        save_docs_index(docs)

    return f"Successfully ingested {len(chunks)} chunks from {filename}"