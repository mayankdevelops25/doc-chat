from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.embeddings import Embeddings
from dotenv import load_dotenv
import google.generativeai as genai
import os

load_dotenv()

CHROMA_PATH = "chroma_db"

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

def make_collection_name(session_id: str, filename: str) -> str:
    """
    Build a ChromaDB-safe collection name scoped to this session.
    ChromaDB limits: 3-63 chars, alphanumeric + underscores/hyphens/dots,
    must start and end with alphanumeric.
    """
    # Use first 12 hex chars of session_id (no hyphens) as prefix
    session_prefix = session_id.replace("-", "")[:12]
    # Sanitize filename: spaces and dots -> underscore, truncate to 40 chars
    name_part = filename.replace(" ", "_").replace(".", "_")[:40]
    # Ensure name_part ends with alphanumeric
    name_part = name_part.rstrip("_")
    return f"s{session_prefix}_{name_part}"

def ingest_document(file_path: str, session_id: str) -> dict:
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    filename = os.path.basename(file_path)
    collection_name = make_collection_name(session_id, filename)

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

    return {
        "filename": filename,
        "collection": collection_name,
        "chunks": len(chunks)
    }