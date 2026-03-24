from langchain_community.vectorstores import Chroma
from langchain.embeddings.base import Embeddings
from dotenv import load_dotenv
import google.generativeai as genai
from groq import Groq
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

def query_document(question: str, collection_name: str, history: list = []):
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    embeddings = GeminiEmbeddings()

    db = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings,
        collection_name=collection_name
    )

    retriever = db.as_retriever(search_kwargs={"k": 3})
    docs = retriever.invoke(question)

    context = "\n\n".join([doc.page_content for doc in docs])

    system_prompt = f"""You are a helpful assistant that answers questions strictly based on the provided document context.
If the answer is not in the context, say "I couldn't find that in the document."

Document context:
{context}"""

    messages = [{"role": "system", "content": system_prompt}]
    for h in history[-6:]:
        messages.append({"role": h["role"], "content": h["content"]})
    messages.append({"role": "user", "content": question})

    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages,
        temperature=0.3
    )

    return response.choices[0].message.content