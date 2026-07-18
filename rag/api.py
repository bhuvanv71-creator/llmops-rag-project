from fastapi import FastAPI
from pydantic import BaseModel
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
import requests

app = FastAPI(title="RAG API")

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectordb = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)

class Query(BaseModel):
    question: str
    k: int = 2

@app.get("/")
def health():
    return {"status": "ok"}

@app.post("/ask")
def ask(query: Query):
    docs = vectordb.similarity_search(query.question, k=query.k)
    context = "\n\n".join([d.page_content for d in docs])

    prompt = f"""Answer the question using only the context below.

Context:
{context}

Question: {query.question}

Answer:"""

    response = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": "qwen2.5:0.5b", "prompt": prompt, "stream": False}
    )
    answer = response.json()["response"]

    return {
        "question": query.question,
        "answer": answer,
        "sources": [d.page_content for d in docs]
    }
