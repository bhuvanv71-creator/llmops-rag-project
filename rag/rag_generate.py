from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
import requests

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectordb = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)

def rag_query(question, k=2):
    docs = vectordb.similarity_search(question, k=k)
    context = "\n\n".join([d.page_content for d in docs])

    prompt = f"""Answer the question using only the context below.

Context:
{context}

Question: {question}

Answer:"""

    response = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": "qwen2.5:0.5b", "prompt": prompt, "stream": False}
    )
    return response.json()["response"]

if __name__ == "__main__":
    question = "What is Terraform and how does it work?"
    print(f"Question: {question}\n")
    answer = rag_query(question)
    print(f"Answer: {answer}")
