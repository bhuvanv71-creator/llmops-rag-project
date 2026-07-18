from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectordb = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)

query = "What is Terraform used for?"
results = vectordb.similarity_search(query, k=2)

print(f"Query: {query}\n")
for i, doc in enumerate(results):
    print(f"--- Result {i+1} ---")
    print(doc.page_content)
    print()
