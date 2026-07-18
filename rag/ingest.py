from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

print("Loading documents...")
loader = DirectoryLoader("./docs", glob="*.txt", loader_cls=TextLoader)
documents = loader.load()
print(f"Loaded {len(documents)} documents")

print("Splitting into chunks...")
splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
chunks = splitter.split_documents(documents)
print(f"Created {len(chunks)} chunks")

print("Loading embedding model (this downloads ~90MB first run)...")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

print("Storing in ChromaDB...")
vectordb = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db"
)
print("Done. Vector DB persisted to ./chroma_db")
