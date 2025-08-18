from langchain.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings

# 1. Load docs
loader = PyPDFLoader("docs/myfile.pdf")   # or TextLoader("docs/myfile.txt")
documents = loader.load()

# 2. Split docs into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = splitter.split_documents(documents)

# 3. Initialize embeddings
embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# 4. Store in Chroma
db = Chroma.from_documents(docs, embedding=embedder, persist_directory="chroma_db")
db.persist()

print("Docs ingested and saved in chroma_db/")
