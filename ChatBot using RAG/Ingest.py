from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
import os

# Make sure docs folder exists
os.makedirs("docs", exist_ok=True)

# 1. Load docs
# 👉 If you have a text file
loader = TextLoader("Pakistan.txt")

# 👉 Or if you want to use PDF instead:
# loader = PyPDFLoader("Pakistan.pdf")

documents = loader.load()

# 2. Split docs into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = splitter.split_documents(documents)

# 3. Initialize embeddings
embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# 4. Store in Chroma
db = Chroma.from_documents(docs, embedding=embedder, persist_directory="chroma_db")
db.persist()

print("Docs Ingested and Saved in chroma_db/")
