
import sys
import os
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

# Use dynamic file path from CLI argument or default
pdf_path = sys.argv[1] if len(sys.argv) > 1 else 'Haidt, New Synthesis (1).pdf'


if not os.path.isfile(pdf_path):
    raise FileNotFoundError(f"PDF file not found at: {pdf_path}")

# Load the PDF file
loader = PyPDFLoader(pdf_path)
docs = loader.load()

# Split the documents into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
docs = text_splitter.split_documents(docs)

# Initialize the embedding function
embedding_function = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'}
)

# Create the Chroma vectorstore
vectorstore = Chroma.from_documents(docs, embedding_function, persist_directory='./chroma_db_nccn')

# Print the document count (safe version)
print("Document chunks stored:", len(vectorstore.get()['documents']))