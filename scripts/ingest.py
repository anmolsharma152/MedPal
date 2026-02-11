import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma

# 1. Configuration
PDF_PATH = "medical_report.pdf"  # We will create this file next
DB_PATH = "./chroma_db"          # This folder will be created automatically

def ingest_document():
    print(f"📄 Loading {PDF_PATH}...")
    
    # Check if file exists
    if not os.path.exists(PDF_PATH):
        print(f"❌ Error: {PDF_PATH} not found. Please create it first.")
        return

    # 2. Load the PDF
    loader = PyPDFLoader(PDF_PATH)
    pages = loader.load()
    print(f"   - Found {len(pages)} pages.")

    # 3. Split into chunks (Critical for RAG)
    # 500 characters is a good balance for medical context
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", ".", "!", "?"]
    )
    chunks = text_splitter.split_documents(pages)
    print(f"   - Split into {len(chunks)} text chunks.")

    # 4. Initialize Embedding Model (The one we just benchmarked)
    print("🧠 Initializing Embeddings (all-MiniLM-L6-v2)...")
    embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

    # 5. Create/Update Vector Database
    print("💾 Saving to ChromaDB...")
    # This automatically: Embeds chunks -> Indexes them -> Saves to disk
    db = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_function,
        persist_directory=DB_PATH
    )
    
    print(f"✅ Success! Data saved to {DB_PATH}.")
    print("   Your vector database is ready for searching.")

if __name__ == "__main__":
    ingest_document()
