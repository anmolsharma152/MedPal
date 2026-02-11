import os
import shutil
import json
from huggingface_hub import hf_hub_download
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from dotenv import load_dotenv

load_dotenv()

# Separate DB for global medical knowledge
PMC_DB_PATH = "./chroma_pmc_db"

def build_knowledge_base():
    print("📥 Downloading raw dataset file (800MB) - this might take a moment...")
    
    # 1. Download the specific JSON file directly, bypassing the 'datasets' library strictness
    # This caches the file locally so subsequent runs are instant
    file_path = hf_hub_download(
        repo_id="zhengyun21/PMC-Patients",
        filename="PMC-Patients-V2.json",
        repo_type="dataset"
    )
    
    print(f"✅ File downloaded to: {file_path}")
    print("🔄 Loading JSON and extracting first 1,000 cases...")

    # 2. Open the file efficiently
    docs = []
    with open(file_path, "r", encoding="utf-8") as f:
        # We load the whole thing because standard JSON parsing is safer than streaming for broken schemas
        # If 800MB is too big for your RAM, we can switch to ijson, but this usually works on most laptops.
        data = json.load(f)
        
        # Take the first 1000
        subset = data[:1000]
        
        for item in subset:
            summary = item.get('patient', '')
            if not summary: continue
            
            # Use title or truncated summary
            title = item.get('title', f"Case Study: {summary[:50]}...")
            
            doc = Document(
                page_content=summary,
                metadata={"source": "PMC-Patients", "title": title}
            )
            docs.append(doc)

    print(f"📦 Embedding {len(docs)} real-world cases into Vector DB...")
    
    # 3. Reset DB
    if os.path.exists(PMC_DB_PATH):
        shutil.rmtree(PMC_DB_PATH)

    # 4. Save to Chroma
    embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    Chroma.from_documents(docs, embedding_function, persist_directory=PMC_DB_PATH)
    print("✅ Knowledge Base Built! MedPal can now cite precedents.")

if __name__ == "__main__":
    build_knowledge_base()