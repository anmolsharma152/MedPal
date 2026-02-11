import sys
import os
import shutil
import json
import time
import gc
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from groq import Groq

load_dotenv()

# Configuration
DB_PATH = "./chroma_db"          # Patient Memory
PMC_DB_PATH = "./chroma_pmc_db"  # External Medical Library
API_KEY = os.environ.get("GROQ_API_KEY")

# --- 1. MEMORY MANAGEMENT (Ingestion) ---
def ingest_document(text_content):
    """
    Refreshes the Vector Database for the current patient.
    Includes 'Retry Logic' to handle file locking errors.
    """
    # A. Clear existing Patient DB (Robustly)
    if os.path.exists(DB_PATH):
        # Force garbage collection to release any held file locks
        gc.collect()
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                shutil.rmtree(DB_PATH)
                break # Success!
            except PermissionError:
                # If locked, wait a split second and try again
                time.sleep(0.5)
                gc.collect()
            except Exception as e:
                print(f"⚠️ Warning during DB wipe: {e}")
                break
    
    # B. Create chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(text_content)
    docs = [Document(page_content=chunk) for chunk in chunks]
    
    # C. Save to Vector DB
    embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    Chroma.from_documents(docs, embedding_function, persist_directory=DB_PATH)
    
    print(f"✅ Memory Updated: Ingested {len(chunks)} chunks for new patient.")

# --- 2. RETRIEVAL (Dual-Brain Chat) ---
def get_rag_response(user_query):
    if not API_KEY: return "Error: GROQ_API_KEY not set."
    
    context_text = ""
    patient_context_for_search = ""
    embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # A. Search Patient Record (Local Memory)
    if os.path.exists(DB_PATH):
        try:
            db_patient = Chroma(persist_directory=DB_PATH, embedding_function=embedding_function)
            results_patient = db_patient.similarity_search(user_query, k=3)
            if results_patient:
                context_text += "--- CURRENT PATIENT RECORD ---\n"
                # Capture context for PMC search
                patient_context_for_search = results_patient[0].page_content
                context_text += "\n".join([doc.page_content for doc in results_patient])
                context_text += "\n\n"
            
            # Explicit cleanup
            db_patient = None
        except Exception as e:
            print(f"Patient DB Error: {e}")

    # B. Search Medical Library (Global Knowledge)
    if os.path.exists(PMC_DB_PATH):
        try:
            db_pmc = Chroma(persist_directory=PMC_DB_PATH, embedding_function=embedding_function)
            
            # Context-Aware Search
            search_query = patient_context_for_search if patient_context_for_search else user_query
            results_pmc = db_pmc.similarity_search(search_query, k=3)
            
            if results_pmc:
                context_text += "--- SIMILAR CASES FROM LITERATURE ---\n"
                context_text += "\n".join([f"Case Study: {doc.page_content[:500]}..." for doc in results_pmc])
                context_text += "\n\n"
            
            # Explicit cleanup
            db_pmc = None
        except Exception as e:
            print(f"PMC DB Error: {e}")

    if not context_text:
        return "⚠️ Memory is empty. Please upload a report first."

    # C. Generate Answer
    try:
        client = Groq(api_key=API_KEY)
        system_prompt = """
        You are MedPal, an advanced medical AI. 
        You have access to the current patient's file AND a database of similar past cases.
        
        1. Answer the user's question based primarily on the 'CURRENT PATIENT RECORD'.
        2. If relevant, mention insights from 'SIMILAR CASES FROM LITERATURE' to support your answer.
        3. If the answer is not in either, say you don't know.
        """
        
        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Context:\n{context_text}\n\nQuestion: {user_query}"}
            ],
            model="llama-3.3-70b-versatile",
            temperature=0.3
        )
        return response.choices[0].message.content

    except Exception as e:
        print(f"RAG Error: {e}")
        return "Sorry, I encountered an error analyzing the documents."

def extract_patient_data(full_text):
    client = Groq(api_key=API_KEY)
    
    # This prompt is strictly aligned with the CDC Diabetes Health Indicators dataset
    system_prompt = """
    You are a specialized Clinical Data Extractor. Extract 21 specific health indicators from the medical report.
    Return ONLY a single JSON object. 
    CRITICAL: Use -1 for any value that is NOT explicitly mentioned or cannot be reasonably inferred.

    Keys and Mapping Rules:
    - HighBP: 1 if patient has hypertension/high blood pressure, 0 if normal.
    - HighChol: 1 if high cholesterol/hyperlipidemia, 0 if normal.
    - CholCheck: 1 if cholesterol was checked in last 5 years, else 0.
    - BMI: numeric value (calculate from height/weight if available).
    - Smoker: 1 if smoked 100+ cigarettes in life / current smoker, else 0.
    - Stroke: 1 if history of stroke, else 0.
    - HeartDiseaseorAttack: 1 if CHD or Myocardial Infarction history, else 0.
    - PhysActivity: 1 if any physical activity/exercise mentioned in past 30 days, else 0.
    - Fruits: 1 if eats fruit 1+ times per day, else 0.
    - Veggies: 1 if eats vegetables 1+ times per day, else 0.
    - HvyAlcoholConsump: 1 if adult men >14 drinks/week or women >7 drinks/week, else 0.
    - AnyHealthcare: 1 if insurance or healthcare coverage mentioned, else 0.
    - NoDocbcCost: 1 if skipped doctor due to cost in last year, else 0.
    - GenHlth: 1 (Excellent), 2 (Very Good), 3 (Good), 4 (Fair), 5 (Poor).
    - MentHlth: days of poor mental health in past 30 days (0-30).
    - PhysHlth: days of physical illness/injury in past 30 days (0-30).
    - DiffWalk: 1 if serious difficulty walking or climbing stairs, else 0.
    - Sex: 1 for Male, 0 for Female.
    - Age: The patient's actual age in years.
    - Education: 1-6 scale (1: Never attended; 6: College graduate).
    - Income: 1-8 scale (1: <$10k; 8: >$75k).
    """
    
    try:
        # Increase text window to 40k characters to catch lifestyle and social history
        safe_text = full_text[:40000] 
        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Extract the indicators from this report:\n{safe_text}"}
            ],
            model="llama-3.3-70b-versatile",
            temperature=0,
            response_format={"type": "json_object"}
        )
        # Parse the JSON response from the LLM
        extracted_data = json.loads(response.choices[0].message.content)
        print(f"✅ Extracted 21 Features: {extracted_data}")
        return extracted_data
        
    except Exception as e:
        print(f"⚠️ Extraction Error: {e}")
        return {}

# --- 4. SUMMARIZATION (Unchanged) ---
def generate_clinical_summary(full_text):
    client = Groq(api_key=API_KEY)
    prompt = """
    Summarize this medical report for a doctor. Format with Markdown.
    Include: Patient ID, Chief Complaint, Diagnosis, Plan.
    """
    try:
        safe_text = full_text[:30000]
        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are an expert Medical Scribe."},
                {"role": "user", "content": f"{prompt}\n\nREPORT:\n{safe_text}"}
            ],
            model="llama-3.3-70b-versatile",
            temperature=0.3
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {e}"

# --- 5. NEW GENAI FUNCTIONS (Letter & Billing) ---
def generate_patient_letter(full_text):
    """
    GenAI: Translates medical jargon into a warm, easy-to-read letter for the patient.
    """
    client = Groq(api_key=API_KEY)
    prompt = """
    You are a compassionate medical assistant. Write a letter to this patient explaining their results.
    Rules:
    1. Use simple, non-medical language (5th grade reading level).
    2. Explain the diagnosis gently.
    3. Suggest 3 simple lifestyle changes (diet, exercise, sleep).
    4. Tone: Encouraging and warm.
    5. Format: Professional Letter.
    """
    try:
        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": f"Medical Report:\n{full_text[:25000]}"}
            ],
            model="llama-3.3-70b-versatile",
            temperature=0.7 
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error generating letter: {e}"

def suggest_billing_codes(full_text):
    """
    GenAI: Extracts ICD-10 Codes for billing.
    """
    client = Groq(api_key=API_KEY)
    prompt = """
    Analyze this report and suggest appropriate ICD-10-CM billing codes.
    Return ONLY a JSON object with a list of codes.
    Format: {"codes": [{"code": "I10", "description": "Essential Hypertension"}, ...]}
    """
    try:
        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": f"Report:\n{full_text[:25000]}"}
            ],
            model="llama-3.3-70b-versatile",
            temperature=0,
            response_format={"type": "json_object"}
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        return {"codes": []}