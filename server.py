import time
import shutil
import os
import torch
import joblib
import numpy as np
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel

# Import local project modules
from utils.ocr import extract_text_from_scanned_pdf
from rag_chat import (
    extract_patient_data, 
    get_rag_response, 
    generate_clinical_summary, 
    ingest_document,
    generate_patient_letter, 
    suggest_billing_codes
)
from models.definitions import TabularResNet 

app = FastAPI(title="MedPal AI V5 (CDC-Enhanced)")

# --- 1. Load Neural Networks (Updated for CDC Dimensions) ---
print("🧠 Loading CDC-BRFSS Neural Models...")

def load_artifacts(name, input_dim):
    model = TabularResNet(input_dim)
    model.load_state_dict(torch.load(f"models/{name}_resnet.pth", map_location=torch.device('cpu')))
    model.eval()
    scaler = joblib.load(f"models/{name}_scaler.pkl")
    return model, scaler

# CDC Feature counts: Diabetes=21, Heart=20 (HeartDisease removed), Stroke=20 (Stroke removed)
d_model, d_scaler = load_artifacts("diabetes", 21)
h_model, h_scaler = load_artifacts("heart", 20)
s_model, s_scaler = load_artifacts("stroke", 20)
print("✅ 21-Feature Models Loaded Successfully!")

# --- 2. The 21-Feature Mapping Bridge ---

def get_cdc_feature_vector(vitals, target_type="diabetes"):
    """
    Maps LLM-extracted JSON to the exact order expected by CDC-trained models.
    Uses clinical neutral defaults for missing (-1) values.
    """
    # Define clinical 'Healthy' or 'Average' defaults for missing data
    feat_map = {
        "HighBP": 0 if vitals.get("HighBP", -1) == -1 else vitals.get("HighBP"),
        "HighChol": 0 if vitals.get("HighChol", -1) == -1 else vitals.get("HighChol"),
        "CholCheck": 1, # Default to yes for patients in clinical settings
        "BMI": 25.0 if vitals.get("BMI", -1) == -1 else vitals.get("BMI"),
        "Smoker": 0 if vitals.get("Smoker", -1) == -1 else vitals.get("Smoker"),
        "Stroke": 0 if vitals.get("Stroke", -1) == -1 else vitals.get("Stroke"),
        "HeartDiseaseorAttack": 0 if vitals.get("HeartDiseaseorAttack", -1) == -1 else vitals.get("HeartDiseaseorAttack"),
        "PhysActivity": 1 if vitals.get("PhysActivity", -1) == -1 else vitals.get("PhysActivity"),
        "Fruits": 1, "Veggies": 1, # Neutral lifestyle defaults
        "HvyAlcoholConsump": 0,
        "AnyHealthcare": 1, "NoDocbcCost": 0,
        "GenHlth": 3 if vitals.get("GenHlth", -1) == -1 else vitals.get("GenHlth"),
        "MentHlth": 0, "PhysHlth": 0,
        "DiffWalk": 0 if vitals.get("DiffWalk", -1) == -1 else vitals.get("DiffWalk"),
        "Sex": 0 if vitals.get("Sex", -1) == -1 else vitals.get("Sex"),
        "Age": 50 if vitals.get("Age", -1) == -1 else vitals.get("Age"),
        "Education": 5, "Income": 5
    }

    # Order must match training script exactly
    keys = ["HighBP", "HighChol", "CholCheck", "BMI", "Smoker", "Stroke", 
            "HeartDiseaseorAttack", "PhysActivity", "Fruits", "Veggies", 
            "HvyAlcoholConsump", "AnyHealthcare", "NoDocbcCost", "GenHlth", 
            "MentHlth", "PhysHlth", "DiffWalk", "Sex", "Age", "Education", "Income"]
    
    # Remove the target variable from features for specific models
    if target_type == "heart": keys.remove("HeartDiseaseorAttack")
    if target_type == "stroke": keys.remove("Stroke")
        
    return np.array([[feat_map[k] for k in keys]])

# --- 3. API Endpoints ---

@app.post("/analyze_full")
async def analyze_full_report(file: UploadFile = File(...)):
    start_time = time.time()
    temp_filename = f"temp_{file.filename}"
    
    try:
        with open(temp_filename, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        raw_text = extract_text_from_scanned_pdf(temp_filename)
        ingest_document(raw_text) # Update Patient RAG Memory
        
        # GenAI Insights
        summary = generate_clinical_summary(raw_text)
        letter = generate_patient_letter(raw_text)
        billing = suggest_billing_codes(raw_text)
        
        # 21-Feature Extraction
        vitals = extract_patient_data(raw_text)

        # Neural Inference
        with torch.no_grad():
            # Diabetes Risk
            d_feat = get_cdc_feature_vector(vitals, "diabetes")
            d_risk = d_model(torch.tensor(d_scaler.transform(d_feat), dtype=torch.float32)).item()
            
            # Heart Risk
            h_feat = get_cdc_feature_vector(vitals, "heart")
            h_risk = h_model(torch.tensor(h_scaler.transform(h_feat), dtype=torch.float32)).item()
            
            # Stroke Risk
            s_feat = get_cdc_feature_vector(vitals, "stroke")
            s_risk = s_model(torch.tensor(s_scaler.transform(s_feat), dtype=torch.float32)).item()

        # Symbolic Override for Pre-existing Conditions
        if vitals.get("HeartDiseaseorAttack") == 1: h_risk = 0.999
        if vitals.get("Stroke") == 1: s_risk = 0.999

        os.remove(temp_filename)
        
        return {
            "analysis_time": round(time.time() - start_time, 2),
            "generated_summary": summary,
            "patient_letter": letter,
            "billing_codes": billing,
            "extracted_vitals": vitals,
            "risk_assessment": {
                "diabetes_probability": f"{d_risk*100:.2f}%",
                "heart_disease_probability": f"{h_risk*100:.2f}%",
                "stroke_probability": f"{s_risk*100:.2f}%"
            }
        }
    except Exception as e:
        if os.path.exists(temp_filename): os.remove(temp_filename)
        return {"error": str(e)}

@app.post("/chat")
async def chat_endpoint(payload: dict):
    # Standard RAG Chat
    return {"ai_insight": get_rag_response(payload.get("question"))}