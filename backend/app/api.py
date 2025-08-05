"""
FastAPI application for the Medical Record Summarizer
"""
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict, Any
import os
import uuid
from .processor import MedicalRecordProcessor
from pydantic import BaseModel

app = FastAPI(
    title="AI Medical Record Summarizer API",
    description="API for processing and summarizing medical records",
    version="0.1.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the processor
processor = None

class SummaryResponse(BaseModel):
    summary: str
    entities: List[Dict[str, Any]]
    insights: List[Dict[str, Any]]
    medical_codes: Dict[str, List[str]]

@app.on_event("startup")
async def startup_event():
    """Initialize the processor on startup"""
    global processor
    try:
        processor = MedicalRecordProcessor()
    except Exception as e:
        print(f"Error initializing processor: {str(e)}")
        raise

@app.post("/summarize", response_model=SummaryResponse)
async def process_medical_record(file: UploadFile = File(...)):
    """
    Process a medical record file and return a summary
    """
    if not processor:
        raise HTTPException(status_code=503, detail="Processor not initialized")
    
    try:
        # Save the uploaded file temporarily
        file_extension = os.path.splitext(file.filename)[1].lower()
        temp_file = f"/tmp/{str(uuid.uuid4())}{file_extension}"
        
        with open(temp_file, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Process the file
        result = processor.process_document(temp_file)
        
        # Clean up
        os.remove(temp_file)
        
        return {
            "summary": result.get("summary", ""),
            "entities": result.get("entities", []),
            "insights": result.get("insights", []),
            "medical_codes": result.get("medical_codes", {})
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}
