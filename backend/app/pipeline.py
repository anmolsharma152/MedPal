"""
AI Medical Record Summarizer - Complete Model Pipeline
Hackathon Implementation with Langchain, RAG, Transformers, and Clinical NLP
"""

import os
import json
import spacy
import torch
import pandas as pd
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import logging

# Core ML/NLP Libraries
from transformers import (
    AutoTokenizer, AutoModel, AutoModelForSequenceClassification,
    pipeline, BertTokenizer, BertForSequenceClassification
)
from sentence_transformers import SentenceTransformer

# Langchain Components
from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.schema import Document

# Additional Libraries
import chromadb
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MedicalRecordProcessor:
    """Main pipeline for processing medical records"""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the medical record processor"""
        self.config = self._load_config(config_path)
        self.models = {}
        self.vectorstore = None
        self.setup_pipeline()
    
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration settings"""
        default_config = {
            "clinical_bert_model": "emilyalsentzer/Bio_ClinicalBERT",
            "embeddings_model": "dmis-lab/biobert-base-cased-v1.2",
            "summarization_model": "microsoft/BioGPT-Large",
            "chunk_size": 1000,
            "chunk_overlap": 200,
            "max_entities": 50,
            "confidence_threshold": 0.7
        }
        
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                user_config = json.load(f)
                default_config.update(user_config)
        
        return default_config
    
    def setup_pipeline(self):
        """Initialize all models and components"""
        logger.info("Setting up medical NLP pipeline...")
        
        # 1. Load spaCy model for medical NER
        try:
            self.nlp = spacy.load("en_core_sci_lg")
            logger.info("Loaded spaCy scientific model")
        except OSError:
            logger.warning("Scientific spaCy model not found. Install with: pip install scispacy")
            self.nlp = spacy.load("en_core_web_sm")
        
        # 2. Load Clinical BERT for classification
        self._setup_clinical_bert()
        
        # 3. Setup embeddings for RAG
        self._setup_embeddings()
        
        # 4. Initialize medical knowledge base
        self._setup_medical_knowledge_base()
        
        # 5. Setup summarization pipeline
        self._setup_summarization()
        
        logger.info("Pipeline setup complete!")
    
    def _setup_clinical_bert(self):
        """Setup Clinical BERT for medical text classification"""
        try:
            self.clinical_tokenizer = AutoTokenizer.from_pretrained(
                self.config["clinical_bert_model"]
            )
            self.clinical_model = AutoModel.from_pretrained(
                self.config["clinical_bert_model"]
            )
            
            # Setup urgency classifier (you'd train this on labeled data)
            self.urgency_classifier = pipeline(
                "text-classification",
                model="emilyalsentzer/Bio_ClinicalBERT",
                return_all_scores=True
            )
            
            logger.info("Clinical BERT models loaded successfully")
        except Exception as e:
            logger.error(f"Error loading Clinical BERT: {e}")
            self.clinical_model = None
    
    def _setup_embeddings(self):
        """Setup embeddings for medical knowledge retrieval"""
        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.config["embeddings_model"],
            model_kwargs={'device': 'cuda' if torch.cuda.is_available() else 'cpu'}
        )
    
    def _setup_medical_knowledge_base(self):
        """Create vector database with medical knowledge"""
        # This would contain ICD-10 codes, medical guidelines, etc.
        medical_knowledge = self._load_medical_ontologies()
        
        if medical_knowledge:
            self.vectorstore = Chroma.from_documents(
                documents=medical_knowledge,
                embedding=self.embeddings,
                persist_directory="./medical_knowledge_db"
            )
            logger.info("Medical knowledge base created")
    
    def _setup_summarization(self):
        """Setup medical text summarization"""
        try:
            self.summarizer = pipeline(
                "summarization",
                model="microsoft/BioGPT-Large",
                tokenizer="microsoft/BioGPT-Large",
                max_length=512,
                min_length=100
            )
            logger.info("Medical summarization model loaded")
        except Exception as e:
            logger.warning(f"BioGPT not available, using fallback: {e}")
            self.summarizer = pipeline(
                "summarization",
                model="facebook/bart-large-cnn"
            )
    
    def _load_medical_ontologies(self) -> List[Document]:
        """Load medical ontologies and create documents"""
        # Sample medical knowledge - in production, load from actual ontologies
        medical_knowledge = [
            # ICD-10 samples
            "E11.9 Type 2 diabetes mellitus without complications. Characterized by insulin resistance and relative insulin deficiency.",
            "I10 Essential hypertension. High blood pressure without known secondary cause.",
            "J44.1 Chronic obstructive pulmonary disease with acute exacerbation.",
            
            # Medication information
            "Metformin: First-line treatment for type 2 diabetes. Monitor for lactic acidosis.",
            "Lisinopril: ACE inhibitor for hypertension. Monitor kidney function and potassium levels.",
            "Albuterol: Short-acting beta agonist for acute bronchospasm relief.",
            
            # Clinical guidelines
            "Diabetes management requires HbA1c monitoring every 3-6 months.",
            "Hypertension diagnosis requires multiple elevated readings on separate occasions.",
            "COPD exacerbations require assessment of severity and appropriate bronchodilator therapy."
        ]
        
        documents = [Document(page_content=text) for text in medical_knowledge]
        return documents
    
    def process_document(self, file_path: str) -> Dict:
        """Main processing function for medical records"""
        logger.info(f"Processing medical record: {file_path}")
        
        # Step 1: Load and preprocess document
        raw_text = self._load_document(file_path)
        processed_chunks = self._preprocess_text(raw_text)
        
        # Step 2: Extract medical entities
        entities = self._extract_medical_entities(raw_text)
        
        # Step 3: Classify urgency and conditions
        classifications = self._classify_medical_content(raw_text)
        
        # Step 4: Generate insights using RAG
        insights = self._generate_rag_insights(raw_text, entities)
        
        # Step 5: Create comprehensive summary
        summary = self._generate_summary(raw_text, entities, insights)
        
        # Step 6: Map to medical codes
        medical_codes = self._map_to_medical_codes(entities)
        
        result = {
            "patient_id": self._extract_patient_id(raw_text),
            "processing_timestamp": datetime.now().isoformat(),
            "summary": summary,
            "entities": entities,
            "classifications": classifications,
            "insights": insights,
            "medical_codes": medical_codes,
            "confidence_scores": self._calculate_confidence_scores(entities, insights)
        }
        
        return result
    
    def _load_document(self, file_path: str) -> str:
        """Load document from various formats"""
        file_extension = os.path.splitext(file_path)[1].lower()
        
        if file_extension == '.pdf':
            loader = PyPDFLoader(file_path)
            pages = loader.load()
            return "\n".join([page.page_content for page in pages])
        
        elif file_extension in ['.txt', '.md']:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        
        else:
            raise ValueError(f"Unsupported file format: {file_extension}")
    
    def _preprocess_text(self, text: str) -> List[str]:
        """Preprocess medical text"""
        # Clean and normalize text
        text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
        text = re.sub(r'[^\w\s\.\,\;\:\!\?]', '', text)  # Remove special chars
        
        # Split into manageable chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config["chunk_size"],
            chunk_overlap=self.config["chunk_overlap"],
            separators=["\n\n", "\n", ". ", "! ", "? "]
        )
        
        chunks = text_splitter.split_text(text)
        return chunks
    
    def _extract_medical_entities(self, text: str) -> List[Dict]:
        """Extract medical entities using spaCy"""
        doc = self.nlp(text)
        entities = []
        
        for ent in doc.ents:
            entity_info = {
                "text": ent.text,
                "label": ent.label_,
                "start": ent.start_char,
                "end": ent.end_char,
                "confidence": getattr(ent, 'confidence', 0.8),
                "description": self._get_entity_description(ent.label_)
            }
            entities.append(entity_info)
        
        # Custom medical entity extraction
        entities.extend(self._extract_custom_medical_entities(text))
        
        return entities[:self.config["max_entities"]]
    
    def _extract_custom_medical_entities(self, text: str) -> List[Dict]:
        """Custom extraction for specific medical patterns"""
        entities = []
        
        # Extract vital signs
        vital_patterns = {
            "blood_pressure": r'(\d{2,3})/(\d{2,3})\s*mmHg',
            "heart_rate": r'(\d{2,3})\s*bpm',
            "temperature": r'(\d{2,3}\.?\d*)\s*°?[FC]',
            "weight": r'(\d{2,3}\.?\d*)\s*(kg|lbs?)'
        }
        
        for vital_type, pattern in vital_patterns.items():
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                entities.append({
                    "text": match.group(0),
                    "label": "VITAL_SIGN",
                    "subtype": vital_type,
                    "value": match.groups(),
                    "start": match.start(),
                    "end": match.end(),
                    "confidence": 0.9
                })
        
        return entities
    
    def _classify_medical_content(self, text: str) -> Dict:
        """Classify medical content for urgency and condition type"""
        classifications = {}
        
        # Urgency classification
        urgency_keywords = {
            "emergency": ["emergency", "urgent", "critical", "severe", "acute"],
            "routine": ["routine", "follow-up", "stable", "chronic"],
            "preventive": ["screening", "checkup", "vaccination", "prevention"]
        }
        
        urgency_scores = {}
        for category, keywords in urgency_keywords.items():
            score = sum(1 for keyword in keywords if keyword.lower() in text.lower())
            urgency_scores[category] = score
        
        classifications["urgency"] = max(urgency_scores, key=urgency_scores.get)
        classifications["urgency_confidence"] = max(urgency_scores.values()) / len(text.split()) * 100
        
        # Medical specialty classification
        specialty_keywords = {
            "cardiology": ["heart", "cardiac", "cardiovascular", "blood pressure"],
            "endocrinology": ["diabetes", "thyroid", "hormone", "insulin"],
            "pulmonology": ["lung", "respiratory", "breathing", "asthma"],
            "neurology": ["brain", "neurological", "seizure", "headache"],
            "general": ["general", "primary care", "family medicine"]
        }
        
        specialty_scores = {}
        for specialty, keywords in specialty_keywords.items():
            score = sum(1 for keyword in keywords if keyword.lower() in text.lower())
            specialty_scores[specialty] = score
        
        classifications["recommended_specialty"] = max(specialty_scores, key=specialty_scores.get)
        
        return classifications
    
    def _generate_rag_insights(self, text: str, entities: List[Dict]) -> List[Dict]:
        """Generate insights using RAG with medical knowledge"""
        insights = []
        
        if not self.vectorstore:
            return self._generate_rule_based_insights(entities)
        
        # Query medical knowledge base for relevant information
        entity_texts = [ent["text"] for ent in entities if ent["label"] in ["DISEASE", "SYMPTOM", "MEDICATION"]]
        
        for entity_text in entity_texts[:10]:  # Limit queries for demo
            try:
                # Retrieve relevant medical knowledge
                docs = self.vectorstore.similarity_search(entity_text, k=3)
                
                if docs:
                    insight = {
                        "entity": entity_text,
                        "type": "knowledge_retrieval",
                        "content": docs[0].page_content,
                        "confidence": 0.8,
                        "source": "medical_knowledge_base"
                    }
                    insights.append(insight)
            
            except Exception as e:
                logger.warning(f"RAG query failed for {entity_text}: {e}")
        
        return insights
    
    def _generate_rule_based_insights(self, entities: List[Dict]) -> List[Dict]:
        """Fallback rule-based insights when RAG is not available"""
        insights = []
        
        # Drug interaction checking
        medications = [ent for ent in entities if ent["label"] == "MEDICATION"]
        if len(medications) > 1:
            insights.append({
                "type": "drug_interaction_check",
                "content": f"Multiple medications detected: {[med['text'] for med in medications]}. Review for potential interactions.",
                "priority": "medium",
                "confidence": 0.7
            })
        
        # Vital signs analysis
        vitals = [ent for ent in entities if ent["label"] == "VITAL_SIGN"]
        for vital in vitals:
            if vital.get("subtype") == "blood_pressure":
                try:
                    systolic, diastolic = vital["value"]
                    if int(systolic) > 140 or int(diastolic) > 90:
                        insights.append({
                            "type": "vital_sign_alert",
                            "content": f"Elevated blood pressure detected: {vital['text']}",
                            "priority": "high",
                            "confidence": 0.9
                        })
                except:
                    pass
        
        return insights
    
    def _generate_summary(self, text: str, entities: List[Dict], insights: List[Dict]) -> Dict:
        """Generate comprehensive medical record summary"""
        
        # Create structured summary
        summary = {
            "executive_summary": "",
            "key_findings": [],
            "medications": [],
            "vital_signs": [],
            "recommendations": []
        }
        
        # Extract key information
        medications = [ent for ent in entities if ent["label"] == "MEDICATION"]
        symptoms = [ent for ent in entities if ent["label"] in ["SYMPTOM", "DISEASE"]]
        vitals = [ent for ent in entities if ent["label"] == "VITAL_SIGN"]
        
        # Generate executive summary using summarization model
        try:
            if len(text) > 100:
                summary_text = self.summarizer(
                    text[:1024],  # Limit input length
                    max_length=150,
                    min_length=50,
                    do_sample=False
                )[0]['summary_text']
                summary["executive_summary"] = summary_text
        except Exception as e:
            logger.warning(f"Summarization failed: {e}")
            summary["executive_summary"] = self._create_rule_based_summary(entities)
        
        # Populate structured fields
        summary["medications"] = [med["text"] for med in medications]
        summary["vital_signs"] = [vital["text"] for vital in vitals]
        summary["key_findings"] = [symptom["text"] for symptom in symptoms]
        
        # Generate recommendations from insights
        summary["recommendations"] = [
            insight["content"] for insight in insights 
            if insight.get("priority") in ["high", "medium"]
        ]
        
        return summary
    
    def _create_rule_based_summary(self, entities: List[Dict]) -> str:
        """Create summary using rule-based approach"""
        medications = [ent["text"] for ent in entities if ent["label"] == "MEDICATION"]
        symptoms = [ent["text"] for ent in entities if ent["label"] in ["SYMPTOM", "DISEASE"]]
        
        summary_parts = []
        
        if symptoms:
            summary_parts.append(f"Patient presents with: {', '.join(symptoms[:3])}")
        
        if medications:
            summary_parts.append(f"Current medications include: {', '.join(medications[:3])}")
        
        return ". ".join(summary_parts) + "."
    
    def _map_to_medical_codes(self, entities: List[Dict]) -> Dict:
        """Map extracted entities to medical codes (ICD-10, CPT)"""
        # Simple mapping - in production, use comprehensive medical coding APIs
        icd_mapping = {
            "diabetes": "E11.9",
            "hypertension": "I10",
            "copd": "J44.1",
            "asthma": "J45.9"
        }
        
        mapped_codes = {}
        
        for entity in entities:
            entity_text = entity["text"].lower()
            for condition, code in icd_mapping.items():
                if condition in entity_text:
                    mapped_codes[entity["text"]] = {
                        "icd_10": code,
                        "confidence": entity.get("confidence", 0.7)
                    }
        
        return mapped_codes
    
    def _calculate_confidence_scores(self, entities: List[Dict], insights: List[Dict]) -> Dict:
        """Calculate overall confidence scores for the analysis"""
        entity_confidences = [ent.get("confidence", 0.5) for ent in entities]
        insight_confidences = [ins.get("confidence", 0.5) for ins in insights]
        
        return {
            "overall_confidence": np.mean(entity_confidences + insight_confidences),
            "entity_extraction_confidence": np.mean(entity_confidences) if entity_confidences else 0.0,
            "insight_generation_confidence": np.mean(insight_confidences) if insight_confidences else 0.0,
            "total_entities_found": len(entities),
            "high_confidence_entities": len([e for e in entities if e.get("confidence", 0) > 0.8])
        }
    
    def _extract_patient_id(self, text: str) -> Optional[str]:
        """Extract patient ID from medical record"""
        # Look for common patient ID patterns
        patterns = [
            r'Patient ID[:\s]+([A-Z0-9]+)',
            r'MRN[:\s]+([A-Z0-9]+)',
            r'ID[:\s]+([A-Z0-9]{6,})'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1)
        
        return f"UNKNOWN_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    def _get_entity_description(self, label: str) -> str:
        """Get description for entity labels"""
        descriptions = {
            "PERSON": "Patient or healthcare provider name",
            "MEDICATION": "Prescribed or mentioned medication",
            "DISEASE": "Medical condition or diagnosis",
            "SYMPTOM": "Patient-reported symptom",
            "VITAL_SIGN": "Measured vital sign or lab value",
            "PROCEDURE": "Medical procedure or test",
            "ANATOMY": "Body part or anatomical reference"
        }
        return descriptions.get(label, "Medical entity")
    
    def batch_process(self, file_paths: List[str]) -> List[Dict]:
        """Process multiple medical records"""
        results = []
        
        for file_path in file_paths:
            try:
                result = self.process_document(file_path)
                results.append(result)
                logger.info(f"Successfully processed: {file_path}")
            except Exception as e:
                logger.error(f"Failed to process {file_path}: {e}")
                results.append({"error": str(e), "file": file_path})
        
        return results
    
    def generate_analytics_report(self, results: List[Dict]) -> Dict:
        """Generate analytics across multiple processed records"""
        if not results:
            return {}
        
        # Aggregate statistics
        total_entities = sum(len(r.get("entities", [])) for r in results)
        avg_confidence = np.mean([
            r.get("confidence_scores", {}).get("overall_confidence", 0) 
            for r in results if "confidence_scores" in r
        ])
        
        # Most common entities
        all_entities = []
        for result in results:
            all_entities.extend([ent["text"] for ent in result.get("entities", [])])
        
        entity_counts = pd.Series(all_entities).value_counts()
        
        analytics = {
            "total_records_processed": len(results),
            "total_entities_extracted": total_entities,
            "average_confidence": avg_confidence,
            "most_common_entities": entity_counts.head(10).to_dict(),
            "processing_timestamp": datetime.now().isoformat()
        }
        
        return analytics


# Demo and Testing Functions
def create_sample_medical_record():
    """Create a sample medical record for testing"""
    sample_record = """
    PATIENT MEDICAL RECORD
    
    Patient ID: MRN123456
    Date: 2024-08-01
    
    CHIEF COMPLAINT: 
    Patient presents with chest pain and shortness of breath for the past 2 days.
    
    VITAL SIGNS:
    Blood Pressure: 145/92 mmHg
    Heart Rate: 88 bpm
    Temperature: 98.6°F
    Weight: 185 lbs
    
    CURRENT MEDICATIONS:
    - Metformin 500mg twice daily
    - Lisinopril 10mg once daily
    - Albuterol inhaler as needed
    
    ASSESSMENT:
    Type 2 diabetes mellitus with hypertension. Possible cardiac evaluation needed.
    Continue current medications. Schedule cardiology consultation.
    
    PLAN:
    1. EKG and chest X-ray
    2. Follow up in 1 week
    3. Monitor blood pressure
    """
    
    return sample_record


def demo_pipeline():
    """Demonstrate the medical record processing pipeline"""
    print("🏥 AI Medical Record Summarizer Demo")
    print("=" * 50)
    
    # Initialize processor
    processor = MedicalRecordProcessor()
    
    # Create sample record
    sample_text = create_sample_medical_record()
    
    # Save sample to file for processing
    with open("sample_record.txt", "w") as f:
        f.write(sample_text)
    
    # Process the record
    print("Processing medical record...")
    result = processor.process_document("sample_record.txt")
    
    # Display results
    print("\n📋 PROCESSING RESULTS:")
    print(f"Patient ID: {result['patient_id']}")
    print(f"\n📝 Summary: {result['summary']['executive_summary']}")
    
    print(f"\n🔍 Entities Found ({len(result['entities'])}):")
    for ent in result['entities'][:10]:
        print(f"  • {ent['text']} ({ent['label']}) - Confidence: {ent.get('confidence', 'N/A'):.2f}")
    
    print(f"\n💡 Insights ({len(result['insights'])}):")
    for insight in result['insights']:
        print(f"  • {insight['content']}")
    
    print(f"\n🏷️ Medical Codes:")
    for entity, code_info in result['medical_codes'].items():
        print(f"  • {entity}: {code_info['icd_10']}")
    
    print(f"\n📊 Confidence Scores:")
    conf = result['confidence_scores']
    print(f"  • Overall: {conf['overall_confidence']:.2f}")
    print(f"  • Entity Extraction: {conf['entity_extraction_confidence']:.2f}")
    print(f"  • High Confidence Entities: {conf['high_confidence_entities']}/{conf['total_entities_found']}")
    
    # Clean up
    os.remove("sample_record.txt")
    
    return result


if __name__ == "__main__":
    # Run demo
    demo_result = demo_pipeline()
    
    print("\n🚀 Pipeline setup complete! Ready for hackathon demo.")
    print("\nNext steps:")
    print("1. pip install spacy scispacy transformers langchain chromadb")
    print("2. python -m spacy download en_core_web_sm")
    print("3. Customize medical knowledge base with real ontologies")
    print("4. Add web interface with Streamlit/Gradio")
