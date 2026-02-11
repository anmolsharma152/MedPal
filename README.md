<div align="center">

# 🏥 MedPal AI (V5)

**The First Neuro-Symbolic Clinical Decision Support System with Dual-RAG**

*Bridging the gap between Mathematical Precision and Medical Reasoning.*

[Report Bug](https://github.com/anmolsharma152/MedPal/issues) · [Request Feature](https://github.com/anmolsharma152/MedPal/issues)

</div>

---

## 🚨 The Problem: The "Black Box" of Medical AI

**Doctors do not trust "Black Box" AI.** If a model predicts a 90% risk of heart disease, a physician needs to know *why* and *if this has happened before*.

- ❌ Standard Chatbots **hallucinate** facts.
- ❌ Standard Neural Networks **cannot explain** their math or read clinical notes.

---

## 💡 The Solution: MedPal AI

**MedPal AI** is a **Neuro-Symbolic system** that combines the precision of Deep Learning with the reasoning of Large Language Models (LLMs), validated by real-world medical literature.

---

## 🔥 Key Features (The "Wow" Factors)

### 1. 🧠 Neuro-Symbolic Risk Engine

**MedPal doesn't just guess; it calculates.**

- **The "Neuro" (Math):** A specialized **Tabular ResNet** (PyTorch) analyzes raw extracted vitals (BP, Glucose, BMI) to output a precise **0-100% risk probability** for Diabetes and Heart Disease.
  
- **The "Symbolic" (Logic):** A logic layer **overrides** the neural network if critical medical history (e.g., "History of Heart Failure") is detected in the unstructured text, ensuring **100% safety compliance**.

### 2. 📚 Evidence-Based Medicine (Dual-RAG)

**Most RAG apps only talk to one document. MedPal has two brains:**

- **Local Brain:** Reads the specific patient's PDF report to answer questions about them.
  
- **Global Brain:** Cross-references the patient's symptoms against **167,000 real-world case studies** from the PMC-Patients dataset.

**Result:** MedPal validates its diagnosis by citing similar cases (e.g., *"This patient's profile matches Case #4201 regarding Metabolic Syndrome..."*).

### 3. ⚡ GenAI Clinical Assistant

- **Zero-Shot Extraction:** Zero-Shot Extraction: Uses Tesseract OCR for optical character recognition, followed by Llama 3 for schema-aligned clinical entity extraction.
  
- **Interactive Simulator:** Doctors can adjust sliders (*"What if the patient stops smoking?"*) to see risk scores drop in real-time.
  
- **Automated Workflow:** Generates professional Referral Letters and suggests ICD-10 Billing Codes instantly.

---

## 🛠️ Tech Stack

| Component | Technology | Description |
|-----------|------------|-------------|
| **LLM Engine** | Llama 3.3 70B | Powered by Groq LPUs for instant inference. |
| **Risk Models** | PyTorch | Custom Tabular ResNet trained on UCI & OpenML datasets. |
| **Vector DB** | ChromaDB | Dual-instance setup (Patient Memory + Global Medical Library). |
| **Embeddings** | HuggingFace | all-MiniLM-L6-v2 for semantic search. |
| **Frontend** | Streamlit | Professional 3-column clinical dashboard. |
| **Backend** | FastAPI | High-performance async microservice. |
| **OCR** | Tesseract | Optical Character Recognition for scanned reports. |

---

## 💻 Getting Started

### 1. Prerequisites

- Python 3.10+
- A free [Groq API Key](https://console.groq.com/)
- **System Dependencies (For OCR):**
  - **Arch Linux:** `sudo pacman -S tesseract tesseract-data-eng poppler`
  - **Ubuntu/Debian:** `sudo apt-get install tesseract-ocr poppler-utils`
  - **Mac:** `brew install tesseract poppler`

### 2. Installation
```bash
# Clone the repository
git clone https://github.com/yourusername/MedPal.git
cd MedPal

# Create Virtual Environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install Python Dependencies
pip install -r requirements.txt
```

### 3. Configuration

Create a `.env` file in the root directory:
```env
GROQ_API_KEY=gsk_your_key_here
```

### 4. Build the Medical Knowledge Base

This downloads a slice of the PMC-Patients dataset to power the "Evidence Engine."
```bash
# Downloads ~1,000 real-world case studies for the RAG system
python scripts/ingest_pmc.py
```

### 5. Run the System

You need to run the **Backend** and **Frontend** in separate terminals.

**Terminal 1: The Backend (Neural Engine)**
```bash
uvicorn server:app --reload --port 8000
```

**Terminal 2: The Frontend (Dashboard)**
```bash
streamlit run app.py
```

---

## 📸 Demo Workflow

1. **Upload Patient PDF:** The system runs OCR, extracts vitals, and updates memory.

2. **View Dashboard:**
   - **Left:** Control Panel.
   - **Middle:** Clinical Summary & Chat (RAG).
   - **Right:** Real-time Risk Gauges (Neural Net).

3. **Check Evidence:** Expand the "Evidence-Based Medicine" section to see matched cases from the literature.

4. **Simulate:** Switch to "Simulator Mode" to adjust vitals and see how risk changes.

---

## ⚖️ Disclaimer

**MedPal AI** is a proof-of-concept research tool designed for the GenAI Hackathon. It is **not a certified medical device** and should **not be used for actual clinical diagnosis or treatment**.

---

<div align="center">
  
<sub>Built with ❤️ by Anmol Sharma</sub>

</div>
