import time
from sentence_transformers import SentenceTransformer

# 1. Load the model (this will download ~80MB on first run)
print("⏳ Loading model... (this might take a moment)")
start_load = time.time()
model = SentenceTransformer('all-MiniLM-L6-v2')
load_time = time.time() - start_load
print(f"✅ Model loaded in {load_time:.2f} seconds.")

# 2. Define a sample "Medical Record" chunk (~300 words)
medical_text = """
Patient presented with severe shortness of breath and chest pain radiating to the left arm. 
History of hypertension and type 2 diabetes mellitus. 
ECG performed at 14:00 hours showed ST-segment elevation in leads V1-V4. 
Troponin I levels were elevated at 2.5 ng/mL. 
Diagnosis: Acute Anterior Myocardial Infarction. 
Treatment initiated: Aspirin 300mg, Clopidogrel 300mg, and Heparin infusion. 
Patient transferred to Cath Lab for immediate PCI. 
Post-procedure vitals: BP 130/85, HR 78, O2 Sat 98% on room air.
Discharge planned for follow-up in Cardiology clinic in 2 weeks.
Medications prescribed: Atorvastatin 40mg, Metoprolol 25mg bd, Ramipril 2.5mg od.
"""

# 3. Warmup run (compilation overhead)
print("\n🔥 Warming up CPU...")
model.encode("warmup sentence")

# 4. Benchmark loop
print("\n🏃 Starting Benchmark (processing 10 chunks)...")
times = []

for i in range(10):
    start = time.time()
    embedding = model.encode(medical_text)
    end = time.time()
    times.append(end - start)
    print(f"   Pass {i+1}: {end - start:.4f} seconds")

# 5. Results Analysis
avg_time = sum(times) / len(times)
print(f"\n📊 AVERAGE TIME: {avg_time:.4f} seconds per chunk")
print("-" * 40)

if avg_time < 0.2:
    print("🚀 RESULT: EXCELLENT. Your CPU handles this easily.")
elif avg_time < 1.0:
    print("⚠️ RESULT: ACCEPTABLE. It might be slightly slow for large PDFs.")
else:
    print("❌ RESULT: TOO SLOW. Your CPU is struggling.")
