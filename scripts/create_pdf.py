from fpdf import FPDF

pdf = FPDF()
pdf.add_page()
pdf.set_font("Arial", size=12)

text = """
Patient Record #99281
Date: Jan 12, 2026
Patient: John Doe (Male, 45)

Chief Complaint: Persistent dry cough and low-grade fever (38C) for 5 days.
History: Patient has a history of mild asthma. No known allergies.
Vitals: BP 120/80, HR 88, O2 Sat 96%.
Lab Results:
- WBC: 11,000 (Slightly elevated)
- CRP: 15 mg/L (Elevated)
- Chest X-Ray: Clear fields, no sign of pneumonia.

Diagnosis: Acute Viral Bronchitis.
Plan:
1. Rest and hydration.
2. Acetaminophen 500mg every 6 hours for fever.
3. Monitor breathing. Return if O2 drops below 94%.
"""

pdf.multi_cell(0, 10, text)
pdf.output("medical_report.pdf")
print("✅ medical_report.pdf created!")
