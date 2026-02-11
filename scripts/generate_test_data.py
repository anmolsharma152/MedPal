from fpdf import FPDF

class PDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 15)
        self.cell(0, 10, 'Memorial Hospital - Patient Diagnostic Report', 0, 1, 'C')

def create_dummy_report():
    pdf = PDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    
    text = """
    Patient Name: John Doe
    Date: Jan 20, 2026
    
    CLINICAL VITALS:
    --------------------------------------
    Age: 55 years
    Sex: Male
    BMI: 32.5 (Obese Class I)
    
    LABORATORY RESULTS:
    --------------------------------------
    Blood Pressure: 145/90 mmHg (Hypertension)
    Fasting Glucose: 160 mg/dL
    HbA1c: 7.2%
    Cholesterol: 240 mg/dL
    
    CARDIAC ASSESSMENT:
    --------------------------------------
    Resting ECG: ST-T wave abnormality
    Max Heart Rate: 155
    Exercise Induced Angina: Yes
    Chest Pain Type: Non-anginal pain
    
    PHYSICIAN NOTES:
    Patient complains of frequent thirst and fatigue. 
    Recommending further cardiology consultation due to elevated risk factors.
    """
    
    pdf.multi_cell(0, 10, text)
    pdf.output("test_patient.pdf")
    print("✅ Created test_patient.pdf")

if __name__ == "__main__":
    create_dummy_report()
