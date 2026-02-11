from fpdf import FPDF

class PDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 15)
        self.cell(0, 10, 'Memorial Hospital - Annual Checkup', 0, 1, 'C')

def create_healthy_report():
    pdf = PDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    
    text = """
    Patient Name: Jane Smith
    Date: Jan 20, 2026
    
    CLINICAL VITALS:
    --------------------------------------
    Age: 25 years
    Sex: Female
    BMI: 21.0 (Normal Range)
    
    LABORATORY RESULTS:
    --------------------------------------
    Blood Pressure: 110/70 mmHg (Optimal)
    Fasting Glucose: 85 mg/dL (Normal)
    Cholesterol: 150 mg/dL (Desirable)
    
    PHYSICIAN NOTES:
    Patient is in excellent health. No concerns.
    """
    
    pdf.multi_cell(0, 10, text)
    pdf.output("data/healthy_patient.pdf")
    print("✅ Created data/healthy_patient.pdf")

if __name__ == "__main__":
    create_healthy_report()
