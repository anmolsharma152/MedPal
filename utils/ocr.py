# utils/ocr.py
import pytesseract
from pdf2image import convert_from_path
import os

def extract_text_from_scanned_pdf(file_path: str) -> str:
    """
    Converts a PDF (scanned or digital) to images, then applies Tesseract OCR.
    """
    try:
        # 1. Convert PDF pages to images
        # poppler_path is usually in /usr/bin on Arch, so we don't need to specify it explicitly
        images = convert_from_path(file_path)
        
        full_text = ""
        print(f"👁️  OCR: Processing {len(images)} pages...")
        
        for i, img in enumerate(images):
            # 2. Extract text from image
            page_text = pytesseract.image_to_string(img)
            full_text += f"\n--- Page {i+1} ---\n{page_text}"
            
        return full_text
        
    except Exception as e:
        print(f"❌ OCR Error: {e}")
        return ""
