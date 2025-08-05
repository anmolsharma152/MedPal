"""
OCR Processing for Medical Documents

This module provides functionality to extract text from scanned medical documents
using OCR (Optical Character Recognition).
"""
import os
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any
import logging
import tempfile

# Try to import OCR dependencies
try:
    import pytesseract
    from pdf2image import convert_from_path
    from PIL import Image, ImageEnhance, ImageFilter
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

logger = logging.getLogger(__name__)

class OCRProcessor:
    """Handles OCR processing for medical documents."""
    
    def __init__(self, dpi: int = 300, lang: str = 'eng'):
        """Initialize the OCR processor.
        
        Args:
            dpi: Dots per inch for image conversion
            lang: Language for OCR (default: 'eng' for English)
        """
        self.dpi = dpi
        self.lang = lang
        self.supported_formats = ['.pdf', '.png', '.jpg', '.jpeg', '.tiff', '.bmp']
        
        if not OCR_AVAILABLE:
            logger.warning(
                "OCR dependencies not available. Install with: "
                "pip install pytesseract pdf2image pillow"
            )
    
    def is_supported_file(self, file_path: str) -> bool:
        """Check if the file format is supported for OCR."""
        return Path(file_path).suffix.lower() in self.supported_formats
    
    def preprocess_image(self, image: Image.Image) -> Image.Image:
        """Preprocess image to improve OCR accuracy."""
        # Convert to grayscale
        image = image.convert('L')
        
        # Enhance contrast
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(2)
        
        # Apply slight blur to reduce noise
        image = image.filter(ImageFilter.MedianFilter())
        
        # Sharpen the image
        enhancer = ImageEnhance.Sharpness(image)
        image = enhancer.enhance(2)
        
        return image
    
    def process_pdf(self, pdf_path: str) -> Tuple[str, List[Dict[str, Any]]]:
        """Process a PDF file with OCR.
        
        Returns:
            Tuple of (extracted_text, pages_metadata)
        """
        if not OCR_AVAILABLE:
            raise RuntimeError("OCR dependencies not installed")
        
        # Convert PDF to images
        pages = convert_from_path(pdf_path, dpi=self.dpi)
        
        full_text = []
        pages_metadata = []
        
        for i, page in enumerate(pages, 1):
            # Preprocess image
            processed_image = self.preprocess_image(page)
            
            # Save to temp file for OCR
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_img:
                temp_path = temp_img.name
                processed_image.save(temp_path, 'PNG')
            
            # Extract text using Tesseract
            text = pytesseract.image_to_string(
                temp_path,
                lang=self.lang,
                config='--psm 6'  # Assume a single uniform block of text
            )
            
            # Clean up temp file
            os.unlink(temp_path)
            
            # Store results
            full_text.append(text)
            pages_metadata.append({
                'page_number': i,
                'dimensions': f"{page.width}x{page.height}",
                'dpi': self.dpi,
                'word_count': len(text.split())
            })
        
        return '\n\n'.join(full_text), pages_metadata
    
    def process_image(self, image_path: str) -> Tuple[str, Dict[str, Any]]:
        """Process a single image file with OCR."""
        if not OCR_AVAILABLE:
            raise RuntimeError("OCR dependencies not installed")
        
        # Open and preprocess image
        with Image.open(image_path) as img:
            processed_image = self.preprocess_image(img)
            
            # Extract text using Tesseract
            text = pytesseract.image_to_string(
                processed_image,
                lang=self.lang,
                config='--psm 6'
            )
            
            metadata = {
                'dimensions': f"{img.width}x{img.height}",
                'format': img.format,
                'mode': img.mode,
                'dpi': self.dpi,
                'word_count': len(text.split())
            }
            
            return text, metadata
    
    def process_file(self, file_path: str) -> dict:
        """Process a file with appropriate OCR method based on file type."""
        if not OCR_AVAILABLE:
            raise RuntimeError("OCR dependencies not installed")
        
        file_path = str(file_path)  # Convert Path to string if needed
        file_ext = Path(file_path).suffix.lower()
        
        try:
            if file_ext == '.pdf':
                text, pages_metadata = self.process_pdf(file_path)
                return {
                    'text': text,
                    'file_type': 'pdf',
                    'page_count': len(pages_metadata),
                    'pages': pages_metadata,
                    'success': True
                }
            elif file_ext in ['.png', '.jpg', '.jpeg', '.tiff', '.bmp']:
                text, metadata = self.process_image(file_path)
                return {
                    'text': text,
                    'file_type': 'image',
                    'metadata': metadata,
                    'success': True
                }
            else:
                return {
                    'success': False,
                    'error': f'Unsupported file type: {file_ext}'
                }
                
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {str(e)}", exc_info=True)
            return {
                'success': False,
                'error': str(e)
            }

# Example usage
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python ocr_processor.py <path_to_file>")
        sys.exit(1)
    
    ocr = OCRProcessor()
    result = ocr.process_file(sys.argv[1])
    
    if result['success']:
        print("\nExtracted Text:")
        print("="*50)
        print(result['text'][:1000] + "..." if len(result['text']) > 1000 else result['text'])
        print("\nMetadata:")
        print("="*50)
        import pprint
        pprint.pprint({k: v for k, v in result.items() if k != 'text'})
    else:
        print(f"Error: {result.get('error', 'Unknown error')}")
