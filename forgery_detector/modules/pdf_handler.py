# modules/pdf_handler.py

import fitz  # PyMuPDF
import os
from PIL import Image

def pdf_to_images(pdf_path, output_folder='uploads/pdf_pages'):
    """
    Converts each page of a PDF to a separate image.
    
    Why? Our ELA and CNN work on images, not PDFs.
    So we convert PDF pages to images first.
    """
    
    os.makedirs(output_folder, exist_ok=True)
    image_paths = []
    
    # Open the PDF
    pdf_document = fitz.open(pdf_path)
    
    print(f"📄 PDF has {len(pdf_document)} page(s)")
    
    for page_number in range(len(pdf_document)):
        # Get the page
        page = pdf_document[page_number]
        
        # Render page to image at 200 DPI (higher = better quality)
        # mat = transformation matrix, 2.0 = 2x zoom = ~200 DPI
        mat = fitz.Matrix(2.0, 2.0)
        pix = page.get_pixmap(matrix=mat)
        
        # Save as PNG
        image_path = os.path.join(output_folder, f'page_{page_number + 1}.png')
        pix.save(image_path)
        image_paths.append(image_path)
        
        print(f"✅ Page {page_number + 1} saved to {image_path}")
    
    pdf_document.close()
    return image_paths