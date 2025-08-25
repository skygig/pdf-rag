import fitz  # PyMuPDF
from .getTextWithOCR import extract_text_with_ocr 

def extract_text_from_pdf(pdf_path: str) -> list[tuple[int, str]]:
    """
    Extracts text from each page of a PDF file with OCR fallback.

    Args:
        pdf_path: The file path to the PDF document.
        use_ocr_fallback: Whether to use OCR if regular text extraction fails or yields minimal text.

    Returns:
        A list of tuples, where each tuple contains the page number
        (starting from 1) and the extracted text from that page.
        Returns an empty list if an error occurs.
    """
    try:
        # Open the provided PDF file
        doc = fitz.open(pdf_path)
    except FileNotFoundError:
        print(f"Error: The file '{pdf_path}' was not found.")
        return []
    except Exception as e:
        # Catch other potential errors from fitz, like corrupted files
        print(f"Error opening or reading PDF: {e}")
        return []

    # A list to hold the extracted text from each page
    extracted_data = []
    total_text_length = 0

    print(f"Processing '{pdf_path}'...")
    
    # First pass: try regular text extraction
    for page_num, page in enumerate(doc):
        # page_num is 0-indexed, so we add 1 for human-readable page numbers
        human_readable_page_num = page_num + 1
        
        # Extract text from the current page
        page_text = page.get_text()
        
        # Append the page number and its text to our list
        extracted_data.append((human_readable_page_num, page_text))
        total_text_length += len(page_text.strip())

    # Close the document to free up resources
    doc.close()
    
    # Use OCR if total extracted text is very short (likely scanned PDF) or if explicitly requested
    needs_ocr = (total_text_length < 100)
    
    if needs_ocr:
        print(f"Regular text extraction yielded minimal content ({total_text_length} characters). Attempting OCR...")
        ocr_data = extract_text_with_ocr(pdf_path)
        if ocr_data:
            print(f"OCR extraction successful. Using OCR results instead.")
            return ocr_data
        else:
            print("OCR extraction failed. Using original extraction results.")
    
    print(f"Successfully extracted text from {len(extracted_data)} pages.")
    return extracted_data
