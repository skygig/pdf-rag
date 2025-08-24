
import fitz  # PyMuPDF
import sys   # Used to access command-line arguments

def extract_text_from_pdf(pdf_path: str) -> list[tuple[int, str]]:
    """
    Extracts text from each page of a PDF file.

    Args:
        pdf_path: The file path to the PDF document.

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

    print(f"Processing '{pdf_path}'...")
    # Iterate through each page of the document
    for page_num, page in enumerate(doc):
        # page_num is 0-indexed, so we add 1 for human-readable page numbers
        human_readable_page_num = page_num + 1
        
        # Extract text from the current page
        page_text = page.get_text()
        
        # Append the page number and its text to our list
        extracted_data.append((human_readable_page_num, page_text))

    # Close the document to free up resources
    doc.close()
    
    print(f"Successfully extracted text from {len(extracted_data)} pages.")
    return extracted_data

# This block allows the script to be run directly from the command line
if __name__ == "__main__":
    # Check if a file path was provided as a command-line argument
    if len(sys.argv) < 2:
        print("Usage: python pdf_parser.py <path_to_pdf_file>")
        sys.exit(1) # Exit the script with an error code

    # The first argument (sys.argv[0]) is the script name,
    # so the file path is the second argument (sys.argv[1])
    pdf_file_path = sys.argv[1]

    # Call our function to extract the text
    pages = extract_text_from_pdf(pdf_file_path)

    # If the extraction was successful, print the results
    if pages:
        print("\n--- Extracted Content ---")
        for page_number, text in pages:
            print(f"\n--- Page {page_number} ---\n")
            print(text)
        print("\n--- End of Content ---")

