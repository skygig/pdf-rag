import sys 
from pathlib import Path
from utils.getTextFromPDF import extract_text_from_pdf
from utils.textChunker import chunk_text
from utils.dbLoader import get_db_connection, load_document_from_file

def print_extraction_results(pages: list[tuple[int, str]], preview_length: int = 300):
    """
    Prints the extraction results in a formatted way.

    Args:
        pages: List of tuples containing page number and text.
        preview_length: Number of characters to show in preview.
    """
    if pages:
        print("\n--- Extracted Content ---")
        for page_number, text in pages:
            print(f"\n--- Page {page_number} ---\n")
            print(text)
        print("\n--- End of Content ---")
        
        # Print summary statistics
        total_pages = len(pages)
        total_chars = sum(len(text) for _, text in pages)
        print(f"\nSummary: {total_pages} pages processed, {total_chars} characters extracted.")

# This block allows the script to be run directly from the command line
if __name__ == "__main__":
    # Check if a file path was provided as a command-line argument
    if len(sys.argv) < 2:
        print("Usage: python pdf_parser.py <path_to_pdf_file>")
        sys.exit(1)

    pdf_file_path = sys.argv[1]
    
    # Validate file exists
    if not Path(pdf_file_path).exists():
        print(f"Error: File '{pdf_file_path}' does not exist.")
        sys.exit(1)

    # Call our function to extract the text
    pages = extract_text_from_pdf(pdf_file_path)

    print_extraction_results(pages)

    print("--- Starting Text Chunking ---")
    text_chunks = chunk_text(pages, chunk_size=400, chunk_overlap=50)

    if text_chunks:
        print(f"\nSuccessfully created {len(text_chunks)} chunks.")
        for i, chunk in enumerate(text_chunks):
            print(f"\n--- Chunk {i+1} ---")
            print(chunk)
            print(f"Content: {chunk.content}")
    else:
        print("No chunks were created.")
    
    
    # Get a database connection
    db_connection = get_db_connection()

    if db_connection:
        print("\n--- Starting Database Load ---")
        success = load_document_from_file(
            conn=db_connection,
            file_path=pdf_file_path,
            chunks=text_chunks,
        )
        if success:
            print("\n✅ Database load completed successfully.")
        else:
            print("\n🔴 Database load failed.")
        # Close the connection
        db_connection.close()
    else:
        print("🔴 Could not establish database connection.")