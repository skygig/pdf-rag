import sys
import logging
from pathlib import Path
from utils.getTextFromPDF import extract_text_from_pdf
from utils.textChunker import chunk_text
from utils.dbLoader import get_db_connection, load_document_from_file
from utils.local_similarity_search import (
    interactive_search,
    quick_search,
    check_embedding_status,
    display_search_results
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def print_extraction_results(pages: list[tuple[int, str]], preview_length: int = 300):
    """
    Prints the extraction results in a formatted way.

    Args:
        pages: List of tuples containing page number and text.
        preview_length: Number of characters to show in preview.
    """
    if not pages:
        print("🔴 No pages were extracted from the PDF.")
        return
    
    print("\n--- Extracted Content ---")
    for page_number, text in pages:
        print(f"\n--- Page {page_number} ---\n")
        # Show preview if text is very long
        if len(text) > preview_length * 10:  # Show full text only if reasonable size
            print(f"{text[:preview_length]}...")
            print(f"\n[Content truncated - Full page has {len(text)} characters]")
        else:
            print(text)
    
    print("\n--- End of Content ---")
    
    # Print summary statistics
    total_pages = len(pages)
    total_chars = sum(len(text) for _, text in pages)
    print(f"\nSummary: {total_pages} pages processed, {total_chars:,} characters extracted.")

def process_document(pdf_file_path: str):
    """Process a PDF document: extract, chunk, and store in database."""
    logger.info(f"Processing document: {pdf_file_path}")
    
    try:
        # Validate file exists and is readable
        pdf_path = Path(pdf_file_path)
        if not pdf_path.exists():
            print(f"🔴 Error: File '{pdf_file_path}' does not exist.")
            return False
        
        if not pdf_path.is_file():
            print(f"🔴 Error: '{pdf_file_path}' is not a file.")
            return False
        
        print(f"\n🔄 Processing document: {pdf_file_path}")
        
        # Extract text from PDF
        print("\n--- Starting Text Extraction ---")
        pages = extract_text_from_pdf(pdf_file_path)
        
        if not pages:
            print("🔴 Failed to extract text from PDF.")
            print("💡 Make sure the PDF contains readable text (not just images)")
            return False
        
        print_extraction_results(pages)

        # Create text chunks
        print("\n--- Starting Text Chunking ---")
        text_chunks = chunk_text(pages, chunk_size=400, chunk_overlap=50)

        if not text_chunks:
            print("🔴 No chunks were created.")
            print("💡 This might indicate the PDF content is too short or has formatting issues")
            return False
        
        print(f"\n✅ Successfully created {len(text_chunks)} chunks.")
        
        # Show sample chunks
        sample_count = min(3, len(text_chunks))
        for i, chunk in enumerate(text_chunks[:sample_count]):
            print(f"\n--- Chunk {i+1} ---")
            print(f"Pages: {chunk.page_start}-{chunk.page_end}")
            print(f"Character count: {len(chunk.content)}")
            content_preview = chunk.content[:200]
            if len(chunk.content) > 200:
                content_preview += "..."
            print(f"Content preview: {content_preview}")
        
        if len(text_chunks) > sample_count:
            print(f"\n... and {len(text_chunks) - sample_count} more chunks")
        
        # Load to database with embeddings
        print("\n--- Connecting to Database ---")
        db_connection = get_db_connection()
        if not db_connection:
            print("🔴 Could not establish database connection.")
            print("💡 Check your database configuration and ensure the database is running")
            return False

        print("\n--- Starting Database Load with Embedding Generation ---")
        success = load_document_from_file(
            conn=db_connection,
            file_path=pdf_file_path,
            chunks=text_chunks,
            embedding_model="multi-qa-MiniLM-L6-cos-v1",
            batch_size=32
        )
        
        if success:
            print("\n✅ Database load and embedding generation completed successfully.")
        else:
            print("\n🔴 Database load failed.")
            print("💡 Check the database logs for more details")
        
        db_connection.close()
        return success
        
    except Exception as e:
        logger.error(f"Error processing document: {e}")
        print(f"\n🔴 Unexpected error processing document: {e}")
        return False

def search_mode():
    """Enter search mode to query the document database."""
    print("\n🔍 Entering Search Mode")
    
    try:
        # Get database connection
        db_connection = get_db_connection()
        if not db_connection:
            print("🔴 Could not establish database connection.")
            print("💡 Check your database configuration and ensure the database is running")
            return
        
        # Check embedding status first
        print("\n--- Checking Embedding Status ---")
        if not check_embedding_status(db_connection):
            print("⚠️ No embeddings found. Please process documents first.")
            print("💡 Use: python main.py <pdf_file> to process a document")
            db_connection.close()
            return
        
        # Start interactive search
        interactive_search(db_connection)
        
    except Exception as e:
        logger.error(f"Error in search mode: {e}")
        print(f"🔴 Error in search mode: {e}")
    finally:
        if 'db_connection' in locals() and db_connection:
            db_connection.close()

def quick_search_mode(query: str, limit: int = 5, threshold: float = 0.7):
    """Perform a quick search and display results."""
    print(f"\n🔍 Quick Search: '{query}'")
    
    try:
        # Validate parameters
        if not query.strip():
            print("🔴 Error: Empty search query provided.")
            return
        
        if limit <= 0:
            print("🔴 Error: Limit must be a positive number.")
            return
        
        if not (0.0 <= threshold <= 1.0):
            print("🔴 Error: Threshold must be between 0.0 and 1.0.")
            return
        
        # Get database connection
        db_connection = get_db_connection()
        if not db_connection:
            print("🔴 Could not establish database connection.")
            print("💡 Check your database configuration and ensure the database is running")
            return
        
        # Check if there are embeddings available
        if not check_embedding_status(db_connection):
            print("⚠️ No embeddings found. Please process documents first.")
            print("💡 Use: python main.py <pdf_file> to process a document")
            db_connection.close()
            return
        
        # Perform search
        results = quick_search(db_connection, query, limit, threshold)
        
        # Display results
        display_search_results(results, query)
        
    except Exception as e:
        logger.error(f"Error during quick search: {e}")
        print(f"🔴 Error during quick search: {e}")
    finally:
        if 'db_connection' in locals() and db_connection:
            db_connection.close()

def show_help():
    """Display help information."""
    print("\n📖 PDF Document Processing & Search System")
    print("=" * 50)
    print("\nUsage modes:")
    print("  python main.py <pdf_file>                    - Process a PDF document")
    print("  python main.py --search                      - Enter interactive search mode")
    print("  python main.py --quick-search <query>        - Perform quick search")
    print("  python main.py --status                      - Check database status")
    print("  python main.py --help                        - Show this help")
    print("\nExamples:")
    print("  python main.py document.pdf                  - Process document.pdf")
    print("  python main.py --search                      - Start interactive search")
    print('  python main.py --quick-search "solar system" - Search for "solar system"')
    print("  python main.py --status                      - Check embedding status")
    print("\nSearch Options:")
    print("  --limit <n>        - Number of results (default: 5)")
    print("  --threshold <f>    - Similarity threshold 0-1 (default: 0.7)")
    print("\nFull Example:")
    print('  python main.py --quick-search "planets" --limit 3 --threshold 0.8')
    print("\nNotes:")
    print("  - Supported PDF formats: Text-based PDFs (not scanned images)")
    print("  - First time processing may take longer due to model downloads")
    print("  - Database connection required for all operations")

def check_status():
    """Check the status of the database and embeddings."""
    print("\n📊 Checking Database Status")
    
    try:
        db_connection = get_db_connection()
        if not db_connection:
            print("🔴 Could not establish database connection.")
            print("💡 Check your database configuration and ensure the database is running")
            return
        
        check_embedding_status(db_connection)
        
    except Exception as e:
        logger.error(f"Error checking status: {e}")
        print(f"🔴 Error checking status: {e}")
    finally:
        if 'db_connection' in locals() and db_connection:
            db_connection.close()

def parse_arguments():
    """Parse command line arguments with improved validation."""
    args = sys.argv[1:]
    
    if not args or args[0] == "--help":
        show_help()
        return None
    
    # Check for different modes
    if args[0] == "--search":
        return {"mode": "interactive_search"}
    
    elif args[0] == "--status":
        return {"mode": "status"}
    
    elif args[0] == "--quick-search":
        if len(args) < 2:
            print("🔴 Error: --quick-search requires a query.")
            print("Usage: python main.py --quick-search <query>")
            return None
        
        # Parse query and optional parameters
        config = {
            "mode": "quick_search",
            "query": args[1],
            "limit": 5,
            "threshold": 0.7
        }
        
        # Parse optional parameters
        i = 2
        while i < len(args):
            if args[i] == "--limit" and i + 1 < len(args):
                try:
                    limit = int(args[i + 1])
                    if limit <= 0:
                        print(f"🔴 Invalid limit value: {args[i + 1]} (must be positive)")
                        return None
                    config["limit"] = limit
                    i += 2
                except ValueError:
                    print(f"🔴 Invalid limit value: {args[i + 1]} (must be an integer)")
                    return None
            elif args[i] == "--threshold" and i + 1 < len(args):
                try:
                    threshold = float(args[i + 1])
                    if not (0.0 <= threshold <= 1.0):
                        print(f"🔴 Invalid threshold value: {args[i + 1]} (must be between 0.0 and 1.0)")
                        return None
                    config["threshold"] = threshold
                    i += 2
                except ValueError:
                    print(f"🔴 Invalid threshold value: {args[i + 1]} (must be a number)")
                    return None
            else:
                print(f"🔴 Unknown parameter: {args[i]}")
                print("💡 Use --help to see available options")
                return None
        
        return config
    
    else:
        # Assume it's a PDF file path
        pdf_file_path = args[0]
        
        # Validate file exists and is a PDF
        file_path = Path(pdf_file_path)
        if not file_path.exists():
            print(f"🔴 Error: File '{pdf_file_path}' does not exist.")
            return None
        
        if not file_path.is_file():
            print(f"🔴 Error: '{pdf_file_path}' is not a file.")
            return None
        
        # Check file extension (optional warning)
        if not pdf_file_path.lower().endswith('.pdf'):
            print(f"⚠️ Warning: '{pdf_file_path}' does not have a .pdf extension")
            response = input("Continue anyway? (y/N): ").strip().lower()
            if response not in ['y', 'yes']:
                print("Operation cancelled.")
                return None
        
        return {"mode": "process", "file_path": pdf_file_path}

# This block allows the script to be run directly from the command line
if __name__ == "__main__":
    try:
        # Parse command line arguments
        config = parse_arguments()
        
        if config is None:
            sys.exit(1)
        
        # Execute based on mode
        if config["mode"] == "process":
            success = process_document(config["file_path"])
            if success:
                print(f"\n🎉 Document processing completed!")
                print("💡 You can now search the document using:")
                print("   python main.py --search")
                print("   python main.py --quick-search \"your query here\"")
            else:
                print("\n🔴 Document processing failed.")
                sys.exit(1)
        
        elif config["mode"] == "interactive_search":
            search_mode()
        
        elif config["mode"] == "quick_search":
            quick_search_mode(
                config["query"], 
                config["limit"], 
                config["threshold"]
            )
        
        elif config["mode"] == "status":
            check_status()
    
    except KeyboardInterrupt:
        print("\n\n👋 Operation cancelled by user.")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        print(f"\n🔴 Unexpected error: {e}")
        print("💡 Run with --help for usage information")
        sys.exit(1)