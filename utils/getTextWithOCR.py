import fitz  # PyMuPDF
import tempfile
import os
import ocrmypdf

def extract_text_with_ocr(pdf_path: str) -> list[tuple[int, str]]:
    try:
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_file:
            temp_pdf_path = temp_file.name

        print("Performing OCR... This may take a while for large documents.")
        ocrmypdf.ocr(
            input_file=pdf_path,
            output_file=temp_pdf_path,
            language='eng',
            force_ocr=True,
            skip_text=False,
            deskew=True,
            rotate_pages=True,
            remove_background=False,
            optimize=1,
            quiet=True
        )

        doc = fitz.open(temp_pdf_path)
        extracted_data = []
        for page_num, page in enumerate(doc):
            human_readable_page_num = page_num + 1
            page_text = page.get_text()
            extracted_data.append((human_readable_page_num, page_text))
        doc.close()

        return extracted_data

    except ocrmypdf.exceptions.MissingDependencyError as e:
        print(f"OCR failed due to missing dependency: {e}")
        return []
    except ocrmypdf.exceptions.SubprocessOutputError as e:
        print(f"OCR subprocess failed: {e}")
        return []
    except Exception as e:
        print(f"OCR extraction failed: {e}")
        return []
    finally:
        if 'temp_pdf_path' in locals() and os.path.exists(temp_pdf_path):
            try:
                os.unlink(temp_pdf_path)
            except:
                pass
