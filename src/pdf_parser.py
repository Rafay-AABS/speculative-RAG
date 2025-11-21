import fitz  # PyMuPDF
from .strings import ERROR_PDF_READ, DOC_SEPARATOR


def parse_pdf(pdf_path):
    """
    Parse a single PDF file and extract all text content.
    
    Args:
        pdf_path (str): Path to the PDF file
        
    Returns:
        str: Extracted text from all pages
        
    Raises:
        Exception: If PDF cannot be read or parsed
    """
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
        return text
    except Exception as e:
        raise Exception(ERROR_PDF_READ.format(filename=pdf_path, error=str(e)))


def parse_pdfs(pdf_files):
    """
    Parse multiple PDF files and combine their text content.
    
    Args:
        pdf_files (list): List of PDF file paths
        
    Returns:
        str: Combined text from all PDFs, separated by DOC_SEPARATOR
    """
    pdf_texts = []
    for pdf_file in pdf_files:
        try:
            text = parse_pdf(pdf_file)
            if text.strip():  # Only add non-empty texts
                pdf_texts.append(text)
        except Exception as e:
            print(f"Warning: {str(e)}")
            continue
    
    return DOC_SEPARATOR.join(pdf_texts)
