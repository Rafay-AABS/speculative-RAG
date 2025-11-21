from typing import List
from pathlib import Path
import fitz  # PyMuPDF
from .strings import ERROR_PDF_READ, DOC_SEPARATOR
from .exceptions import PDFParsingError
from .logger import logger


def parse_pdf(pdf_path: str) -> str:
    """
    Parse a single PDF file and extract all text content.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        Extracted text from all pages
        
    Raises:
        PDFParsingError: If PDF cannot be read or parsed
    """
    try:
        logger.debug(f"Parsing PDF: {pdf_path}")
        doc = fitz.open(pdf_path)
        text = ""
        for page_num, page in enumerate(doc, 1):
            text += page.get_text()
        doc.close()
        logger.info(f"Successfully parsed PDF with {page_num} pages: {Path(pdf_path).name}")
        return text
    except Exception as e:
        error_msg = ERROR_PDF_READ.format(filename=pdf_path, error=str(e))
        logger.error(error_msg)
        raise PDFParsingError(error_msg) from e


def parse_pdfs(pdf_files: List[str]) -> str:
    """
    Parse multiple PDF files and combine their text content.
    
    Args:
        pdf_files: List of PDF file paths
        
    Returns:
        Combined text from all PDFs, separated by DOC_SEPARATOR
        
    Raises:
        PDFParsingError: If all PDFs fail to parse
    """
    logger.info(f"Parsing {len(pdf_files)} PDF files")
    pdf_texts = []
    failed_files = []
    
    for pdf_file in pdf_files:
        try:
            text = parse_pdf(pdf_file)
            if text.strip():  # Only add non-empty texts
                pdf_texts.append(text)
            else:
                logger.warning(f"PDF contains no text: {pdf_file}")
        except PDFParsingError as e:
            failed_files.append(pdf_file)
            logger.warning(f"Skipping failed PDF: {pdf_file}")
            continue
    
    if not pdf_texts:
        raise PDFParsingError(f"Failed to parse any PDFs. Failed files: {failed_files}")
    
    if failed_files:
        logger.warning(f"Failed to parse {len(failed_files)} out of {len(pdf_files)} PDFs")
    
    logger.info(f"Successfully parsed {len(pdf_texts)} PDFs")
    return DOC_SEPARATOR.join(pdf_texts)
