"""
Document loading and chunking utilities.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

from pypdf import PdfReader

logger = logging.getLogger(__name__)

def load_document(file_path: str) -> str:
    """Load document text from file."""
    path = Path(file_path)
    if not path.exists():
        logger.error(f"File not found: {file_path}")
        return ""
    
    if path.suffix.lower() == '.pdf':
        return load_pdf(file_path)
    else:
        return load_text(file_path)

def load_pdf(file_path: str) -> str:
    """Extract text from PDF."""
    try:
        reader = PdfReader(file_path)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        return text.strip()
    except Exception as e:
        logger.error(f"PDF extraction error: {e}")
        return ""

def load_text(file_path: str) -> str:
    """Extract text from plain text file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except UnicodeDecodeError:
        with open(file_path, 'r', encoding='latin-1') as f:
            return f.read()
    except Exception as e:
        logger.error(f"Text loading error: {e}")
        return ""

def chunk_text(text: str, chunk_size: int = 800, overlap: int = 100) -> list[str]:
    """Split text into overlapping chunks."""
    if not text or len(text.strip()) < 20:
        return []
    
    # Split into sentences
    sentences = [s.strip() for s in text.replace('\n', ' ').split('. ') if s.strip()]
    
    chunks = []
    current = ""
    
    for sentence in sentences:
        if len(current) + len(sentence) < chunk_size:
            current += ". " + sentence if current else sentence
        else:
            if current:
                chunks.append(current)
                # Create overlap
                overlap_text = current[-overlap:] if overlap > 0 else ""
                current = overlap_text + ". " + sentence if overlap_text else sentence
    
    if current:
        chunks.append(current)
    
    # Clean up chunks
    chunks = [c.strip() for c in chunks if c.strip()]
    logger.info(f"Created {len(chunks)} chunks from {len(text)} characters")
    return chunks