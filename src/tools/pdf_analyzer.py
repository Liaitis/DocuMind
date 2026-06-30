"""
PDF Analysis Tool using Gemini (new google-genai SDK).
Supports extracting, summarizing, and querying PDF content.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Dict, Optional

from google import genai
from pypdf import PdfReader

logger = logging.getLogger(__name__)

class PDFAnalyzerTool:
    """Tool for analyzing PDF documents using Gemini."""
    
    def __init__(self, api_key: str):
        """Initialize the PDF analyzer with new genai client."""
        self.client = genai.Client(api_key=api_key)
        
        # Select a model
        self.model_name = self._select_model()
        logger.info(f"PDF Analyzer using model: {self.model_name}")
        self.description = "Analyze PDF documents: extract text, summarize, answer questions"
    
    def _select_model(self) -> str:
        """Select the best available model."""
        candidates = [
            "gemini-2.0-flash",
            "gemini-flash-latest",
            "gemini-pro-latest",
            "gemini-2.0-flash-lite",
        ]
        for model in candidates:
            try:
                # Quick test
                self.client.models.generate_content(model=model, contents="Test")
                return model
            except Exception:
                continue
        return "gemini-2.0-flash"  # fallback
    
    def execute(self, query: str, files: Optional[List[Dict]] = None) -> str:
        """Execute PDF analysis."""
        if not files:
            return "Please upload a PDF file for analysis."
        
        pdf_files = [f for f in files if f.get('type') == 'application/pdf' or f.get('name', '').endswith('.pdf')]
        if not pdf_files:
            return "No PDF files found in the upload."
        
        results = []
        for pdf_file in pdf_files[:3]:
            try:
                result = self._analyze_single_pdf(query, pdf_file)
                results.append(result)
            except Exception as e:
                logger.error(f"PDF analysis error: {e}")
                results.append(f"Error analyzing {pdf_file['name']}: {str(e)}")
        
        return "\n\n".join(results)
    
    def _analyze_single_pdf(self, query: str, file: Dict) -> str:
        """Analyze a single PDF file."""
        if not Path(file['path']).exists():
            return f"File not found: {file['name']}"
        
        text = self._extract_pdf_text(file['path'])
        if not text:
            return f"Could not extract text from {file['name']}"
        
        if len(text) > 100000:
            text = text[:100000] + "\n...[truncated]..."
        
        prompt = f"""
You are a research analyst. Analyze this PDF document and answer the user's query.

PDF Document: {file['name']}

Document Content:
{text}

User Query: {query}

Provide a detailed analysis that:
1. Directly answers the user's question
2. Cites specific sections from the document
3. Summarizes key findings
4. Highlights important quotes or data
"""
        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt
            )
            return f"**Analysis of {file['name']}:**\n{response.text}"
        except Exception as e:
            logger.error(f"PDF analysis error: {e}")
            return f"Error analyzing PDF: {str(e)}"
    
    def _extract_pdf_text(self, file_path: str) -> str:
        """Extract text from a PDF file."""
        try:
            reader = PdfReader(file_path)
            text = ""
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
            return text
        except Exception as e:
            logger.error(f"PDF extraction error: {e}")
            return ""