"""
Excel Analysis Tool using Gemini (new google-genai SDK).
Supports analyzing XLSX files with direct API calls for better accuracy on structured data.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import time
import random

import pandas as pd
from google import genai

logger = logging.getLogger(__name__)

class ExcelAnalyzerTool:
    """Tool for analyzing Excel files using Gemini with enhanced retry logic."""
    
    def __init__(self, api_key: str):
        """Initialize the Excel analyzer."""
        self.api_key = api_key
        self.client = genai.Client(api_key=api_key)
        self.model_name = self._select_model()
        self.max_retries = 5
        self.base_delay = 2
        logger.info(f"Excel Analyzer using model: {self.model_name}")
        self.description = "Analyze Excel files: extract data, generate insights, create summaries"
    
    def _select_model(self) -> str:
        """Select the best available model with fallback."""
        candidates = [
            "gemini-2.0-flash",
            "gemini-flash-latest",
            "gemini-1.5-flash",
            "gemini-pro-latest",
            "gemini-2.0-flash-lite",
        ]
        for model in candidates:
            try:
                self.client.models.generate_content(model=model, contents="Test")
                logger.info(f"Excel analyzer using: {model}")
                return model
            except Exception:
                continue
        logger.warning("Using fallback model: gemini-2.0-flash")
        return "gemini-2.0-flash"
    
    def _call_with_retry(self, prompt: str, max_tokens: int = 4096) -> str:
        """Call API with retry logic for rate limits."""
        retry_count = 0
        total_wait = 0
        
        while retry_count < self.max_retries:
            try:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config={
                        "temperature": 0.2,
                        "max_output_tokens": max_tokens,
                    }
                )
                if response and response.text:
                    return response.text
                else:
                    raise Exception("Empty response from model")
                    
            except Exception as e:
                error_str = str(e)
                is_retryable = any(x in error_str for x in [
                    "429", "503", "UNAVAILABLE", "quota", "rate",
                    "deadline exceeded", "timeout"
                ])
                
                if is_retryable:
                    retry_count += 1
                    delay = self.base_delay * (2 ** (retry_count - 1)) + random.uniform(0, 1)
                    total_wait += delay
                    
                    logger.warning(
                        f"Excel analysis retry {retry_count}/{self.max_retries}. "
                        f"Waiting {delay:.1f}s..."
                    )
                    
                    if retry_count >= self.max_retries:
                        raise Exception(
                            f"API overloaded after {total_wait:.0f}s. "
                            f"Please try again in a moment. Error: {error_str[:100]}"
                        )
                    
                    time.sleep(delay)
                else:
                    raise Exception(f"Excel analysis failed: {error_str[:200]}")
        
        raise Exception("Max retries exceeded during Excel analysis")
    
    def execute(self, query: str, files: Optional[List[Dict]] = None) -> str:
        """Execute Excel analysis on uploaded files."""
        if not files:
            return "Please upload an Excel file for analysis."
        
        # Filter for Excel files
        excel_files = [
            f for f in files 
            if f.get('type') in ['application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                                  'application/vnd.ms-excel'] or 
            f.get('name', '').endswith(('.xlsx', '.xls'))
        ]
        
        if not excel_files:
            return "❌ No Excel files found in the uploads."
        
        results = []
        for excel_file in excel_files[:3]:  # Limit to 3 files
            try:
                logger.info(f"Analyzing Excel file: {excel_file['name']}")
                result = self._analyze_single_excel(query, excel_file)
                results.append(result)
            except Exception as e:
                logger.error(f"Excel analysis error for {excel_file['name']}: {e}")
                results.append(f"❌ Error analyzing {excel_file['name']}: {str(e)[:150]}")
        
        return "\n\n".join(results) if results else "❌ No Excel files could be analyzed."
    
    def _analyze_single_excel(self, query: str, file: Dict) -> str:
        """Analyze a single Excel file with better data extraction."""
        file_path = file.get('path')
        file_name = file.get('name', 'Unknown')
        
        if not Path(file_path).exists():
            return f"❌ File not found: {file_name}"
        
        try:
            # Read Excel file with better error handling
            df_dict = self._read_excel_file(file_path)
            if not df_dict:
                return f"❌ Could not read data from {file_name}"
            
            # Prepare summary with actual data
            summary = self._prepare_summary(df_dict, file_name)
            
            # Create focused prompt
            prompt = f"""
You are a data analyst. Analyze this Excel file data and answer the user's specific question.

File: {file_name}

Excel Data Summary:
{summary}

User Question: {query}

IMPORTANT INSTRUCTIONS:
1. Answer DIRECTLY using the actual data provided
2. If the data contains what's asked, extract specific values and counts
3. Do NOT make up data or use placeholders
4. For "unique" questions, count distinct values from the data
5. For "how many" questions, provide exact counts from the data
6. Include specific examples from the data when relevant
7. If specific data is not available, clearly state that

Provide a direct, factual analysis:
"""
            
            logger.info(f"Sending analysis request for {file_name} to Gemini...")
            response = self._call_with_retry(prompt, max_tokens=2048)
            
            return f"**Analysis of {file_name}:**\n{response}"
            
        except Exception as e:
            logger.error(f"Error analyzing {file_name}: {e}")
            raise
    
    def _read_excel_file(self, file_path: str) -> Dict[str, pd.DataFrame]:
        """Read Excel file with proper error handling."""
        try:
            df_dict = {}
            xl_file = pd.ExcelFile(file_path)
            
            for sheet_name in xl_file.sheet_names:
                try:
                    # Try to read with automatic header detection
                    df = pd.read_excel(file_path, sheet_name=sheet_name)
                    
                    # Clean up
                    df = df.dropna(how='all').dropna(axis=1, how='all')
                    
                    if len(df) > 0:
                        df_dict[sheet_name] = df
                        logger.info(f"Read sheet '{sheet_name}': {len(df)} rows, {len(df.columns)} columns")
                    
                except Exception as e:
                    logger.warning(f"Could not read sheet {sheet_name}: {e}")
                    continue
            
            return df_dict
            
        except Exception as e:
            logger.error(f"Error reading Excel file: {e}")
            return {}
    
    def _prepare_summary(self, df_dict: Dict[str, pd.DataFrame], file_name: str) -> str:
        """Prepare detailed summary of Excel data."""
        summary_parts = [f"File: {file_name}\n"]
        
        for sheet_name, df in df_dict.items():
            summary_parts.append(f"\n{'='*60}")
            summary_parts.append(f"Sheet: {sheet_name}")
            summary_parts.append(f"{'='*60}")
            summary_parts.append(f"Dimensions: {len(df)} rows × {len(df.columns)} columns\n")
            
            # Column names and types
            summary_parts.append("Columns:")
            for col in df.columns:
                dtype = str(df[col].dtype)
                summary_parts.append(f"  - {col} ({dtype})")
            
            summary_parts.append("\n")
            
            # Data preview
            summary_parts.append("First 10 rows:")
            summary_parts.append(df.head(10).to_string())
            summary_parts.append("\n")
            
            # Statistical summary for numeric columns
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            if numeric_cols:
                summary_parts.append("Numeric Summary:")
                summary_parts.append(df[numeric_cols].describe().round(2).to_string())
                summary_parts.append("\n")
            
            # Unique value counts for categorical columns
            categorical_cols = df.select_dtypes(include=['object']).columns.tolist()[:5]
            if categorical_cols:
                summary_parts.append("Categorical Data:")
                for col in categorical_cols:
                    unique_count = df[col].nunique()
                    summary_parts.append(f"  - {col}: {unique_count} unique values")
                    if unique_count <= 20:
                        summary_parts.append(f"    Values: {df[col].unique().tolist()}")
            
            summary_parts.append("")
        
        return "\n".join(summary_parts)