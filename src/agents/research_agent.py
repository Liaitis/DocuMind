"""
Research Agent - Premium Version with Enhanced Rate Limit Handling
Uses the new google-genai SDK with improved retry and error handling.
"""

from __future__ import annotations

import logging
import traceback
from typing import Optional, List, Dict, Any
import time
import random
import os

logger = logging.getLogger(__name__)

from google import genai
from google.genai import types
from google.genai.errors import APIError

# Import vector store
from src.rag.vector_store import VectorStore

class ResearchAgent:
    """Autonomous research agent using Gemini API (new google-genai)."""
    
    def __init__(self, api_key: str, vector_store: Optional[VectorStore] = None):
        """Initialize the research agent."""
        logger.info("Initializing ResearchAgent with google-genai...")
        
        if not api_key:
            raise ValueError("API key is required")
        
        self.api_key = api_key
        self.max_retries = 5
        self.base_delay = 2
        self.max_wait_time = 120  # Max 2 minutes total
        
        # Create the new client
        try:
            logger.info("Creating genai.Client...")
            self.client = genai.Client(api_key=api_key)
            logger.info("[OK] genai.Client created")
        except Exception as e:
            logger.error(f"Error creating genai.Client: {e}")
            raise
        
        try:
            # Initialize vector store
            logger.info("Initializing vector store...")
            self.vector_store = vector_store or VectorStore()
            logger.info("[OK] Vector store initialized")
            
            # Initialize tools
            logger.info("Initializing tools...")
            self.tools = {}
            self._initialize_tools()
            logger.info("[OK] Tools initialized")
            
        except Exception as e:
            logger.error(f"Error during initialization: {e}")
            logger.error(traceback.format_exc())
            raise
        
        # Initialize model (will be set later)
        self.model_name = self._select_model()
        logger.info(f"[OK] Using model: {self.model_name}")
        
        logger.info("[OK][OK][OK] ResearchAgent initialized successfully")
    
    def _initialize_tools(self):
        """Initialize tools with lazy imports."""
        try:
            from src.tools.document_search import DocumentSearchTool
            self.tools["search_documents"] = DocumentSearchTool(self.vector_store)
            logger.info("[OK] DocumentSearchTool initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize DocumentSearchTool: {e}")
            self.tools["search_documents"] = None
        
        try:
            from src.tools.web_search import WebSearchTool
            self.tools["web_search"] = WebSearchTool()
            logger.info("[OK] WebSearchTool initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize WebSearchTool: {e}")
            self.tools["web_search"] = None
        
        try:
            from src.tools.pdf_analyzer import PDFAnalyzerTool
            self.tools["analyze_pdf"] = PDFAnalyzerTool(self.api_key)
            logger.info("[OK] PDFAnalyzerTool initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize PDFAnalyzerTool: {e}")
            self.tools["analyze_pdf"] = None
        
        try:
            from src.tools.excel_analyzer import ExcelAnalyzerTool
            self.tools["analyze_excel"] = ExcelAnalyzerTool(self.api_key)
            logger.info("[OK] ExcelAnalyzerTool initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize ExcelAnalyzerTool: {e}")
            self.tools["analyze_excel"] = None
    
    def _select_model(self) -> str:
        """
        Select the best available model by testing a simple call.
        Returns model name as string (without 'models/' prefix).
        """
        model_priority = [
            "gemini-2.0-flash",
            "gemini-flash-latest",
            "gemini-1.5-flash",
            "gemini-pro-latest",
            "gemini-2.5-flash-lite",
            "gemini-2.0-flash-lite",
        ]
        
        for model_name in model_priority:
            try:
                logger.info(f"Testing model: {model_name}")
                # Try a minimal generate call
                response = self.client.models.generate_content(
                    model=model_name,
                    contents="Say OK"
                )
                if response and response.text:
                    logger.info(f"✅ Model {model_name} is working")
                    return model_name
            except Exception as e:
                if "429" in str(e) or "quota" in str(e).lower():
                    logger.warning(f"Quota exceeded for {model_name}, trying next...")
                else:
                    logger.warning(f"Model {model_name} failed: {e}")
                continue
        
        # Ultimate fallback
        logger.warning("No preferred model worked, falling back to 'gemini-2.0-flash'")
        return "gemini-2.0-flash"
    
    def _call_with_retry(self, prompt: str, temperature: float = 0.2, max_tokens: int = 2048) -> str:
        """
        Call Gemini API with enhanced exponential backoff retry logic.
        Handles 503, 429, and rate limit errors gracefully.
        """
        retry_count = 0
        total_wait_time = 0
        
        while retry_count < self.max_retries:
            try:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config={
                        "temperature": temperature,
                        "max_output_tokens": max_tokens,
                    }
                )
                if response and response.text:
                    return response.text
                else:
                    raise Exception("Empty response from model")
                    
            except Exception as e:
                error_str = str(e)
                
                # Check if it's a retryable error
                is_retryable = any(x in error_str for x in [
                    "429", "503", "UNAVAILABLE", "quota", "rate", 
                    "deadline exceeded", "timeout", "temporarily unavailable"
                ])
                
                if is_retryable:
                    retry_count += 1
                    
                    # Calculate exponential backoff with jitter
                    exponential_delay = self.base_delay * (2 ** (retry_count - 1))
                    jitter = random.uniform(0, 1)
                    delay = exponential_delay + jitter
                    
                    # Cap total wait time
                    if total_wait_time + delay > self.max_wait_time:
                        delay = max(5, self.max_wait_time - total_wait_time)
                    
                    total_wait_time += delay
                    
                    logger.warning(
                        f"Retryable error (attempt {retry_count}/{self.max_retries}). "
                        f"Waiting {delay:.2f}s... Error: {error_str[:100]}"
                    )
                    
                    if retry_count >= self.max_retries:
                        raise Exception(
                            f"Max retries ({self.max_retries}) exceeded after {total_wait_time:.1f}s. "
                            f"API is experiencing high demand. Last error: {error_str[:200]}"
                        )
                    
                    time.sleep(delay)
                else:
                    # Non-retryable error
                    logger.error(f"Non-retryable error: {error_str}")
                    raise e
        
        raise Exception("Max retries exceeded")
    
    def process(self, user_input: str, uploaded_files: Optional[List[Dict]] = None) -> str:
        """Process a user request and return the agent's response."""
        logger.info(f"Processing request: {user_input[:100]}...")
        
        if not user_input.strip():
            return "Please provide a question or task."
        
        try:
            # Step 1: Decide tools based on files and query
            logger.info("Step 1: Deciding which tools to use...")
            tool_plan = self._decide_tools(user_input, uploaded_files)
            logger.info(f"Tools selected: {tool_plan}")
            
            # Step 2: Execute tools
            logger.info("Step 2: Executing tools...")
            results = self._execute_tools(tool_plan, user_input, uploaded_files)
            logger.info("Tool execution completed")
            
            # Step 3: Generate response
            logger.info("Step 3: Generating final response...")
            response = self._generate_response(user_input, results)
            logger.info("Response generated successfully")
            
            return response
            
        except Exception as e:
            logger.error(f"Error in process(): {str(e)}")
            logger.error(traceback.format_exc())
            return f"❌ Error processing request: {str(e)}"
    
    def _decide_tools(self, user_input: str, files: Optional[List[Dict]] = None) -> List[str]:
        """Decide which tools to use based on files and query."""
        logger.info("Deciding tools...")
        
        # Smart tool selection based on file types
        selected_tools = []
        
        if files:
            has_excel = any(f.get('type') in ['application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                                               'application/vnd.ms-excel'] or 
                           f.get('name', '').endswith(('.xlsx', '.xls')) for f in files)
            has_pdf = any(f.get('type') == 'application/pdf' or f.get('name', '').endswith('.pdf') for f in files)
            
            if has_excel:
                selected_tools.append("analyze_excel")
            if has_pdf:
                selected_tools.append("analyze_pdf")
        
        # Always add document search if we have documents
        if self.vector_store.get_count() > 0:
            selected_tools.append("search_documents")
        
        # If no specific file types or searching uploaded files
        if not selected_tools:
            selected_tools.append("search_documents")
        
        # Add web search for current information queries
        current_query_keywords = ["latest", "recent", "current", "today", "news", "breaking"]
        if any(kw in user_input.lower() for kw in current_query_keywords):
            selected_tools.append("web_search")
        
        valid_tools = [t for t in selected_tools if t in self.tools and self.tools[t] is not None]
        logger.info(f"Selected tools: {valid_tools}")
        
        return valid_tools if valid_tools else ["search_documents"]
    
    def _execute_tools(self, tools: List[str], user_input: str, files: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """Execute the chosen tools."""
        logger.info(f"Executing {len(tools)} tools...")
        results = {}
        
        for tool_name in tools:
            tool = self.tools.get(tool_name)
            if not tool:
                continue
            try:
                start_time = time.time()
                if tool_name in ["analyze_pdf", "analyze_excel"]:
                    if files:
                        results[tool_name] = tool.execute(user_input, files)
                    else:
                        results[tool_name] = "No file provided for analysis."
                else:
                    results[tool_name] = tool.execute(user_input)
                elapsed_time = time.time() - start_time
                logger.info(f"  {tool_name}: Completed in {elapsed_time:.2f}s")
            except Exception as e:
                logger.error(f"Tool execution error ({tool_name}): {e}")
                results[tool_name] = f"⚠️ {tool_name} error: {str(e)[:100]}"
        
        return results
    
    def _generate_response(self, user_input: str, results: Dict[str, Any]) -> str:
        """Generate final response with retry logic."""
        logger.info("Generating final response...")
        
        try:
            tool_results = "\n".join([
                f"[{name}]: {result}"
                for name, result in results.items()
                if result
            ])
            if not tool_results.strip():
                tool_results = "No specific information was found."
            
            prompt = f"""
User asked: "{user_input}"

Information gathered from tools:
{tool_results}

Write a comprehensive, well-structured response that:
1. Directly addresses the user's question
2. Cites where information came from
3. Provides insights and key takeaways
4. Is clear and professional
5. Mentions if any tools failed or had issues
"""
            logger.info("Requesting response generation from Gemini...")
            start_time = time.time()
            
            response_text = self._call_with_retry(prompt, temperature=0.2)
            
            elapsed_time = time.time() - start_time
            logger.info(f"Response generated in {elapsed_time:.2f}s")
            
            return response_text
            
        except Exception as e:
            logger.error(f"Response generation error: {e}")
            return f"❌ Error generating response: {str(e)}"