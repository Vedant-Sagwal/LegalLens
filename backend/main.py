import os
import shutil
import logging
import json
import re
import uuid
from pathlib import Path

from dotenv import load_dotenv
import google.generativeai as genai
import fitz  # PyMuPDF

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List

# Load environment variables
load_dotenv()

# Logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load Gemini API key
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    logging.error("FATAL: GEMINI_API_KEY not found in environment variables.")
    raise RuntimeError("GEMINI_API_KEY is required but not found in environment variables.")

genai.configure(api_key=api_key)

# Load Gemini model
llm = genai.GenerativeModel("gemini-1.5-flash")

# Constants
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
ALLOWED_EXTENSIONS = {'.pdf'}

def validate_file(uploaded_file: UploadFile) -> None:
    """Validate uploaded file type and size."""
    if not uploaded_file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")
    
    file_extension = Path(uploaded_file.filename).suffix.lower()
    if file_extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400, 
            detail=f"Only {', '.join(ALLOWED_EXTENSIONS)} files are supported"
        )
    
    # Note: uploaded_file.size might not always be available
    # In production, you might want to check size during reading

def read_pdf_file(file_path: str) -> str:
    """Read text from PDF using PyMuPDF with better error handling."""
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        doc = fitz.open(file_path)
        text = ""
        
        if doc.page_count == 0:
            raise ValueError("PDF contains no pages")
        
        for page_num in range(doc.page_count):
            page = doc[page_num]
            text += page.get_text()
        
        doc.close()
        
        if not text.strip():
            raise ValueError("No text content found in PDF")
        
        return text
    except Exception as e:
        logging.error(f"Error reading PDF file at {file_path}: {e}")
        raise

def extract_json_from_response(response_text: str, context: str = "response") -> dict:
    """Extract and validate JSON from AI response."""
    # Try to find JSON object
    json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
    if not json_match:
        logging.error(f"No JSON object found in {context}")
        raise ValueError(f"No valid JSON object found in {context}")
    
    clean_json = json_match.group(0)
    
    try:
        return json.loads(clean_json)
    except json.JSONDecodeError as e:
        logging.error(f"Invalid JSON in {context}: {e}")
        raise ValueError(f"Invalid JSON in {context}: {e}")

def get_summary(text: str) -> Dict[str, List[str]]:
    """Generate structured JSON summary with improved error handling."""
    try:
        prompt = f"""
        You are an excellent paralegal in a big law firm.
        Summarize the following legal document in a structured and simplified JSON format.

        **Instructions:**
        1. Analyze the document and break it down into clear key headings.
        2. Under each heading, list important points as bullet points (in plain text).
        3. Output the result as a valid JSON object where:
           - Each key is a heading
           - The value is a list of bullet points under that heading
        4. If no useful content is found for a heading, omit it.
        5. **CRITICAL: Do NOT include markdown formatting or extra text. Only output a pure JSON object starting with '{{' and ending with '}}'.**

        Legal Document:
        ---
        {text[:8000]}  # Truncate to avoid token limits
        ---
        """
        
        response = llm.generate_content(prompt)
        
        if not response.text:
            raise ValueError("Empty response from Gemini API")
        
        return extract_json_from_response(response.text, "summary")

    except Exception as e:
        logging.error(f"Error generating structured summary: {e}")
        return {
            "Summary Error": [f"Could not generate summary: {str(e)}"]
        }

def extract_clauses(text: str) -> Dict[str, List[str]]:
    """Extract legal clauses with improved error handling."""
    try:
        prompt = f"""
        Analyze the following legal document. Your task is to extract key clauses and categorize them.

        **Instructions:**
        1. Identify clauses related to "liability", "termination", and "confidentiality".
        2. Format your entire output as a single, valid JSON object.
        3. The keys of the object must be "liability", "termination", and "confidentiality".
        4. The value for each key must be an array of strings, where each string is a direct quote of a clause from the document.
        5. If no clauses are found for a category, the value must be an empty array [].
        6. **CRITICAL: Do NOT include any text, explanations, or markdown formatting (like ```json) before or after the JSON object.** Your response must begin with `{{` and end with `}}`.

        Legal Document Text:
        ---
        {text[:8000]}  # Truncate to avoid token limits
        ---
        """
        
        response = llm.generate_content(prompt)
        
        if not response.text:
            raise ValueError("Empty response from Gemini API")
        
        result = extract_json_from_response(response.text, "clause extraction")
        
        # Validate structure
        expected_keys = {"liability", "termination", "confidentiality"}
        if not all(key in result for key in expected_keys):
            raise ValueError("Response missing required clause categories")
        
        return result

    except Exception as e:
        logging.error(f"Error extracting clauses: {e}")
        return {
            "liability": [f"Error occurred while extracting clauses: {str(e)}"],
            "termination": [],
            "confidentiality": []
        }

# FastAPI app
app = FastAPI(
    title="Legal-Lens API",
    description="API for analyzing legal documents and extracting key information",
    version="1.0.0"
)

# CORS for frontend access
origins = [
    "https://legallensfrontend.onrender.com",
    "http://localhost:3000",  # For development
    "http://127.0.0.1:3000"   # For development
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class ResultStructure(BaseModel):
    summary: Dict[str, List[str]]
    clauses: Dict[str, List[str]]

class HealthResponse(BaseModel):
    status: str
    message: str

# Health check endpoint
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(status="healthy", message="API is running")

# Main API Endpoint
@app.post("/simplify_document", response_model=ResultStructure)
async def simplify_document(uploaded_file: UploadFile = File(...)):
    """
    Analyze a legal document and extract summary and key clauses.
    
    Args:
        uploaded_file: PDF file to analyze
    
    Returns:
        ResultStructure: Contains document summary and extracted clauses
    """
    # Validate file
    validate_file(uploaded_file)
    
    # Generate unique filename to prevent conflicts
    unique_id = str(uuid.uuid4())[:8]
    safe_filename = re.sub(r'[^a-zA-Z0-9._-]', '_', uploaded_file.filename)
    file_path = f"temp_{unique_id}_{safe_filename}"
    
    try:
        # Check file size while saving
        file_size = 0
        with open(file_path, "wb") as buffer:
            while chunk := await uploaded_file.read(8192):  # Read in chunks
                file_size += len(chunk)
                if file_size > MAX_FILE_SIZE:
                    raise HTTPException(
                        status_code=413, 
                        detail=f"File too large. Maximum size: {MAX_FILE_SIZE // (1024*1024)}MB"
                    )
                buffer.write(chunk)

        # Read and parse document
        doc_text = read_pdf_file(file_path)
        
        # Generate summary and extract clauses
        summary = get_summary(doc_text)
        clauses = extract_clauses(doc_text)

        return ResultStructure(summary=summary, clauses=clauses)

    except HTTPException:
        raise  # Re-raise HTTP exceptions as-is
    except FileNotFoundError as e:
        logging.error(f"File not found: {e}")
        raise HTTPException(status_code=404, detail="Uploaded file could not be found")
    except ValueError as e:
        logging.error(f"Value error during processing: {e}")
        raise HTTPException(status_code=400, detail=f"Document processing error: {str(e)}")
    except Exception as e:
        logging.error(f"Unexpected error during document analysis: {e}")
        raise HTTPException(
            status_code=500, 
            detail="An internal server error occurred during document analysis"
        )
    finally:
        # Cleanup temp file
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
                logging.info(f"Cleaned up temporary file: {file_path}")
            except Exception as cleanup_error:
                logging.error(f"Failed to cleanup temporary file {file_path}: {cleanup_error}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
