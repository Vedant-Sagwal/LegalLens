import os
import shutil
import logging
import json
import re

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
else:
    genai.configure(api_key=api_key)

# Load Gemini model
llm = genai.GenerativeModel("gemini-1.5-flash")

# Read text from PDF using PyMuPDF
def read_pdf_file(file_path: str) -> str:
    try:
        doc = fitz.open(file_path)
        text = ""
        for page in doc:
            text += page.get_text()
        return text
    except Exception as e:
        logging.error(f"Error reading PDF file at {file_path}: {e}")
        return ""

# Generate structured JSON summary
def get_summary(text: str) -> Dict[str, List[str]]:
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
        {text}
        ---
        """
        response = llm.generate_content(prompt)

        json_match = re.search(r"\{.*\}", response.text, re.DOTALL)
        if not json_match:
            raise ValueError("No valid JSON object found in Gemini response for summary.")

        clean_json = json_match.group(0)
        return json.loads(clean_json)

    except Exception as e:
        logging.error(f"Error generating structured summary: {e}")
        return {
            "Summary Error": ["Could not generate summary due to an API error."]
        }

# Extract legal clauses (liability, termination, confidentiality)
def extract_clauses(text: str) -> str:
    try:
        prompt = f"""
        Analyze the following legal document. Your task is to extract key clauses and categorize them.

        **Instructions:**
        1.  Identify clauses related to "liability", "termination", and "confidentiality".
        2.  Format your entire output as a single, valid JSON object.
        3.  The keys of the object must be "liability", "termination", and "confidentiality".
        4.  The value for each key must be an array of strings, where each string is a direct quote of a clause from the document.
        5.  If no clauses are found for a category, the value must be an empty array [].
        6.  **CRITICAL: Do NOT include any text, explanations, or markdown formatting (like ```json) before or after the JSON object.** Your response must begin with `{{` and end with `}}`.

        Legal Document Text:
        ---
        {text}
        ---
        """
        response = llm.generate_content(prompt)

        json_match = re.search(r"\{.*\}", response.text, re.DOTALL)
        if not json_match:
            raise ValueError("No valid JSON object found in Gemini response.")

        clean_json = json_match.group(0)
        return clean_json

    except Exception as e:
        logging.error(f"Error with Gemini API during clause extraction: {e}")
        return json.dumps({
            "liability": ["An error occurred while trying to extract clauses from the document."],
            "termination": [],
            "confidentiality": []
        })

# FastAPI app
app = FastAPI(title="Legal-Lens API")

# CORS for frontend access
origins = [
    "https://legallensfrontend.onrender.com"
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic model for API response
class ResultStructure(BaseModel):
    summary: Dict[str, List[str]]
    clauses: Dict[str, List[str]]

# API Endpoint
@app.post("/simplify_document", response_model=ResultStructure)
async def simplify_document(uploaded_file: UploadFile = File(...)):

    file_path = f"temp_{uploaded_file.filename}"

    try:
        # Save file locally
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(uploaded_file.file, buffer)

        # Read and parse
        doc_text = read_pdf_file(file_path)
        if not doc_text:
            logging.error("Document text is empty. The file might be unreadable or empty.")
            raise HTTPException(status_code=400, detail="Could not read text from the uploaded document.")

        # Generate summary & extract clauses
        summary = get_summary(doc_text)
        clauses_json_string = extract_clauses(doc_text)

        # Parse clauses JSON
        try:
            clauses_dict = json.loads(clauses_json_string)
        except json.JSONDecodeError:
            logging.error("Failed to parse clauses JSON from AI response. The response was not valid JSON.")
            raise HTTPException(status_code=500, detail="The AI failed to structure the clauses correctly. Please try again.")

        return {"summary": summary, "clauses": clauses_dict}

    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        logging.error(f"An unexpected server error occurred: {e}")
        raise HTTPException(status_code=500, detail="An internal server error occurred during analysis.")
    finally:
        # Cleanup temp file
        if os.path.exists(file_path):
            os.remove(file_path)
