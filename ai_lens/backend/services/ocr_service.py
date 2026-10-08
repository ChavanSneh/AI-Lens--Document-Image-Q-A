import io
import os
import json
import re
from PIL import Image
import pytesseract
from pdf2image import convert_from_bytes
import pdfplumber
try:
    import docx  # python-docx
except ImportError:
    docx = None

try:
    from google  import genai
    from dotenv import load_dotenv
    load_dotenv()
    api_key = os.getenv("GROQ_API_KEY")
    GEMINI_AVAILABLE = True
except Exception:
    GEMINI_AVAILABLE = False

def ocr_image(file_bytes: bytes) -> str:
    image = Image.open(io.BytesIO(file_bytes)).convert("L")
    return pytesseract.image_to_string(image)

def ocr_pdf(file_bytes: bytes) -> str:
    # 1) Try native text extraction (fast, accurate)
    try:
        text_parts = []
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    text_parts.append(text)
        if text_parts:
            return "\n".join(text_parts)
    except Exception:
        pass

    # 2) Fallback: scanned PDF → OCR each page
    pages = convert_from_bytes(file_bytes, dpi=300)
    return "\n".join(pytesseract.image_to_string(p.convert("L")) for p in pages)

def process_docx(file_bytes: bytes) -> str:
    if docx is None:
        raise RuntimeError("python-docx not installed")
    doc = docx.Document(io.BytesIO(file_bytes))
    return "\n".join(p.text for p in doc.paragraphs)

def process_txt(file_bytes: bytes) -> str:
    return file_bytes.decode("utf-8", errors="ignore")

def clean_text(text: str) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return "\n".join(lines)
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def interpret_with_llm(text: str) -> dict:
    if not GEMINI_AVAILABLE:
        return {"summary": text, "amount": None, "type": None}

    model = genai.GenerativeModel("gemini-2.5-flash-lite")
    prompt = f"""
You are a document parser. 
1. Clean the OCR text (fix broken words, extra spaces, misread characters).
2. Extract the key information as JSON with fields:
   - "summary": short clean version of the text
   - "amount": number if a total/amount exists, else null
   - "type": "Expense" or "Income" if you can tell, else null

Return ONLY valid JSON.

Text:
{text}
"""
    response = model.generate_content(prompt)
    m = re.search(r"\{.*\}", response.text, re.S)
    return json.loads(m.group(0)) if m else {"summary": text, "amount": None, "type": None}

def process_document(file_bytes: bytes, filename: str) -> dict:
    name = filename.lower()
    if name.endswith(".pdf"):
        raw = ocr_pdf(file_bytes)
    elif name.endswith((".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp")):
        raw = ocr_image(file_bytes)
    elif name.endswith(".txt"):
        raw = process_txt(file_bytes)
    elif name.endswith(".docx"):
        raw = process_docx(file_bytes)
    else:
        raise ValueError("Unsupported file type. Use image, PDF, .txt, or .docx")

    cleaned = clean_text(raw)
    parsed = interpret_with_llm(cleaned)
    return {"text": cleaned, "parsed": parsed}