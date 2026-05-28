import io
import re
import json
import os
from PIL import Image
import pytesseract
from google import genai
from google.genai import types  # <-- Needed for config formatting
from pydantic import BaseModel  
from ultralytics import YOLO
from dotenv import load_dotenv

load_dotenv()

# Initialize new SDK client
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("GOOGLE_API_KEY not found")

client = genai.Client(api_key=api_key)
yolo_model = YOLO("yolov8n.pt") 

# 1. Define the exact JSON structure you want from Gemini
class ImageAnalysis(BaseModel):
    image_type: str
    visual_description: str

def ocr_image(file_bytes: bytes) -> str:
    image = Image.open(io.BytesIO(file_bytes)).convert("L")
    return pytesseract.image_to_string(image)

def yolo_detect(file_bytes: bytes) -> dict:
    image = Image.open(io.BytesIO(file_bytes))
    results = yolo_model(image)
    objects = []
    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            objects.append({
                "label": r.names[cls],
                "confidence": float(box.conf[0])
            })
    return {"detected_objects": objects}

def vision_describe(file_bytes: bytes) -> dict:
    img = Image.open(io.BytesIO(file_bytes))
    
    prompt = "Analyze this image and break down its details."
    
    # Corrected SDK layout using types.GenerateContentConfig
    try:
        resp = client.models.generate_content(
            model="gemini-2.5-flash-lite",
            contents=[prompt, img],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=ImageAnalysis,
            ),
        )
        return json.loads(resp.text)
    except Exception:
        # Absolute bulletproof fallback safety net
        return {"image_type": "unknown", "visual_description": "Failed to parse analysis"}

def run_dual_pipelines(file_bytes: bytes) -> dict:
    # Pipeline A: OCR
    extracted_text = ocr_image(file_bytes)

    # Pipeline B: Image understanding (YOLO + Gemini description)
    yolo_info = yolo_detect(file_bytes)
    vision_info = vision_describe(file_bytes)

    merged = {
        "image_type": vision_info.get("image_type"),
        "visual_description": vision_info.get("visual_description"),
        "detected_objects": yolo_info.get("detected_objects"),
        "extracted_text": extracted_text.strip()
    }
    return merged
