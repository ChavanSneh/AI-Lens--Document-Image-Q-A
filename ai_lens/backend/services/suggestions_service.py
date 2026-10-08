import os
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq

# ensure .env is loaded from project root regardless of working directory
env_path = Path(__file__).resolve().parents[2] / ".env"
load_dotenv(dotenv_path=env_path)

api_key = os.getenv("GROQ_API_KEY") or os.getenv("GROQ_KEY")
if not api_key:
    raise ValueError("GROQ_API_KEY (or GROQ_KEY) not found in environment")

client = Groq(api_key=api_key)


def generate_suggestions(document_text: str) -> list[str]:
    prompt = f"""
You are an assistant that generates helpful questions a user might ask about a document or image.

The content may include:
- Extracted text (OCR)
- Detected objects
- Image description

Based on this content, generate 5 useful and diverse questions.

Rules:
- Keep questions short and clear
- Cover different aspects (text, meaning, objects)
- Do not repeat similar questions
- Return only a numbered list

Content:
{document_text}
"""

    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {"role": "system", "content": "You generate helpful questions from documents."},
            {"role": "user", "content": prompt}
        ]
    )

    text = response.choices[0].message.content.strip()

    # Convert numbered list into Python list
    suggestions = []
    for line in text.split("\n"):
        line = line.strip()
        if line:
            # remove numbering like "1. "
            if line[0].isdigit():
                line = line.split(".", 1)[-1].strip()
            line = line.lstrip("-* ").strip()  # extra safety
            suggestions.append(line)

    return suggestions[:5]