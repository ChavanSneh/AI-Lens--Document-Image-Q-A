import os
import re
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("GOOGLE_API_KEY not found in environment")

client = genai.Client(api_key=api_key)


def _chunk_text(text: str, size: int = 800) -> list[str]:
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks = []

    for p in paragraphs:
        if len(p) <= size:
            chunks.append(p)
        else:
            chunks.extend([p[i:i+size] for i in range(0, len(p), size)])

    return chunks


def _pick_chunks(chunks: list[str], question: str, k: int = 3) -> list[str]:
    q_words = set(question.lower().split())
    scored = []

    for c in chunks:
        # Boost chunks that contain metadata headers like "Visual Scene" or "Metadata"
        # so visual context never gets discarded for image queries
        boost = 5 if "metadata" in c.lower() or "scene" in c.lower() else 0
        score = sum(c.lower().count(w) for w in q_words) + boost
        scored.append((score, c))

    scored.sort(key=lambda x: x[0], reverse=True)

    top_chunks = [c for score, c in scored[:k]]
    return top_chunks if top_chunks else chunks[:k]


def answer_question(document_text: str, question: str) -> str:
    chunks = _chunk_text(document_text)
    selected_chunks = _pick_chunks(chunks, question)
    context = "\n\n---\n\n".join(selected_chunks)

    # Shifting the system instruction into a permanent spatial & logical analyzer personality
    system_instruction = """
You are the elite cognitive brain of the AI Lens app. Your job is to answer the user's question by conducting a highly detailed logical, spatial, and geometric analysis of the provided text or metadata context.

CRITICAL INSTRUCTION FOR IMAGE QUERIES: 
If the context indicates this is an image, graphic design, or UI screenshot, do not just search for literal text matches. Use your immense real-world common sense, inductive reasoning, and spatial understanding. 
For example, if a user asks about background patterns, lines, colors, or shapes (like rectangles, circles, or selection handles), use the visual scene details and your general knowledge of how these items look in the real world to deduce the answer.

Rules:
1. Be direct, clear, and natural.
2. If the user asks you to count or evaluate something not explicitly mentioned in the text, use everyday logic to fill in the blanks responsibly instead of quitting.
3. Only if a question is completely impossible to answer or infer from both the data and common sense, say: "I couldn't find enough context to answer that accurately."
"""

    prompt = f"""
Context Baseline Data:
{context}

Current User Question / History Thread:
{question}

Answer:
"""

    response = client.models.generate_content(
        model="gemini-2.5-flash-lite",
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=0.4  # Slightly lowered for more grounded, accurate deductions
        )
    )

    return (response.text or "").strip()
