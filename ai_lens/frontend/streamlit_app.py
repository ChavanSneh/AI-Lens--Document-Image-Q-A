import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
from backend.services.ocr_service import process_document
from backend.services.storage_service import save_document, get_document
from backend.services.qa_service import answer_question
from backend.services.vision_service import run_dual_pipelines
from backend.services.suggestions_service import generate_suggestions

st.set_page_config(page_title="AI Lens", page_icon="👁️", layout="wide")

st.title("👁️ AI Lens - Document & Image Q&A")

# Initialize global history tracking across session loops
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

uploaded_file = st.file_uploader(
    "Upload image, PDF, DOCX or TXT",
    type=["png", "jpg", "jpeg", "pdf", "docx", "txt"]
)

# --- Upload Handling ---
if uploaded_file:
    file_bytes = uploaded_file.read()

    if "doc_id" not in st.session_state:
        # IMAGE FLOW
        if uploaded_file.type.startswith("image/"):
            result = run_dual_pipelines(file_bytes)
            description = result.get("visual_description") or ""
            ocr_text = result.get("extracted_text") or ""
            detected_objs = result.get("detected_objects") or []
            image_type = result.get("image_type") or "image"
            
            if len(ocr_text.strip()) < 20:
                ocr_text = ""

            # Pack all visual telemetry together so the QA service gets full context
            full_text = f"Contextual Metadata:\n- Type: {image_type}\n- Scene Details: {description}\n- Found Items: {detected_objs}\n\nExtracted Reading Content:\n{ocr_text}"
            doc_id = save_document(full_text)
            st.session_state.doc_id = doc_id
            st.success("Image processed successfully")
        
        # DOCUMENT FLOW
        else:
            result = process_document(file_bytes, uploaded_file.name)
            doc_id = save_document(result["text"])
            st.session_state.doc_id = doc_id
            st.success("Document processed successfully")

    # --- Suggestions ---
    text = get_document(st.session_state.doc_id)

    if "suggestions" not in st.session_state:
        st.session_state.suggestions = generate_suggestions(text)

    st.divider()
    col1, col2 = st.columns([4, 1])

    with col1:
        st.subheader("💡 Suggested Questions")

    with col2:
        if st.button("Refresh Suggestions 🔄"):
            st.session_state.suggestions = generate_suggestions(text)
            st.rerun()

    # Render out suggested questions
    suggestions = st.session_state.get("suggestions", [])
    for q in suggestions:
        if st.button(q, use_container_width=True):
            st.session_state.pending_suggestion = q
            st.rerun()

    # --- Interactive Chat History Window ---
    st.divider()
    st.subheader("💬 Ask a Question")

    # This container traps the logs inside a smooth scroll block instead of blowing up your footer layout
    chat_container = st.container(height=400)
    with chat_container:
        for chat in st.session_state.chat_history:
            with st.chat_message(chat["role"]):
                st.write(chat["text"])

    # Read user submission (Checks for standard entry fields or button suggestion transfers)
    user_q = st.chat_input("Type a question here about your uploaded file...")
    
    if st.session_state.get("pending_suggestion"):
        user_q = st.session_state.pop("pending_suggestion")

    if user_q:
        # Display instantly into the live container block
        with chat_container:
            with st.chat_message("user"):
                st.write(user_q)
        
        st.session_state.chat_history.append({"role": "user", "text": user_q})

        doc_id = st.session_state.get("doc_id")
        current_text = get_document(doc_id) if doc_id else ""

        with st.spinner("Thinking..."):
            # Stitch the history into a conversation string for Gemini context mapping
            history_context = ""
            for msg in st.session_state.chat_history[:-1]:
                prefix = "User: " if msg["role"] == "user" else "Assistant: "
                history_context += f"{prefix}{msg['text']}\n"
            
            enhanced_prompt = f"Document Data:\n{current_text}\n\nPast Conversation Logs:\n{history_context}\nCurrent Question: {user_q}"
            
            ans = answer_question(current_text, enhanced_prompt)
            
            # Save response data and display immediately
            st.session_state.chat_history.append({"role": "assistant", "text": ans})
            with chat_container:
                with st.chat_message("assistant"):
                    st.write(ans)

else:
    # Drop all session logs clean if an image or file is deleted
    st.session_state.clear()
    st.info("Please upload a file to begin.")
