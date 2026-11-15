"""Streamlit app: summarize text, PDFs and images, and chat with your PDFs."""

from __future__ import annotations

import streamlit as st

from textsum import __version__
from textsum.chunking import word_count
from textsum.config import load_settings
from textsum.ocr import OcrUnavailableError, image_to_text
from textsum.pdf_io import PdfExtractionError, extract_text
from textsum.qa import DocumentQA
from textsum.summarizer import LENGTHS, Summarizer

st.set_page_config(page_title="Text Summarization using LLM", page_icon="📄", layout="wide")
settings = load_settings()


@st.cache_resource(show_spinner="Loading language model (first run downloads ~1 GB)...")
def get_model(name: str):
    from textsum.models import LocalSeq2Seq

    return LocalSeq2Seq(name)


def read_pdfs(files) -> str:
    return "\n\n".join(extract_text(f) for f in files)


def sidebar() -> str:
    with st.sidebar:
        st.title("📄 Text Summarizer")
        st.caption(f"v{__version__}")
        length = st.select_slider("Summary length", options=list(LENGTHS), value="medium")
        st.divider()
        st.markdown(f"**Local model:** `{settings.model_name}`")
        if settings.gemini_enabled:
            st.markdown(f"**Q&A backend:** Gemini (`{settings.gemini_model}`)")
        else:
            st.markdown("**Q&A backend:** local model")
            st.caption("Set `GOOGLE_API_KEY` to answer questions with Gemini.")
    return length


def summarize_tab(length: str) -> None:
    st.subheader("Summarize")
    source = st.radio("Input", ["Paste text", "PDF", "Image (OCR)"], horizontal=True)
    text = ""
    try:
        if source == "Paste text":
            text = st.text_area("Text to summarize", height=220)
        elif source == "PDF":
            files = st.file_uploader("Upload PDF(s)", type=["pdf"], accept_multiple_files=True)
            if files:
                text = read_pdfs(files)
        else:
            image = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
            if image:
                st.image(image, width=360)
                text = image_to_text(image, settings.tesseract_cmd)
        if text and source != "Paste text":
            with st.expander(f"Extracted text ({word_count(text):,} words)"):
                st.text(text[:5000] + ("..." if len(text) > 5000 else ""))
    except (PdfExtractionError, OcrUnavailableError) as exc:
        st.error(str(exc))
        return

    if st.button("Summarize", type="primary", disabled=not text.strip()):
        bar = st.progress(0.0, text="Summarizing...")
        try:
            summary = Summarizer(get_model(settings.model_name)).summarize(
                text, length, progress=lambda p: bar.progress(min(p, 1.0), text="Summarizing...")
            )
        except ValueError as exc:
            bar.empty()
            st.warning(str(exc))
            return
        bar.empty()
        st.success(summary)
        st.caption(f"{word_count(text):,} words → {word_count(summary):,} words")
        st.download_button("Download summary", summary, file_name="summary.txt")


def qa_tab() -> None:
    st.subheader("Ask your PDF")
    files = st.file_uploader(
        "Upload PDF(s)", type=["pdf"], accept_multiple_files=True, key="qa_files"
    )
    if not files:
        st.info("Upload one or more PDFs, then ask a question.")
        return

    key = tuple((f.name, f.size) for f in files)
    if st.session_state.get("qa_key") != key:
        try:
            with st.spinner("Indexing..."):
                text = read_pdfs(files)
                local = None if settings.gemini_enabled else get_model(settings.model_name)
                st.session_state.qa = DocumentQA(text, settings, local)
                st.session_state.qa_key = key
        except PdfExtractionError as exc:
            st.error(str(exc))
            return
    qa: DocumentQA = st.session_state.qa
    st.caption(f"{len(qa.index.passages)} passages indexed · backend: {qa.backend}")

    question = st.text_input("Your question")
    if question:
        with st.spinner("Thinking..."):
            try:
                answer = qa.ask(question)
            except Exception as exc:  # network / API errors from Gemini
                st.error(f"Could not generate an answer: {exc}")
                return
        st.markdown(f"**Answer:** {answer.text}")
        with st.expander("Source passages"):
            for hit in answer.sources:
                st.markdown(f"**Passage {hit.index + 1}** (score {hit.score:.2f})")
                st.write(hit.text)


def about_tab() -> None:
    st.markdown(
        "Summarization runs **locally** with LaMini-Flan-T5 (map-reduce over token chunks, so "
        "long documents work). PDF Q&A retrieves relevant passages with BM25 and answers with "
        "Gemini (if `GOOGLE_API_KEY` is set) or the local model. Images are read with Tesseract "
        "OCR. See the project README for setup details."
    )


def main() -> None:
    length = sidebar()
    tab1, tab2, tab3 = st.tabs(["Summarize", "Ask your PDF", "About"])
    with tab1:
        summarize_tab(length)
    with tab2:
        qa_tab()
    with tab3:
        about_tab()


main()
