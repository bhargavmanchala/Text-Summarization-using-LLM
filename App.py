import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
from PyPDF2 import PdfReader
from PyPDF2.errors import PdfReadError
from PIL import Image
import pytesseract

# Configuration
local_model_path = "D:\\MINI PROJECT\\Text_Summarization\\La-Mini-Flan-T5-248M"

# Set the tesseract executable path
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Load models
try:
    tokenizer = T5Tokenizer.from_pretrained(local_model_path)
    base_model = T5ForConditionalGeneration.from_pretrained(local_model_path, device_map='auto', torch_dtype=torch.float32)
except ImportError as e:
    st.error(f"Failed to load model or tokenizer: {e}")
    st.stop()

# Summarization and QA functions
def file_preprocessing(file):
    loader = PyPDFLoader(file)
    pages = loader.load_and_split()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)
    texts = text_splitter.split_documents(pages)
    final_texts = "".join(text.page_content for text in texts)
    return final_texts

def llm_pipeline(input_text, max_length, task='summarize'):
    if len(input_text.split()) < max_length:
        st.error("Input text is shorter than the desired length. Please adjust the length.")
        return None

    # Tokenize input text with truncation to the max length
    inputs = tokenizer(input_text, max_length=2048, truncation=True, return_tensors="pt")
    input_ids = inputs["input_ids"].to(base_model.device)
    attention_mask = inputs["attention_mask"].to(base_model.device)

    # Generate summary or answer
    if task == 'summarize':
        summary_ids = base_model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=max_length,
            min_length=int(max_length * 0.6),
            no_repeat_ngram_size=3,
            length_penalty=2.0,
            early_stopping=True
        )
        output = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    
    elif task == 'qa':
        prompt = f"Answer the following question based on the given text: {input_text}"
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(base_model.device)
        attention_mask = inputs["attention_mask"].to(base_model.device)
        answer_ids = base_model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=150,  # Adjust as needed
            no_repeat_ngram_size=3,
            length_penalty=2.0,
            early_stopping=True
        )
        output = tokenizer.decode(answer_ids[0], skip_special_tokens=True)
    
    return output

def summarize_in_chunks(text, chunk_size=200, max_length=1000):
    chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
    summaries = []
    
    for chunk in chunks:
        summary = llm_pipeline(chunk, max_length, task='summarize')
        if summary:
            summaries.append(summary)
    
    final_summary = " ".join(summaries)
    return final_summary

def word_count(text):
    return len(text.split())

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        try:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text
                else:
                    st.warning("Some pages could not be read.")
        except PdfReadError:
            st.error(f"Error reading PDF file: {pdf.name}")
        except Exception as e:
            st.error(f"An unexpected error occurred: {str(e)}")
    return text

def process_image(image):
    try:
        img = Image.open(image)
        text = pytesseract.image_to_string(img)
        return text
    except Exception as e:
        st.error(f"An error occurred while processing the image: {str(e)}")
        return ""

# Main function
def main():
    st.set_page_config(layout='wide', page_title="📄 Multi-Function Text Processing Tool")

    # Sidebar for choosing action
    option = st.sidebar.selectbox(
        'Choose an action:',
        ['Text Summarization', 'PDF & Image to Text', 'QA from Text', 'Grammar Check']
    )

    if option == 'Text Summarization':
        st.title('Text Summarization using LLM')
        st.write("Welcome to the **Text Processing Tool** where you can analyze and summarize documents! 🔍📚")
        st.write("Use the **text summarization** and **question answering** features to get insights from your documents. 🔍✍️")

        # File upload section
        st.write("Upload your document 📄 and let’s get started!")

        # Text Summarization
        input_text = st.text_area("Input Text for Summarization", height=150)
        summary_length_words = st.number_input("Select Summary Length (in words)", min_value=25, max_value=1000, value=250)

        if st.button("Get Summary"):
            if len(input_text.split()) < summary_length_words:
                st.error("Input text is shorter than the desired summary length. Please provide a longer text.")
            else:
                summary = llm_pipeline(input_text, summary_length_words, task='summarize')
                if summary:
                    st.success(summary)
                    st.write(f"Word Count: {word_count(summary)}")

    elif option == 'PDF & Image to Text':
        st.title('PDF & Image to Text Processing')

        # PDF Upload and Processing
        pdf_docs = st.file_uploader("Upload your PDF Files", accept_multiple_files=True, type=["pdf"])
        if st.button("Process PDFs") and pdf_docs:
            with st.spinner("Processing PDFs..."):
                raw_text = get_pdf_text(pdf_docs)
                st.write(f"Word Count: {word_count(raw_text)}")
                st.text_area("Extracted Text from PDFs", value=raw_text, height=600)

        # Image Upload and Processing
        uploaded_image = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
        if uploaded_image:
            st.image(uploaded_image, caption='Uploaded Image', use_column_width=True)
            image_text = process_image(uploaded_image)
            if image_text:
                st.text_area("Extracted Text from Image", value=image_text, height=300)
                st.write(f"Word Count: {word_count(image_text)}")

    elif option == 'QA from Text':
        st.title('Question Answering from Text')

        # PDF Upload and QA
        pdf_docs = st.file_uploader("Upload your PDF Files for QA", accept_multiple_files=True, type=["pdf"])
        if pdf_docs:
            with st.spinner("Processing PDFs for QA..."):
                pdf_text = get_pdf_text(pdf_docs)
                st.write(f"Word Count: {word_count(pdf_text)}")
                st.text_area("Text from PDFs", value=pdf_text, height=600)

        # QA Input
        question = st.text_input("Ask a question based on the provided text")
        if st.button("Get Answer") and question and pdf_text:
            answer = llm_pipeline(f"{pdf_text} {question}", max_length=200, task='qa')
            if answer:
                st.success(answer)

    elif option == 'Grammar Check':
        st.title('Grammar Check')

        # Grammar Check Input
        grammar_text = st.text_area("Input Text for Grammar Check", height=150)
        if st.button("Check Grammar"):
            if grammar_text:
                # For grammar checking, use your grammar checking service or API here
                # This is a placeholder response
                st.write("Grammar check results will be highlited above with blue underline")
                st.write("If no errors , there are no hilighted texts")

if __name__ == "__main__":
    main()
