import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration, pipeline
import torch
import base64
import os
from PyPDF2 import PdfReader
import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate

# Configuration
local_model_path = "D:\MINI PROJECT\Text_Summarization\La-Mini-Flan-T5-248M"
GOOGLE_API_KEY = "AIzaSyADMgTC-cHl_FutTDuv43Voy5i-1GxDkYs"
genai.configure(api_key=GOOGLE_API_KEY)

# Load models
try:
    tokenizer = T5Tokenizer.from_pretrained(local_model_path)
    base_model = T5ForConditionalGeneration.from_pretrained(local_model_path, device_map='auto', torch_dtype=torch.float32)
except ImportError as e:
    st.error(f"Failed to load model or tokenizer: {e}")
    st.stop()

# Summarization functions
def file_preprocessing(file):
    loader = PyPDFLoader(file)
    pages = loader.load_and_split()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)
    texts = text_splitter.split_documents(pages)
    final_texts = "".join(text.page_content for text in texts)
    return final_texts

def llm_pipeline(filepath):
    pipe_sum = pipeline(
        "summarization",
        model=base_model,
        tokenizer=tokenizer,
        max_length=100,
        min_length=50
    )
    input_text = file_preprocessing(filepath)
    result = pipe_sum(input_text)
    return result[0]['summary_text']

@st.cache_data
def displayPDF(file):
    with open(file, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600" type="application/pdf"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)

# QA functions
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details. If the answer is not in
    the provided context, just say, "answer is not available in the context", don't provide the wrong answer.\n\n
    Context:\n {context}\n
    Question: \n{question}\n
    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3, google_api_key=GOOGLE_API_KEY)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    st.write("Reply: ", response["output_text"])

# Main function
def main():
    st.set_page_config(layout='wide', page_title="Multi-Function PDF Tool")

    # Sidebar for choosing functionality
    option = st.sidebar.selectbox(
        'Choose an action:',
        ['Text Summarization', 'Chat with PDF']
    )

    if option == 'Text Summarization':
        st.title('Text Summarization using LLM')
        uploaded_file = st.file_uploader('Upload your PDF File for Summarization', type=['pdf'])

        if uploaded_file is not None:
            if st.button('Summarize'):
                col1, col2 = st.columns(2)
                
                # Create the directory if it does not exist
                data_dir = "data"
                os.makedirs(data_dir, exist_ok=True)
                
                filepath = os.path.join(data_dir, uploaded_file.name)
                with open(filepath, 'wb') as temp_file:
                    temp_file.write(uploaded_file.read())

                with col1:
                    st.info('Uploaded PDF File')
                    displayPDF(filepath)

                with col2:
                    st.info('Summarization is here')
                    summary = llm_pipeline(filepath)
                    st.success(summary)

    elif option == 'Chat with PDF':
        st.header("Chat with PDF")

        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        user_question = st.text_input("Ask a Question from the PDF Files")

        if st.button("Submit & Process") and pdf_docs:
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done")

        if user_question:
            user_input(user_question)

if __name__ == "__main__":
    main()
