import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration, pipeline
import torch
import base64
import os

# Local path to the model
local_model_path = "D:\MINI PROJECT\Mini-Project\Ai Anytime\LaMini-LM  Document Summarization\TextSummarization\La-Mini-Flan-T5-248M"
# Model and Tokenizer
try:
    tokenizer = T5Tokenizer.from_pretrained(local_model_path)
    base_model = T5ForConditionalGeneration.from_pretrained(local_model_path, device_map='auto', torch_dtype=torch.float32)
except ImportError as e:
    st.error(f"Failed to load model or tokenizer: {e}")
    st.stop()

# File loader and Preprocessing
def file_preprocessing(file):
    loader = PyPDFLoader(file)
    pages = loader.load_and_split()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)
    texts = text_splitter.split_documents(pages)
    final_texts = ""
    for text in texts:
        final_texts += text.page_content
    return final_texts

# Language Model Pipeline
def llm_pipeline(filepath):
    pipe_sum = pipeline(
        "summarization",
        model=base_model,
        tokenizer=tokenizer,
        max_length=500,
        min_length=50
    )

    input_text = file_preprocessing(filepath)
    result = pipe_sum(input_text)
    result = result[0]['summary_text']
    return result

@st.cache_data
def displayPDF(file):
    # Open file from file path
    with open(file, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')

    # Embedding PDF in HTML
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600" type="application/pdf"></iframe>'

    # Displaying file
    st.markdown(pdf_display, unsafe_allow_html=True)

# Streamlit Code
st.set_page_config(layout='wide', page_title="Summarization Application")

def main():
    st.title('Text Summarization using LLM')
    uploaded_file = st.file_uploader('Upload your PDF File', type=['pdf'])

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

if __name__ == '__main__':
    main()





































# import streamlit as st
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.document_loaders import PyPDFLoader
# from langchain.chains.summarize import load_summarize_chain
# from transformers import T5Tokenizer, T5ForConditionalGeneration, pipeline
# import torch
# import base64

# # Model and Tokenizer
# checkpoint = "La-Mini-Flan-T5-248M"   #"MBZUAI/LaMini-Flan-T5-248M"
# tokenizer = T5Tokenizer.from_pretrained(checkpoint)
# base_model = T5ForConditionalGeneration.from_pretrained(checkpoint, device_map='auto', torch_dtype=torch.float32)

# # File loader and Preprocessing
# def file_preprocessing(file):
#     loader = PyPDFLoader(file)
#     pages = loader.load_and_split()
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)
#     texts = text_splitter.split_documents(pages)
#     final_texts = ""
#     for text in texts:
#         final_texts += text.page_content
#     return final_texts

# # Language Model Pipeline
# def llm_pipeline(filepath):
#     pipe_sum = pipeline(
#         "summarization",
#         model=base_model,
#         tokenizer=tokenizer,
#         max_length=500,
#         min_length=50
#     )

#     input_text = file_preprocessing(filepath)
#     result = pipe_sum(input_text)
#     result = result[0]['summary_text']
#     return result

# @st.cache_data
# def displayPDF(file):
#     # Open file from file path
#     with open(file, "rb") as f:
#         base64_pdf = base64.b64encode(f.read()).decode('utf-8')

#     # Embedding PDF in HTML
#     pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600" type="application/pdf"></iframe>'

#     # Displaying file
#     st.markdown(pdf_display, unsafe_allow_html=True)

# # Streamlit Code
# st.set_page_config(layout='wide', page_title="Summarization Application")

# def main():
#     st.title('Text Summarization using LLM')
#     uploaded_file = st.file_uploader('Upload your PDF File', type=['pdf'])

#     if uploaded_file is not None:
#         if st.button('Summarize'):
#             col1, col2 = st.columns(2)
#             filepath = "data/" + uploaded_file.name
#             with open(filepath, 'wb') as temp_file:
#                 temp_file.write(uploaded_file.read())

#             with col1:
#                 st.info('Uploaded PDF File')
#                 displayPDF(filepath)

#             with col2:
#                 st.info('Summarization is here')
#                 summary = llm_pipeline(filepath)
#                 st.success(summary)

# if __name__ == '__main__':
#     main()























