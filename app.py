import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_community.llms import Anthropic
from dotenv import load_dotenv
import os

load_dotenv()

def extract_text_from_pdf(pdf_file):
    pdf_reader = PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def process_text(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def create_vector_store(chunks):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma.from_texts(chunks, embeddings)
    return vectorstore

def main():
    st.title("PDF Question Answering System using Claude")

    if not os.getenv("ANTHROPIC_API_KEY"):
        st.error("Anthropic API key not found. Please set it in your .env file.")
        return

    uploaded_files = st.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True)

    if uploaded_files:
        all_text = ""
        for file in uploaded_files:
            all_text += extract_text_from_pdf(file)

        chunks = process_text(all_text)
        vectorstore = create_vector_store(chunks)

        claude = Anthropic(model="claude-2")
        qa_chain = RetrievalQA.from_chain_type(
            llm=claude,
            chain_type="stuff",
            retriever=vectorstore.as_retriever()
        )

        user_question = st.text_input("Ask a question about the uploaded PDFs:")
        if user_question:
            response = qa_chain.run(user_question)
            st.write("Answer:", response)

if __name__ == "__main__":
    main()