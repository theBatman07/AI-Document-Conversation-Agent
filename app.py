import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
import os
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_anthropic import ChatAnthropic

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

        retriever = vectorstore.as_retriever(
        search_type="mmr", 
        search_kwargs={"k": 12},
        )

        llm = ChatAnthropic(model="claude-3-5-sonnet-20240620")

        prompt = ChatPromptTemplate.from_messages(
            [
                ("placeholder", "{chat_history}"),
                ("user", "{input}"),
                (
                    "user",
                    "Given the above conversation, generate a search query to look up to get information relevant to the conversation",
                ),
            ]
        )

        retriever_chain = create_history_aware_retriever(llm, retriever, prompt)

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "Answer the user's questions based on the below context:\n\n{context}",
                ),
                ("placeholder", "{chat_history}"),
                ("user", "{input}"),
            ]
        )
        document_chain = create_stuff_documents_chain(llm, prompt)

        qa = create_retrieval_chain(retriever_chain, document_chain)

        user_question = st.text_input("Ask a question about the uploaded PDFs:")

        if user_question:
            result = qa.invoke({"input": user_question})
            st.write("Answer:", result["answer"])

if __name__ == "__main__":
    main()