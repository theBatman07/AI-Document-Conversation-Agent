# RAG PDF QA Application

## Overview

This application is a Retrieval-Augmented Generation (RAG) system that allows users to upload PDF documents, ask questions about their content, and receive AI-generated answers. It uses Streamlit for the user interface, LangChain for the RAG implementation, and Claude (via the Anthropic API) for generating responses.

## Features

- Upload multiple PDF files
- Extract and process text from PDFs
- Create a vector store of document content
- Ask questions about the uploaded documents
- Receive AI-generated answers using Claude

## Tech Stack

- Python
- Langchain
- Chromadb
- Streamlit
- Claude

## Setup

1. Clone the repository:
  
  ```bash
  git clone https://github.com/theBatman07/AI-Document-Conversation-Agent.git
  ```
  
2. Install the required packages:
  
  ```bash
  pip install -r requirements.txt
  ```
  
3. Create a `.env` file in the project root and add your Anthropic API key:
  
  ```bash
  ANTHROPIC_API_KEY=your_anthropic_api_key_here
  ```
  

## Usage

1. Run the Streamlit app:
  
  ```bash
  streamlit run app.py
  ```
  
2. Open your web browser and go to the URL provided by Streamlit (usually `http://localhost:8501`).
  
3. Use the file uploader to upload one or more PDF files.
  
4. Once the files are processed, type your question in the text input field.
  
5. The application will generate an answer based on the content of the uploaded PDFs.
  

## How It Works

1. The app extracts text from uploaded PDF files.
2. The text is split into smaller chunks and processed.
3. These chunks are embedded and stored in a vector database (ChromaDB).
4. When a question is asked, the most relevant chunks are retrieved.
5. Claude, an AI language model, generates an answer based on the retrieved chunks and the question.