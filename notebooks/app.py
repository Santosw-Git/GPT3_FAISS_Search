import streamlit as st
import os
import pickle
import time
from langchain import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQAWithSourcesChain
from langchain_community.document_loaders import UnstructuredURLLoader

from dotenv import load_dotenv

load_dotenv()


st.title("Hello, Streamlit!")
st.sidebar.title("This is a Streamlit app running in VS Code.")

main_placeholder = st.empty()
urls =[]
for i in range(1):
    url=st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)

process_url_clicked = st.sidebar.button("Process URLs")
file_path = "faiss_store_openAI.pkl"
if process_url_clicked:
    
    # Load Data
    
    loader=UnstructuredURLLoader(urls=urls)
    main_placeholder.text("Data Loading Started....")
    data=loader.load()
    
    # Split Data
    text_splitter = RecursiveCharacterTextSplitter(
    separators=[
        "\n\n",
        "\n",
        " ",
        ".",
    ],
    chunk_size=1000,
    chunk_overlap=0,
)
    
    docs=text_splitter.split_documents(data)
    main_placeholder.text("Text Splitter Started....")
    
    # Create Embeddings and Save it in FAISS Index
    embeddings=OpenAIEmbeddings()
    vector_store_openai = FAISS.from_documents(docs , embeddings)
    main_placeholder.text("Embeddings vector started Buliding....")
    
    # save the FAISS index to a pickle file
    
    with open(file_path,"wb") as f:
        pickle.dump(vector_store_openai , f)
        
    
    
    
    
    