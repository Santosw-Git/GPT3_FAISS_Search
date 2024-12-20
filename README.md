# GPT3_FAISS_Search


This project leverages OpenAI's GPT-3 model and FAISS to build a search engine that can retrieve relevant information from large datasets. The process involves:

1. **Data Collection**: Fetches data from multiple URLs.
2. **Text Chunking**: Breaks the content into smaller chunks for easier processing.
3. **Embedding Creation**: Embeds each chunk and stores them in FAISS for efficient similarity search.
4. **Question Answering**: When a query is made, it generates an embedding for the question, searches the FAISS index for the most relevant chunks, and then uses GPT-3 to generate answers based on the retrieved text.

## Features
- Efficiently processes large datasets using FAISS.
- Provides a seamless interface for querying and retrieving information.
- Uses GPT-3 for answering questions from the most relevant text chunks.

## Requirements
- Python 3.x
- OpenAI API key
- FAISS library
- langchain (for efficient chaining of models)

## Installation
1. Clone the repository.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
