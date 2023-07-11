import streamlit as st
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma, Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
import pinecone
import os
import io
import tempfile

OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY', '')
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY', '')
PINECONE_API_ENV = os.environ.get('PINECONE_API_ENV', '')

# Initialize Pinecone outside the caching decorator
pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_API_ENV)

def process_pdf(file):
    # Save the BytesIO object as a temporary file
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    temp_file.write(file.read())
    temp_file.close()

    # 1. Load the document
    loader = PyPDFLoader(temp_file.name)
    data = loader.load()

    # 2. Chunk the document
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=0)
    texts = text_splitter.split_documents(data)

    return texts

def main():
    st.title("AskPDF Web App")

    file = st.file_uploader("Upload a PDF file", type="pdf")
    
    if file is not None:
        texts = process_pdf(file)

        embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        vector = Pinecone.from_texts([t.page_content for t in texts], embeddings, index_name="askpdf")

        query = st.text_input("Enter the query:")
        if query:
            docs = vector.similarity_search(query)
            llm = OpenAI(temperature=0, openai_api_key=OPENAI_API_KEY)
            chain = load_qa_chain(llm, chain_type="stuff")
            res = chain.run(input_documents=docs, question=query)
            st.write(res)

if __name__ == '__main__':
    main()
