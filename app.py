from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma, Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
import pinecone
import os

OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY', 'your-openai-api-key')
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY', 'your-pinecone-api-key')
PINECONE_API_ENV = os.environ.get('PINECONE_API_ENV', 'your-pinecone-api-env')

# 1. Load the document
file_path = input("Enter the file path: ")
loader = PyPDFLoader(file_path)
data = loader.load()

# 2. Chunk the document
text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=0)
texts = text_splitter.split_documents(data)

# 3. Create Embeddings Vector Store
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
pinecone.init(
    api_key=PINECONE_API_KEY,  
    environment=PINECONE_API_ENV  
)
index_name = "askpdf"
vector = Pinecone.from_texts([t.page_content for t in texts], embeddings, index_name=index_name)

# 4. Query the pdf
llm = OpenAI(temperature=0, openai_api_key=OPENAI_API_KEY)
chain = load_qa_chain(llm, chain_type="stuff")
while True:
    query = input("Enter the query (Press 'q' to stop): ")
    if query == 'q':
        break
    docs = vector.similarity_search(query)
    res=chain.run(input_documents=docs, question=query)
    print("\n"+res+"\n")
