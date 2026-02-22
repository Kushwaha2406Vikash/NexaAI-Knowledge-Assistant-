
# retriever_loader.py

import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
import getpass 

load_dotenv()
if not os.environ.get("GOOGLE_API_KEY"):
        os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter Google API Key: ")


        
def get_company_retriever():
    

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

    vector_store = Chroma(
        collection_name="company_docs",
        embedding_function=embeddings,
        persist_directory=os.path.join(BASE_DIR, "../chroma_langchain_db"),
    )

    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    return retriever