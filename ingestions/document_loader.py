
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_community.document_loaders import PyMuPDFLoader 
# import os
# from langchain_google_genai import GoogleGenerativeAIEmbeddings 
# from dotenv import load_dotenv 
# from langchain_chroma import Chroma
# import getpass

# load_dotenv()

# if not os.environ.get("GOOGLE_API_KEY"):
#   os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter API key for Google Gemini: ")


# BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# docs = (
#     PyMuPDFLoader(os.path.join(BASE_DIR, "../documents/Company_Policies.pdf")).load()
#     + PyMuPDFLoader(os.path.join(BASE_DIR, "../documents/Company_Profile.pdf")).load()
#     + PyMuPDFLoader(os.path.join(BASE_DIR, "../documents/Product_and_Pricing.pdf")).load()
# )

# splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
# chunks = splitter.split_documents(docs) 

# embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001") 

# vector_store = Chroma(
#     collection_name="company_docs",
#     embedding_function=embeddings,
#     persist_directory=os.path.join(BASE_DIR, "../chroma_langchain_db"),
# )

# # Add documents to Chroma
# vector_store.add_documents(chunks)
# retriver = vector_store.as_retriever


# docs = vector_store.similarity_search("What is company policy?", k=3)

# for d in docs:
#     print("\nRESULT:")
#     print(d.page_content[:300])
#     print("META:", d.metadata)

# print("Total docs:", len(docs))
# print("Total chunks:", len(chunks))
# print(chunks[0].page_content[:300]) 

# ingest_vector_db.py

import os
import getpass
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma


def ingest_company_documents():
    load_dotenv()

    # API Key Setup
    if not os.environ.get("GOOGLE_API_KEY"):
        os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter Google API Key: ")

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    # Load PDFs
    docs = (
        PyMuPDFLoader(os.path.join(BASE_DIR, "../documents/Company_Policies.pdf")).load()
        + PyMuPDFLoader(os.path.join(BASE_DIR, "../documents/Company_Profile.pdf")).load()
        + PyMuPDFLoader(os.path.join(BASE_DIR, "../documents/Product_and_Pricing.pdf")).load()
    )

    # Split text
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)

    # Embeddings
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

    # Vector DB
    vector_store = Chroma(
        collection_name="company_docs",
        embedding_function=embeddings,
        persist_directory=os.path.join(BASE_DIR, "../chroma_langchain_db"),
    )

    # Ingest only once
    if vector_store._collection.count() == 0:
        vector_store.add_documents(chunks)
        print(" Documents ingested successfully!")
    else:
        print(" Data already exists. Skipping ingestion.")

    return vector_store