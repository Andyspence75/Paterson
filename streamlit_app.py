
import os
import streamlit as st
from langchain_community.document_loaders import (
    PyPDFLoader,
    UnstructuredFileLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredPowerPointLoader,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Qdrant
from qdrant_client import QdrantClient
from redactor import redact_text

st.set_page_config(page_title="Housing Disrepair QA", layout="wide")
st.title("Housing Disrepair QA System (Qdrant Cloud)")

QDRANT_COLLECTION = "housing_rag_collection"

# Qdrant Cloud credentials (set these as Streamlit secrets!)
QDRANT_URL = st.secrets["QDRANT_URL"]
QDRANT_API_KEY = st.secrets["QDRANT_API_KEY"]

qdrant_client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
)

def get_loader_for_file(filename):
    ext = filename.lower().split(".")[-1]
    if ext == "pdf":
        return PyPDFLoader(filename)
    elif ext == "txt":
        return UnstructuredFileLoader(filename)
    elif ext in ["doc", "docx"]:
        return UnstructuredWordDocumentLoader(filename)
    elif ext in ["ppt", "pptx"]:
        return UnstructuredPowerPointLoader(filename)
    else:
        return None

def get_vectorstore():
    try:
        return Qdrant(
            client=qdrant_client,
            collection_name=QDRANT_COLLECTION,
            embeddings=OpenAIEmbeddings(),
        )
    except Exception as e:
        st.warning(f"Could not connect to collection: {e}")
        return None

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = get_vectorstore()

with st.sidebar:
    st.header("Document Upload")
    uploaded_files = st.file_uploader(
        "Upload PDF, DOC, PPT, or TXT files",
        type=["pdf", "txt", "doc", "docx", "ppt", "pptx"],
        accept_multiple_files=True
    )

    if uploaded_files:
        all_docs = []
        for uploaded_file in uploaded_files:
            with open(uploaded_file.name, "wb") as f:
                f.write(uploaded_file.read())

            loader = get_loader_for_file(uploaded_file.name)
            if loader is not None:
                docs = loader.load()
                docs = [doc for doc in docs if doc.page_content.strip()]
                docs = [doc.__class__(page_content=redact_text(doc.page_content)) for doc in docs]
                all_docs.extend(docs)
            else:
                st.warning(f"Unsupported file type: {uploaded_file.name}")

        if all_docs:
            splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            splits = splitter.split_documents(all_docs)
            embeddings = OpenAIEmbeddings()
            vectorstore = Qdrant.from_documents(
                splits,
                embeddings,
                client=qdrant_client,
                collection_name=QDRANT_COLLECTION
            )
            st.session_state.vectorstore = vectorstore
            st.success("Documents processed and indexed (Qdrant Cloud, persistent!).")

if st.session_state.vectorstore:
    llm = ChatOpenAI(model_name="gpt-3.5-turbo")
    retriever = st.session_state.vectorstore.as_retriever()
    qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=True)

    query = st.text_input("Ask a question about the documents:")
    if query:
        result = qa({"query": query})
        st.write("**Answer:**", result["result"])
        with st.expander("See Sources"):
            for doc in result["source_documents"]:
                st.markdown(doc.page_content[:300] + "...")
