
# streamlit_app.py

import os
import openai
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.vectorstores import Qdrant
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredPDFLoader
from langchain.chains import RetrievalQA
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
import streamlit as st

# Set OpenAI API Key from Streamlit secrets
openai.api_key = st.secrets["OPENAI_API_KEY"]

st.title("Housing Disrepair QA System")
uploaded_file = st.file_uploader("Upload a PDF Survey Report", type="pdf")

if uploaded_file:
    st.info("Processing document...")
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())

    loader = UnstructuredPDFLoader("temp.pdf")
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    splits = splitter.split_documents(docs)

    embeddings = OpenAIEmbeddings()
    client = QdrantClient(path="./qdrant_local")

    client.recreate_collection(
        collection_name="housing_reports",
        vectors_config=VectorParams(size=1536, distance=Distance.COSINE)
    )

    vectordb = Qdrant.from_documents(
        documents=splits,
        embedding=embeddings,
        collection_name="housing_reports",
        client=client,
    )

    retriever = vectordb.as_retriever()
    llm = ChatOpenAI(model_name="gpt-4-turbo")
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )

    question = st.text_input("Ask about the report or housing standards:")
    if question:
        response = qa_chain({"query": question})
        st.write(response['result'])

        with st.expander("See referenced sections"):
            for doc in response['source_documents']:
                st.markdown(doc.page_content[:300] + "...")
