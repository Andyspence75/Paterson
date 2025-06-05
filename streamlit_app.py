
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
import os

from redactor import redact_text

st.title("Housing Disrepair QA System")

uploaded_file = st.file_uploader("Upload a PDF Survey Report", type=["pdf"])
question = st.text_input("Ask about housing conditions:")

if uploaded_file:
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())

    loader = PyPDFLoader("temp.pdf")
    docs = loader.load()

    # Redact personal info
    for doc in docs:
        doc.page_content = redact_text(doc.page_content)

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    splits = splitter.split_documents(docs)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectordb = FAISS.from_documents(splits, embeddings)
    retriever = vectordb.as_retriever()

    llm = ChatOpenAI(model_name="gpt-3.5-turbo")
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=True)

    if question:
        response = qa_chain({"query": question})
        st.write(response["result"])

        with st.expander("Sources"):
            for doc in response["source_documents"]:
                st.markdown(doc.page_content[:300] + "...")
