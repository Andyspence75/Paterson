from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
import streamlit as st
import os

st.title("Housing Disrepair QA System")

uploaded_file = st.file_uploader("Upload a PDF Survey Report", type="pdf")

question = st.text_input("Ask a question (you can ask even without uploading):")

docs = []
if uploaded_file:
    st.info("Processing document...")
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())
    loader = PyPDFLoader("temp.pdf")
    docs = loader.load()

if docs:
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    splits = splitter.split_documents(docs)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectordb = FAISS.from_documents(splits, embeddings)
    retriever = vectordb.as_retriever()
else:
    retriever = None

if question:
    st.info("Getting answer...")
    llm = ChatOpenAI(model_name="gpt-3.5-turbo")
    if retriever:
        qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=True)
        response = qa_chain({"query": question})
        st.write(response['result'])
        with st.expander("See referenced sections"):
            for doc in response['source_documents']:
                st.markdown(doc.page_content[:300] + "...")
    else:
        response = llm.invoke(question)
        st.write(response.content)
