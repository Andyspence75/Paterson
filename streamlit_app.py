
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI

st.title("Housing Disrepair Q&A Assistant")

uploaded_file = st.file_uploader("Upload a PDF report or training doc", type="pdf")

if uploaded_file:
    st.info("Processing the document...")

    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())

    loader = PyPDFLoader("temp.pdf")
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    splits = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectordb = FAISS.from_documents(splits, embedding=embeddings)
    retriever = vectordb.as_retriever()

    llm = ChatOpenAI(model_name="gpt-3.5-turbo")
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )

    question = st.text_input("Ask a question about the uploaded document:")

    if question:
        response = qa_chain({"query": question})
        st.write("**Answer:**", response["result"])

        with st.expander("Show sources"):
            for doc in response["source_documents"]:
                st.markdown(doc.page_content[:500] + "...")
