
import streamlit as st
import sqlite3
import os
from langchain_community.document_loaders import PyPDFLoader, UnstructuredPowerPointLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI

# Database setup
conn = sqlite3.connect('users.db', check_same_thread=False)
c = conn.cursor()
c.execute('''CREATE TABLE IF NOT EXISTS users (username TEXT, role TEXT)''')
conn.commit()

# Auth setup
st.sidebar.header("Login")
username = st.sidebar.text_input("Username")
role = st.sidebar.selectbox("Role", ["user", "admin"])
if st.sidebar.button("Login"):
    c.execute("INSERT INTO users VALUES (?, ?)", (username, role))
    conn.commit()
    st.session_state['username'] = username
    st.session_state['role'] = role
    st.sidebar.success(f"Logged in as {username} ({role})")

if 'username' not in st.session_state:
    st.stop()

# Main interface
st.title("RAG Housing Support App")
general_query = st.text_input("Ask a general question (no upload required):")
if general_query:
    st.info("Querying general knowledge model...")
    llm = ChatOpenAI(model_name="gpt-3.5-turbo")
    response = llm.invoke(general_query)
    st.write(response)

uploaded_files = st.file_uploader("Upload PDFs or PPTX reports", type=["pdf", "pptx"], accept_multiple_files=True)
qa_json = st.file_uploader("Optional: Upload Q&A JSON", type=["json"])

if uploaded_files:
    st.info("Processing uploaded documents...")
    docs = []
    for file in uploaded_files:
        with open(file.name, "wb") as f:
            f.write(file.getbuffer())
        if file.name.endswith(".pdf"):
            loader = PyPDFLoader(file.name)
        else:
            loader = UnstructuredPowerPointLoader(file.name)
        docs.extend(loader.load())

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    splits = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectordb = FAISS.from_documents(splits, embeddings)

    retriever = vectordb.as_retriever()
    llm = ChatOpenAI(model_name="gpt-3.5-turbo")
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=True)

    question = st.text_input("Ask a question based on your uploads:")
    if question:
        result = qa_chain.invoke({"query": question})
        st.write(result['result'])
        with st.expander("Sources"):
            for doc in result['source_documents']:
                st.markdown(doc.page_content[:300] + "...")

# Admin tools
if st.session_state['role'] == "admin":
    st.sidebar.subheader("Admin Panel")
    if st.sidebar.button("View Users"):
        st.subheader("Registered Users")
        users = c.execute("SELECT * FROM users").fetchall()
        st.table(users)
