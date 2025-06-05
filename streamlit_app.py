
import streamlit as st
import os
import tempfile
import json
import sqlite3
import pandas as pd
from datetime import datetime
from langchain_community.document_loaders import PyPDFLoader, UnstructuredPowerPointLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain.docstore.document import Document

# --- Database Setup ---
def init_db():
    conn = sqlite3.connect("qa_app.db")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            username TEXT PRIMARY KEY,
            is_admin INTEGER DEFAULT 0
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS qa_log (
            username TEXT,
            question TEXT,
            answer TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

def log_qa(username, question, answer):
    conn = sqlite3.connect("qa_app.db")
    cursor = conn.cursor()
    cursor.execute("INSERT INTO qa_log (username, question, answer) VALUES (?, ?, ?)",
                   (username, question, answer))
    conn.commit()
    conn.close()

# --- User Authentication ---
init_db()
username = st.text_input("Enter your username to begin:", key="user_input")
is_admin = False

if username:
    conn = sqlite3.connect("qa_app.db")
    cursor = conn.cursor()
    cursor.execute("SELECT is_admin FROM users WHERE username = ?", (username,))
    result = cursor.fetchone()
    if result is None:
        cursor.execute("INSERT INTO users (username) VALUES (?)", (username,))
        conn.commit()
    else:
        is_admin = result[0] == 1
    conn.close()
    st.success(f"Welcome {username} {'(Admin)' if is_admin else ''}")

    # --- Upload Section ---
    st.header("Upload Training Data")
    uploaded_files = st.file_uploader("Upload PDF or PPTX files", type=["pdf", "pptx"], accept_multiple_files=True)
    qa_file = st.file_uploader("Optional: Upload Q&A pairs in JSON or CSV", type=["json", "csv"])

    persist_dir = "faiss_index"
    os.makedirs(persist_dir, exist_ok=True)
    docs = []

    if uploaded_files:
        st.info("Processing uploaded documents...")
        for file in uploaded_files:
            with tempfile.NamedTemporaryFile(delete=False, suffix="." + file.name.split('.')[-1]) as tmp_file:
                tmp_file.write(file.read())
                if file.name.endswith(".pdf"):
                    loader = PyPDFLoader(tmp_file.name)
                elif file.name.endswith(".pptx"):
                    loader = UnstructuredPowerPointLoader(tmp_file.name)
                docs.extend(loader.load())

    if qa_file:
        st.info("Loading Q&A pairs...")
        if qa_file.name.endswith(".json"):
            qa_data = json.load(qa_file)
        else:
            qa_data = pd.read_csv(qa_file).to_dict(orient="records")
        for pair in qa_data:
            question = pair.get("question", "")
            answer = pair.get("answer", "")
            docs.append(Document(page_content=f"Q: {question}\nA: {answer}", metadata={"source": "qa_upload"}))

    if docs:
        st.success(f"Ingested {len(docs)} items for training.")
        splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
        splits = splitter.split_documents(docs)

        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vectordb = FAISS.from_documents(splits, embedding=embeddings)
        vectordb.save_local(persist_dir)
    else:
        try:
            vectordb = FAISS.load_local(persist_dir, embeddings=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2"))
            st.info("Loaded previously saved data.")
        except:
            vectordb = None
            st.warning("No data found. Please upload documents.")

    # --- Query Interface ---
    st.header("Ask a Question")
    general_query = st.text_input("You can ask a general question below:")

    if vectordb:
        retriever = vectordb.as_retriever()
        llm = ChatOpenAI(model_name="gpt-3.5-turbo")
        qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=True)

        if general_query:
            response = qa_chain({"query": general_query})
            st.markdown(f"**Answer:** {response['result']}")
            with st.expander("See source text"):
                for doc in response['source_documents']:
                    st.markdown(doc.page_content[:500] + "...")
            log_qa(username, general_query, response["result"])

    # --- Admin Panel ---
    if is_admin:
        st.header("Admin Panel")
        if st.button("Show Q&A Logs"):
            conn = sqlite3.connect("qa_app.db")
            df = pd.read_sql_query("SELECT * FROM qa_log ORDER BY timestamp DESC", conn)
            st.dataframe(df)
            conn.close()
