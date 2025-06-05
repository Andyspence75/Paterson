
import os
import sqlite3
import streamlit as st
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_openai import ChatOpenAI

DB_PATH = "vectorstore_index"
st.title("Housing Disrepair QA System (with Persistent Storage)")

# Initialize DB
if "db_conn" not in st.session_state:
    st.session_state.db_conn = sqlite3.connect("qa_log.db", check_same_thread=False)
    with st.session_state.db_conn:
        st.session_state.db_conn.execute('''
            CREATE TABLE IF NOT EXISTS qa_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                question TEXT,
                answer TEXT
            )
        ''')

# Load vector store or initialize
def load_vector_store():
    if os.path.exists(DB_PATH):
        return FAISS.load_local(DB_PATH, HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2"), allow_dangerous_deserialization=True)
    return None

# Save vector store
def save_vector_store(vstore):
    vstore.save_local(DB_PATH)

uploaded_file = st.file_uploader("Upload a PDF or TXT file", type=["pdf", "txt"])
qa_pairs_file = st.file_uploader("Optional: Upload Training Q&A Pairs (TXT)", type="txt")

vectordb = load_vector_store()
if uploaded_file:
    st.info("Processing document...")
    with open("temp_file", "wb") as f:
        f.write(uploaded_file.read())

    if uploaded_file.name.endswith(".pdf"):
        loader = PyPDFLoader("temp_file")
    else:
        loader = TextLoader("temp_file")

    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    splits = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    if vectordb:
        vectordb.add_documents(splits)
    else:
        vectordb = FAISS.from_documents(splits, embeddings)
    save_vector_store(vectordb)
    st.success("File processed and stored.")

if qa_pairs_file:
    text = qa_pairs_file.read().decode("utf-8")
    qas = []
    for block in text.strip().split("\n\n"):
        if block.startswith("Q:") and "A:" in block:
            q = block.split("A:")[0].replace("Q:", "").strip()
            a = block.split("A:")[1].strip()
            qas.append((q, a))
    if qas:
        qa_docs = [f"Q: {q}\nA: {a}" for q, a in qas]
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        qa_splits = splitter.create_documents(qa_docs)
        vectordb.add_documents(qa_splits)
        save_vector_store(vectordb)
        st.success(f"Loaded {len(qas)} training Q&A pairs.")

# User query
question = st.text_input("Ask a question:")
if question and vectordb:
    retriever = vectordb.as_retriever()
    llm = ChatOpenAI(model_name="gpt-3.5-turbo")
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=True)
    response = qa_chain({"query": question})
    st.write("**Answer:**", response["result"])
    with st.expander("Referenced context"):
        for doc in response["source_documents"]:
            st.markdown(doc.page_content[:300] + "...")
    with st.session_state.db_conn:
        st.session_state.db_conn.execute("INSERT INTO qa_log (question, answer) VALUES (?, ?)", (question, response["result"]))
elif question:
    st.warning("Please upload a document or training data first.")

# View past Q&A log
if st.checkbox("Show Q&A Log"):
    with st.session_state.db_conn:
        rows = st.session_state.db_conn.execute("SELECT question, answer FROM qa_log ORDER BY id DESC LIMIT 10").fetchall()
    for q, a in rows:
        st.markdown(f"**Q:** {q}\n**A:** {a}")
