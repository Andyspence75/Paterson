import os
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, UnstructuredWordDocumentLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI

st.set_page_config(page_title="Housing Disrepair QA", layout="wide")
st.title("📄 Housing Disrepair QA System")

persist_dir = "vectorstore"
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectordb = None

if os.path.exists(persist_dir):
    try:
        vectordb = FAISS.load_local(persist_dir, embeddings, allow_dangerous_deserialization=True)
        st.sidebar.success("🔄 Previous data loaded.")
    except:
        st.sidebar.warning("⚠️ Failed to load existing data.")

llm = ChatOpenAI(model_name="gpt-3.5-turbo")
retriever = vectordb.as_retriever() if vectordb else None
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=True) if retriever else None

uploaded_files = st.file_uploader("Upload documents (PDF, Word)", type=["pdf", "docx"], accept_multiple_files=True)

if uploaded_files:
    docs = []
    for uploaded_file in uploaded_files:
        with open(uploaded_file.name, "wb") as f:
            f.write(uploaded_file.read())

        if uploaded_file.name.endswith(".pdf"):
            loader = PyPDFLoader(uploaded_file.name)
        else:
            loader = UnstructuredWordDocumentLoader(uploaded_file.name)

        docs.extend(loader.load())

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    splits = splitter.split_documents(docs)

    if vectordb:
        vectordb.add_documents(splits)
    else:
        vectordb = FAISS.from_documents(splits, embeddings)

    vectordb.save_local(persist_dir)
    retriever = vectordb.as_retriever()
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=True)
    st.success("✅ Documents processed and stored.")

question = st.text_input("Ask a question about housing disrepair, legislation, or uploaded documents:")

if qa_chain and question:
    response = qa_chain({"query": question})
    st.markdown("**Answer:**")
    st.write(response["result"])

    with st.expander("See referenced document excerpts"):
        for doc in response["source_documents"]:
            st.markdown(doc.page_content[:300] + "...")