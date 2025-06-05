
# streamlit_app.py

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
import streamlit as st
import openai

st.title("Housing Disrepair QA System")

# API Key Input
openai_api_key = st.text_input("Enter your OpenAI API Key:", type="password")
if not openai_api_key:
    st.warning("Please enter your OpenAI API key to proceed.")
    st.stop()
openai.api_key = openai_api_key

uploaded_file = st.file_uploader("Upload a PDF Survey Report", type="pdf")

# Initialize session history
if "history" not in st.session_state:
    st.session_state.history = []

# Key section checks
required_sections = [
    "mold", "damp", "heating", "insulation", "ventilation",
    "electrical safety", "gas safety", "structure", "roof", "windows"
]

def validate_sections(text):
    missing = []
    lower_text = text.lower()
    for item in required_sections:
        if item not in lower_text:
            missing.append(item)
    return missing

if uploaded_file:
    st.info("Processing document...")
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())

    loader = PyPDFLoader("temp.pdf")
    docs = loader.load()

    if not docs:
        st.error("No text could be extracted from the PDF.")
        st.stop()

    full_text = "\n".join([doc.page_content for doc in docs])
    missing = validate_sections(full_text)
    if missing:
        st.warning(f"The following key sections were NOT found: {', '.join(missing)}")
    else:
        st.success("All key sections are present in the report!")

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    splits = splitter.split_documents(docs)

    if not splits:
        st.error("The document was parsed but no usable chunks were found.")
        st.stop()

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectordb = FAISS.from_documents(splits, embeddings)
    retriever = vectordb.as_retriever()

    llm = ChatOpenAI(model_name="gpt-3.5-turbo")
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )

    question = st.text_input("Ask about the report or housing standards:")
    if question:
        st.write(f"Running query: {question}")
        response = qa_chain({"question": question})
        st.session_state.history.append({"q": question, "a": response['result']})
        st.write(response['result'])

        with st.expander("See referenced sections"):
            for doc in response['source_documents']:
                st.markdown(doc.page_content[:300] + "...")

if st.session_state.history:
    with st.expander("Query History"):
        for entry in reversed(st.session_state.history):
            st.markdown(f"**Q:** {entry['q']}\n**A:** {entry['a']}")
