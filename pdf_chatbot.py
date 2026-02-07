import os
import shutil
import streamlit as st

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import uuid
# ================= CONFIG =================
CHROMA_DB_DIR = "./chroma_db"
UPLOAD_DIR = "./uploaded_pdfs"

MODEL_PATH = "./sentence-transformers_all-MiniLM-L6-v2"
OLLAMA_MODEL_NAME = "mistral"

os.makedirs(UPLOAD_DIR, exist_ok=True)

# ================= STREAMLIT UI =================
st.set_page_config(page_title="PDF Chatbot", layout="wide")
st.title("📚 Chat with Your PDF (Offline)")

# ================= SESSION STATE =================
if "llm" not in st.session_state:
    st.session_state.llm = Ollama(
        model=OLLAMA_MODEL_NAME,
        temperature=0
    )

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None

if "current_pdf_hash" not in st.session_state:
    st.session_state.current_pdf_hash = None


# ================= EMBEDDINGS =================
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name=MODEL_PATH,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

embedding_model = load_embeddings()

import hashlib

def get_pdf_hash(file_bytes):
    return hashlib.md5(file_bytes).hexdigest()

# ================= PDF PROCESSING =================
@st.cache_resource
def build_vectorstore(file_bytes, pdf_hash):
    temp_path = f"temp_{pdf_hash}.pdf"
    with open(temp_path, "wb") as f:
        f.write(file_bytes)

    loader = PyPDFLoader(temp_path)

    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(docs)

    db_path = f"./chroma_db_{pdf_hash}"

    if os.path.exists(db_path):
        return Chroma(
            persist_directory=db_path,
            embedding_function=embedding_model
        )

    vs = Chroma.from_documents(
        chunks,
        embedding_model,
        persist_directory=db_path
    )
    vs.persist()
    return vs

PDF_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are a helpful assistant.
Answer the question using ONLY the information from the context.
If the answer is not present, say:
"I could not find this information in the uploaded PDF."

Context:
{context}

Question:
{question}

Answer:
"""
)
SUMMARY_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are a helpful assistant.
Summarize the document based on the context below.
You MAY combine and infer information across sections.
Answer in bullet points.

Context:
{context}

Question:
{question}

Answer:
"""
)
def is_summary_question(q):
    if not q:   # handles None and empty string
        return False

    keywords = ["summary", "summarize", "points", "overview", "about the pdf"]
    return any(k in q.lower() for k in keywords)

# ================= PDF UPLOADER =================
uploaded_pdf = st.file_uploader("📄 Upload a PDF", type=["pdf"])


if uploaded_pdf:
    pdf_bytes = uploaded_pdf.read()
    pdf_hash = get_pdf_hash(pdf_bytes)

    if st.session_state.current_pdf_hash != pdf_hash:
        vectorstore = build_vectorstore(pdf_bytes, pdf_hash)

        # 👇 create retriever ONCE per PDF
        st.session_state.retriever = vectorstore.as_retriever(
            search_kwargs={"k": 8}
        )

        # 👇 create QA chain ONCE per PDF
        st.session_state.qa_chain = RetrievalQA.from_chain_type(
            llm=st.session_state.llm,
            retriever=st.session_state.retriever,
            chain_type="stuff",
            chain_type_kwargs={"prompt": PDF_PROMPT},
            return_source_documents=False
        )

        st.session_state.chat_history = []
        st.session_state.current_pdf_hash = pdf_hash


# ================= CHAT UI =================
query = st.chat_input("Ask a question from the PDF")

if not query:
    st.stop()

if st.session_state.qa_chain is None:
    st.warning("⚠️ Please upload a PDF first.")
    st.stop()

# Decide which chain to use
if is_summary_question(query):
    qa_chain = RetrievalQA.from_chain_type(
        llm=st.session_state.llm,
        retriever=st.session_state.retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": SUMMARY_PROMPT},
        return_source_documents=False
    )
else:
    qa_chain = st.session_state.qa_chain

# Run the chosen chain
st.session_state.chat_history.append(
    {"role": "user", "content": query}
)

response = qa_chain.invoke({"query": query})

st.session_state.chat_history.append(
    {"role": "assistant", "content": response["result"]}
)



if st.button("🔄 Reset PDF"):
    st.session_state.clear()
    st.rerun()

# ================= DISPLAY CHAT =================
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
