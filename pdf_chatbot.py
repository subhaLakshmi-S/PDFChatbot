import os
import shutil
import streamlit as st
from sentence_transformers import SentenceTransformer
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import uuid
import hashlib

# ================= CONFIG =================
CHROMA_DB_DIR = "./chroma_db"   # (kept for reference; we’ll suffix with hash per-PDF)
UPLOAD_DIR = "./uploaded_pdfs"

MODEL_PATH = "./sentence-transformers_all-MiniLM-L6-v2"
OLLAMA_MODEL_NAME = "phi"

os.makedirs(UPLOAD_DIR, exist_ok=True)

# ================= STREAMLIT UI =================
st.set_page_config(page_title="PDF Chatbot", layout="wide")
st.title("📚 Chat with Your PDF")

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
    # CPU inference; set device='cpu' (you already did)
    return SentenceTransformer(MODEL_PATH, device='cpu')

_embedding_model = load_embeddings()

class CustomEmbedding:
    def embed_documents(self, texts):
        return _embedding_model.encode(texts, normalize_embeddings=True).tolist()
    def embed_query(self, text):
        return _embedding_model.encode(text, normalize_embeddings=True).tolist()

def get_pdf_hash(file_bytes: bytes) -> str:
    return hashlib.md5(file_bytes).hexdigest()

# ================= PROMPTS =================
PDF_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""
    You are an AI assistant that MUST answer ONLY using the information strictly found in the provided PDF context.
    
    ⚠️ RULES (Follow them exactly):
    - Use ONLY the text inside the "Context" section.
    - Do NOT use general knowledge.
    - Do NOT guess or infer beyond the PDF.
    - Do NOT mix information from other PDFs or previous chats.
    - If the answer is not explicitly present in the context, reply exactly:
      "I could not find this information in the uploaded PDF."
    
    When you give an answer:
    - Prefer quoting short phrases from the context verbatim when possible.
    - Keep the answer concise and factual.
    
    Context (strict source of truth):
    {context}
    
    User Question:
    {question}
    
    Your Answer (ONLY from the context):
    """
    )


SUMMARY_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are a helpful assistant.
Summarize the document based on the uploaded pdf only.
You MAY combine and infer information across sections.
Answer in bullet points.

Context:
{context}

Question:
{question}

Answer:
"""
)

def is_summary_question(q: str) -> bool:
    if not q:
        return False
    keywords = ["summary", "summarize", "points", "overview", "about the pdf"]
    ql = q.lower()
    return any(k in ql for k in keywords)

# ================= VECTOR STORE (cached by hash) =================
@st.cache_resource
def build_vectorstore_from_hash(pdf_hash: str, pdf_bytes: bytes):
    """
    Build (or load) a Chroma vector store for a given PDF hash.
    This function is intentionally cached by the hash + bytes (for correctness),
    but the heavy part (Chroma persist) is keyed by db_path existence.
    """
    temp_path = f"temp_{pdf_hash}.pdf"
    with open(temp_path, "wb") as f:
        f.write(pdf_bytes)

    try:
        loader = PyPDFLoader(temp_path)
        docs = loader.load()
    finally:
        # ensure we clean the temp file
        try:
            os.remove(temp_path)
        except OSError:
            pass

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(docs)

    db_path = f"./chroma_db_{pdf_hash}"

    if os.path.exists(db_path):
        return Chroma(
            persist_directory=db_path,
            embedding_function=CustomEmbedding()
        )

    vs = Chroma.from_documents(
        documents=chunks,
        embedding=CustomEmbedding(),
        persist_directory=db_path
    )
    vs.persist()
    return vs

# ================= PDF UPLOADER =================
uploaded_pdf = st.file_uploader("📄 Upload a PDF", type=["pdf"])

if uploaded_pdf:
    pdf_bytes = uploaded_pdf.read()
    pdf_hash = get_pdf_hash(pdf_bytes)

    if st.session_state.current_pdf_hash != pdf_hash:
        vectorstore = build_vectorstore_from_hash(pdf_hash, pdf_bytes)

        # retriever ONCE per PDF
        st.session_state.retriever = vectorstore.as_retriever(
            search_kwargs={"k": 8}
        )

        # QA chain ONCE per PDF
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

# Decide which chain to use (regular vs summary)
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
st.session_state.chat_history.append({"role": "user", "content": query})

response = qa_chain.invoke({"query": query})
result_text = response.get("result") or response.get("output_text") or str(response)

st.session_state.chat_history.append({"role": "assistant", "content": result_text})

# Reset PDF
if st.button("🔄 Reset PDF"):
    # Clear only app-related keys to be safe
    for k in ["llm", "chat_history", "qa_chain", "current_pdf_hash", "retriever"]:
        if k in st.session_state:
            del st.session_state[k]
    st.rerun()

# ================= DISPLAY CHAT =================
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
