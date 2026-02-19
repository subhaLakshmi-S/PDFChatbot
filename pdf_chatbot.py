import os
import shutil
import streamlit as st
from sentence_transformers import SentenceTransformer
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
import uuid
import hashlib

# ================= CONFIG =================
UPLOAD_DIR = "./uploaded_pdfs"
MODEL_PATH = "./sentence-transformers_all-MiniLM-L6-v2"
OLLAMA_MODEL_NAME = "mistral"

# Retrieval strictness (tune as needed)
K = 8
SCORE_THRESHOLD = 0.35  # higher => stricter; 0.3–0.45 is typical

os.makedirs(UPLOAD_DIR, exist_ok=True)

# ================= STREAMLIT UI =================
st.set_page_config(page_title="PDF Chatbot", layout="wide")
st.title("📚 Chat with Your PDF (Strict PDF-only Answers)")

# ================= SESSION STATE =================
if "llm" not in st.session_state:
    st.session_state.llm = Ollama(
        model=OLLAMA_MODEL_NAME,
        temperature=0
    )

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "retriever" not in st.session_state:
    st.session_state.retriever = None

if "current_pdf_hash" not in st.session_state:
    st.session_state.current_pdf_hash = None

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

# ================= EMBEDDINGS =================
@st.cache_resource
def load_embeddings():
    # CPU inference; device="cpu"
    return SentenceTransformer(MODEL_PATH, device="cpu")

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

⚠️ RULES (Follow exactly):
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
Summarize the document based ONLY on the uploaded PDF context.
You MAY combine details across sections, but do not introduce any external knowledge.
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
    This is version-robust across LangChain/Chroma changes:
    it tries embedding_function first, falls back to embedding.
    """
    temp_path = f"temp_{pdf_hash}.pdf"
    with open(temp_path, "wb") as f:
        f.write(pdf_bytes)

    try:
        loader = PyPDFLoader(temp_path)
        docs = loader.load()
    finally:
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
    collection_name = f"pdf_{pdf_hash}"

    def _load_chroma(path: str):
        # Try embedding_function first; if TypeError, retry with embedding
        try:
            return Chroma(
                persist_directory=path,
                collection_name=collection_name,
                embedding_function=CustomEmbedding()
            )
        except TypeError:
            # Older API fallback
            return Chroma(
                persist_directory=path,
                collection_name=collection_name,
                embedding=CustomEmbedding()
            )

    def _create_chroma_from_docs(_docs, path: str):
        # Try embedding_function first; if TypeError, retry with embedding
        try:
            vs = Chroma.from_documents(
                documents=_docs,
                embedding_function=CustomEmbedding(),
                persist_directory=path,
                collection_name=collection_name,
            )
        except TypeError:
            vs = Chroma.from_documents(
                documents=_docs,
                embedding=CustomEmbedding(),
                persist_directory=path,
                collection_name=collection_name,
            )
        vs.persist()
        return vs

    if os.path.exists(db_path):
        return _load_chroma(db_path)

    return _create_chroma_from_docs(chunks, db_path)

# ================= PDF UPLOADER =================
uploaded_pdf = st.file_uploader("📄 Upload a PDF", type=["pdf"])

if uploaded_pdf:
    pdf_bytes = uploaded_pdf.read()
    pdf_hash = get_pdf_hash(pdf_bytes)

    if st.session_state.current_pdf_hash != pdf_hash:
        vectorstore = build_vectorstore_from_hash(pdf_hash, pdf_bytes)
        st.session_state.vectorstore = vectorstore

        st.session_state.retriever = vectorstore.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": K, "score_threshold": SCORE_THRESHOLD},
        )

        st.session_state.chat_history = []
        st.session_state.current_pdf_hash = pdf_hash

# ================= STRICT ASK FUNCTION =================
def strict_ask(query: str, summary: bool = False, show_sources: bool = False):
    """
    Enforces PDF-only answers:
    1) Retrieve with score thresholding.
    2) If nothing relevant => return fallback (no LLM call).
    3) Build context from the retrieved chunks and call LLM with a strict prompt.
    """
    if st.session_state.retriever is None:
        return "⚠️ Please upload a PDF first.", None

    # Get relevant documents (the retriever will filter by threshold)
    docs = st.session_state.retriever.get_relevant_documents(query)

    if not docs:
        return "I could not find this information in the uploaded PDF.", []

    context = "\n\n".join(d.page_content for d in docs)
    prompt = (SUMMARY_PROMPT if summary else PDF_PROMPT).format(
        context=context,
        question=query
    )

    llm_out = st.session_state.llm.invoke(prompt)
    if isinstance(llm_out, str):
        answer = llm_out.strip()
    else:
        answer = str(llm_out)

    if show_sources:
        # Try to also show scores (fallback to 2nd call through vectorstore API)
        try:
            scored = st.session_state.vectorstore.similarity_search_with_score(query, k=K)
            sources = [(d.metadata, s) for d, s in scored]
        except Exception:
            sources = [(getattr(d, "metadata", {}), None) for d in docs]
        return answer, sources

    return answer, None

# ================= CHAT UI =================
query = st.chat_input("Ask a question from the PDF")

# Quick controls row
col1, col2 = st.columns([1, 1])
with col1:
    show_sources = st.toggle("Show sources (debug)", value=False, help="View the chunks retrieved and their scores.")
with col2:
    st.caption(f"Threshold: {SCORE_THRESHOLD} | Top-K: {K} | Model: {OLLAMA_MODEL_NAME}")

if query:
    st.session_state.chat_history.append({"role": "user", "content": query})
    
    if st.session_state.retriever is None:
        result_text = "⚠️ Please upload a PDF first."
        sources = None
    else:
        # Decide which mode to use
        summary_mode = is_summary_question(query)
        result_text, sources = strict_ask(query, summary=summary_mode, show_sources=show_sources)

    st.session_state.chat_history.append({"role": "assistant", "content": result_text})

    # Show last turn immediately
    with st.chat_message("assistant"):
        st.write(result_text)
        if show_sources and sources is not None:
            st.markdown("**Retrieved Chunks (metadata, score):**")
            for i, (meta, score) in enumerate(sources, start=1):
                st.write(f"[{i}] score={score}")
                st.json(meta)

# Reset PDF / Session
if st.button("🔄 Reset PDF"):
    # Clear only app-related keys to be safe
    for k in ["llm", "chat_history", "current_pdf_hash", "retriever", "vectorstore"]:
        if k in st.session_state:
            del st.session_state[k]
    st.rerun()

# ================= DISPLAY FULL CHAT =================
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
