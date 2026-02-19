import os
import streamlit as st
from sentence_transformers import SentenceTransformer
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
import hashlib

# ================= CONFIG =================
UPLOAD_DIR = "./uploaded_pdfs"
MODEL_PATH = "./sentence-transformers_all-MiniLM-L6-v2"  # local ST model dir or name
OLLAMA_MODEL_NAME = "phi"  # ensure 'ollama run mistral' works on your machine

# Retrieval strictness (tune as needed)
K = 4
SCORE_THRESHOLD = 0.35  # higher => stricter; try 0.30–0.45

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
    
    ⚠️ RULES (Follow them exactly):
    - Use ONLY the text inside the "Context" section.
    - Do NOT use general knowledge.
    - Do NOT guess or infer beyond the PDF.
    - Do NOT mix information from other PDFs or previous chats.
    - If the answer is not explicitly present in the context, reply exactly:
      "I could not find this information in the uploaded PDF."
    
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
    Version-robust across LangChain/Chroma changes:
    tries embedding_function first, falls back to embedding.
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
        chunk_size=400,
        chunk_overlap=40
    )
    chunks = splitter.split_documents(docs)

    if not chunks:
        raise ValueError("No extractable text found in the PDF (it might be a scanned image). Try an OCR version.")

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
    with st.spinner("⏳ Processing PDF…"):
        try:
            # Read ONCE; getvalue() avoids pointer issues after multiple reruns
            pdf_bytes = uploaded_pdf.getvalue()
            if not pdf_bytes:
                st.error("❌ Could not read the uploaded file. Please re-upload the PDF.")
                st.stop()

            pdf_hash = get_pdf_hash(pdf_bytes)

            # Only rebuild if a new/different PDF is uploaded
            if st.session_state.current_pdf_hash != pdf_hash:
                vectorstore = build_vectorstore_from_hash(pdf_hash, pdf_bytes)
                st.session_state.vectorstore = vectorstore

                # Try strict retriever; fallback to basic similarity if unsupported
                try:
                    st.session_state.retriever = vectorstore.as_retriever(
                        search_type="similarity_score_threshold",
                        search_kwargs={"k": K, "score_threshold": SCORE_THRESHOLD},
                    )
                except Exception:
                    st.session_state.retriever = vectorstore.as_retriever(
                        search_kwargs={"k": K}
                    )

                st.session_state.chat_history = []
                st.session_state.current_pdf_hash = pdf_hash

            st.success("✅ PDF processed. You can ask questions now!")
        except Exception as e:
            st.error(f"❌ Error while processing the PDF: {e}")
            st.stop()

# ================= STRICT ASK FUNCTION =================
def strict_ask(query: str, summary: bool = False):
    """
    Enforces PDF-only answers:
    1) Retrieve with score thresholding.
    2) If nothing relevant => return fallback (no LLM call).
    3) Build context from the retrieved chunks and call LLM with a strict prompt.
    """
    if st.session_state.retriever is None:
        return "⚠️ Please upload a PDF first.", None

    # Get relevant documents (the retriever will filter by threshold if configured)
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
        # Some wrappers return objects; ensure string
        answer = str(llm_out)

    return answer, None

# ================= CHAT UI =================
if st.session_state.get("current_pdf_hash"):
    st.caption("📘 PDF loaded. Type your question below.")
else:
    st.caption("📤 Upload a PDF to begin.")

query = st.chat_input("Ask a question from the PDF")

if query:
    st.session_state.chat_history.append({"role": "user", "content": query})

    if st.session_state.retriever is None:
        result_text = "⚠️ Please upload a PDF first."
        sources = None
    else:
        # Decide which mode to use
        summary_mode = is_summary_question(query)
        result_text, sources = strict_ask(query, summary=summary_mode)

    st.session_state.chat_history.append({"role": "assistant", "content": result_text})

    # Show last turn immediately
    with st.chat_message("assistant"):
        st.write(result_text)

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
