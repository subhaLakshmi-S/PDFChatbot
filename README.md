# PDFChatbot
**📄 PDF Chatbot (Offline)**

**An offline PDF-based chatbot built using:**

~Streamlit (UI)

~Sentence Transformers (Embeddings)

~Ollama (Local LLM)

~Python

This application allows users to upload a PDF and ask questions based on its content — completely offline.

**🛠️ System Requirements**

Python 3.9+

Git

Ollama installed locally

Minimum 8GB RAM recommended

**🚀 Step-by-Step Setup Guide
1️⃣ Clone the Repository**
git clone https://github.com/yourusername/pdf-chatbot.git
cd pdf-chatbot

**2️⃣ Create Virtual Environment
On Windows:**
python -m venv venv
venv\Scripts\activate

**On Mac/Linux:**
python3 -m venv venv
source venv/bin/activate

**3️⃣ Install Dependencies**
pip install -r requirements.txt


**If requirements.txt is missing, install manually:**

pip install streamlit sentence-transformers langchain pypdf faiss-cpu

**🧠 Install and Setup Ollama (Required for Offline LLM)**
**🔹 Step 1: Install Ollama**

Go to official website:

👉 https://ollama.com

Download and install based on your OS:

Windows → Install .exe

Mac → Install .dmg

Linux → Use curl command from website

After installation, verify:

ollama --version

**🔹 Step 2: Pull a Model**

Download a local LLM model (example: Llama3):

ollama pull llama3


Other recommended models:

mistral

phi3

llama3

**🔹 Step 3: Test Ollama**

Run:

ollama run llama3


If it opens a chat interface → Ollama is working ✅

Exit using:

/bye

**▶️ Run the Application**

After dependencies and Ollama setup:

streamlit run app.py


It will open in your browser:

http://localhost:8501


Upload a PDF and start asking questions.

**🏗️ How It Works (Architecture)**
User uploads PDF

PDF text is extracted

Text is split into chunks

Sentence Transformer generates embeddings

FAISS stores embeddings in vector store

User asks a question

Relevant chunks are retrieved

Ollama LLM generates final answer

Everything runs locally.
No API keys required.
No internet required after model download.

**📴 Offline Capability**

Once:

Dependencies are installed

Ollama model is downloaded

The chatbot works completely offline.
