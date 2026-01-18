# ===============================
# Sansad RAG Pipeline (FINAL – FIXED)
# Local Embeddings + ChromaDB + Gemini Answering
# ===============================

import os
import sys
from dotenv import load_dotenv

import chromadb
from chromadb.config import Settings

from sentence_transformers import SentenceTransformer
import google.generativeai as genai

import pdfplumber
from textwrap import wrap

# ===============================
# 1. ENV + PATHS (FIXED)
# ===============================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

load_dotenv(os.path.join(BASE_DIR, ".env"))

PDF_DIR = os.path.join(BASE_DIR, "data", "pdfs")
CHROMA_DIR = os.path.join(BASE_DIR, "chroma_db")

os.makedirs(PDF_DIR, exist_ok=True)
os.makedirs(CHROMA_DIR, exist_ok=True)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    print("❌ GEMINI_API_KEY not found in .env")
    sys.exit(1)

print("✅ Environment loaded")

# ===============================
# 2. LOAD MODELS
# ===============================

print("🔧 Loading local embedding model...")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
print("✅ Embedding model ready")

genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel("gemini-1.5-flash")

# ===============================
# 3. CHROMA DB SETUP
# ===============================

chroma_client = chromadb.Client(
    Settings(
        persist_directory=CHROMA_DIR,
        anonymized_telemetry=False
    )
)

collection = chroma_client.get_or_create_collection(
    name="sansad_sessions"
)

# ===============================
# 4. HELPERS
# ===============================

def read_pdf_text(pdf_path: str) -> str:
    """Extract text from PDF"""
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text.strip()


def chunk_text(text: str, chunk_size=1200):
    """Split text into chunks"""
    return wrap(text, chunk_size)


def embed_text(text: str):
    """Convert text → vector"""
    return embedding_model.encode(text).tolist()

# ===============================
# 5. INGESTION
# ===============================

def ingest_pdfs():
    pdf_files = [f for f in os.listdir(PDF_DIR) if f.lower().endswith(".pdf")]

    if not pdf_files:
        print("⚠️ No PDFs found in data/pdfs")
        return

    total_chunks = 0

    for pdf in pdf_files:
        print(f"\n📄 Processing {pdf}")
        pdf_path = os.path.join(PDF_DIR, pdf)

        text = read_pdf_text(pdf_path)
        if not text:
            print("⚠️ No extractable text, skipping")
            continue

        chunks = chunk_text(text)

        documents, embeddings, metadatas, ids = [], [], [], []

        for idx, chunk in enumerate(chunks):
            documents.append(chunk)
            embeddings.append(embed_text(chunk))
            metadatas.append({
                "source": pdf,
                "chunk": idx
            })
            ids.append(f"{pdf}_{idx}")

        collection.add(
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )

        total_chunks += len(chunks)
        print(f"✅ Stored {len(chunks)} chunks from {pdf}")

    print(f"\n🎉 Ingestion complete — total chunks: {total_chunks}")

# ===============================
# 6. ASK QUESTIONS (RAG)
# ===============================

def ask_question():
    count = collection.count()
    print(f"\n🔎 Collection count: {count}")

    if count == 0:
        print("❌ No data found. Run ingestion first.")
        return

    while True:
        question = input("\n❓ Question (or 'exit'): ").strip()
        if question.lower() == "exit":
            break

        query_embedding = embed_text(question)

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=5
        )

        retrieved_chunks = results["documents"][0]

        if not retrieved_chunks:
            print("⚠️ No relevant context found")
            continue

        context = "\n\n".join(retrieved_chunks)

        prompt = f"""
You are a research assistant helping understand Indian Parliament sessions.

Using ONLY the context below:
- Identify key issues discussed
- Mention dates if present
- Summarize clearly in bullet points
- If date is missing, say "date not specified"

Context:
{context}

Question:
{question}

Answer:
"""

        response = gemini_model.generate_content(prompt)

        print("\n🧠 Answer:\n")
        print(response.text)

# ===============================
# 7. MAIN MENU
# ===============================

def main():
    print("\n📚 Sansad RAG Pipeline")
    print("1. Ingest PDFs")
    print("2. Ask a question")

    choice = input("Choose (1/2): ").strip()

    if choice == "1":
        ingest_pdfs()
    elif choice == "2":
        ask_question()
    else:
        print("❌ Invalid choice")

# ===============================
# ENTRY POINT
# ===============================

if __name__ == "__main__":
    main()
