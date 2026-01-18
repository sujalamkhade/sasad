# ingest_service.py
# FastAPI ingest service for Sansad RAG pipeline
# Endpoints:
#   POST /ingest        -> download PDF from URL
#   POST /ingest-file   -> upload PDF file
#
# Run:
#   uvicorn ingest_service:app --reload --port 8000

from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from pydantic import BaseModel
import requests, os, io, hashlib, time, uuid, json

import pdfplumber
from PyPDF2 import PdfReader
from langdetect import detect
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ===============================
# APP INIT
# ===============================

app = FastAPI(title="Sansad Ingest Service")

# ===============================
# DIRECTORIES
# ===============================

DATA_DIR = "data"
PDF_DIR = os.path.join(DATA_DIR, "pdfs")
CHUNK_DIR = os.path.join(DATA_DIR, "chunks")
INDEX_FILE = os.path.join(DATA_DIR, "index.json")

os.makedirs(PDF_DIR, exist_ok=True)
os.makedirs(CHUNK_DIR, exist_ok=True)

# ===============================
# CONSTANTS
# ===============================

USER_AGENT = "MySansadScraper/1.0 (+contact@example.com)"
HEADERS = {"User-Agent": USER_AGENT}
MAX_PDF_SIZE_MB = 25

# ===============================
# MODELS
# ===============================

class IngestRequest(BaseModel):
    pdf_url: str
    source: str | None = None

# ===============================
# UTILS
# ===============================

def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

def load_index() -> dict:
    if os.path.exists(INDEX_FILE):
        with open(INDEX_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_index(index: dict):
    with open(INDEX_FILE, "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2)

def download_pdf(url: str, timeout=30) -> bytes:
    session = requests.Session()
    retries = Retry(
        total=3,
        backoff_factor=0.8,
        status_forcelist=[429, 500, 502, 503, 504],
    )
    adapter = HTTPAdapter(max_retries=retries)
    session.mount("http://", adapter)
    session.mount("https://", adapter)

    resp = session.get(url, headers=HEADERS, timeout=timeout)
    resp.raise_for_status()

    return resp.content

def extract_text_and_meta(pdf_bytes: bytes):
    text = ""
    meta = {}

    try:
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            reader = PdfReader(io.BytesIO(pdf_bytes))
            raw_meta = reader.metadata or {}

            for k, v in raw_meta.items():
                meta[k] = str(v)

            pages = []
            for page in pdf.pages:
                pages.append(page.extract_text() or "")

            text = "\n\n".join(pages).strip()

    except Exception as e:
        print("PDF extraction error:", e)

    return text, meta

def chunk_text(text: str, target_words=400, overlap=0.2):
    if not text:
        return []

    words = text.split()
    chunks = []

    step = int(target_words * (1 - overlap)) or target_words
    i = 0

    while i < len(words):
        chunk = words[i:i + target_words]
        chunks.append(" ".join(chunk))
        i += step

    return chunks

# ===============================
# CORE PROCESSOR
# ===============================

def process_pdf_bytes(pdf_bytes: bytes, source: str | None = None):
    if len(pdf_bytes) > MAX_PDF_SIZE_MB * 1024 * 1024:
        raise HTTPException(413, "PDF file too large")

    checksum = sha256_bytes(pdf_bytes)
    index = load_index()

    # ---- Duplicate check ----
    if checksum in index:
        return {
            "status": "duplicate",
            "existing_pdf": index[checksum],
            "sha256": checksum
        }

    fname = f"{int(time.time())}_{uuid.uuid4().hex[:8]}.pdf"
    pdf_path = os.path.join(PDF_DIR, fname)

    with open(pdf_path, "wb") as f:
        f.write(pdf_bytes)

    text, meta = extract_text_and_meta(pdf_bytes)

    needs_ocr = not text or len(text) < 100

    # ---- Language detection ----
    language = "unknown"
    if text and len(text) > 30:
        try:
            lang = detect(text)
            if lang in {"en", "hi"}:
                language = lang
        except Exception:
            pass

    chunks = chunk_text(text)
    chunk_ids = []

    for i, c in enumerate(chunks):
        cid = f"{fname}.chunk{i}.txt"
        with open(os.path.join(CHUNK_DIR, cid), "w", encoding="utf-8") as fh:
            fh.write(c[:2000])
        chunk_ids.append(cid)

    # ---- Save index ----
    index[checksum] = fname
    save_index(index)

    return {
        "status": "ingested",
        "pdf_filename": fname,
        "pdf_path": pdf_path,
        "sha256": checksum,
        "source": source or "",
        "language": language,
        "needs_ocr": needs_ocr,
        "next_step": "ocr_required" if needs_ocr else "ready_for_embedding",
        "num_chunks": len(chunks),
        "chunk_ids": chunk_ids,
        "meta": meta,
    }

# ===============================
# API ENDPOINTS
# ===============================

@app.post("/ingest")
async def ingest(req: IngestRequest):
    try:
        pdf_bytes = download_pdf(req.pdf_url)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Download failed: {e}")

    return process_pdf_bytes(pdf_bytes, source=req.source)

@app.post("/ingest-file")
async def ingest_file(
    source: str | None = Form(None),
    file: UploadFile = File(...),
):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are supported")

    try:
        pdf_bytes = await file.read()
    except Exception as e:
        raise HTTPException(400, f"Failed to read file: {e}")

    return process_pdf_bytes(pdf_bytes, source=source)
