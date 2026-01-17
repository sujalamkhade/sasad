# ingest_service.py
# Minimal working FastAPI ingest service with two endpoints:
#  - POST /ingest       (download from URL and process)
#  - POST /ingest-file  (upload file bytes and process)
#
# Start with:
#   uvicorn ingest_service:app --reload --port 8000

from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from pydantic import BaseModel
import requests, os, io, hashlib, time, uuid, json
import pdfplumber
from PyPDF2 import PdfReader
from langdetect import detect
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

app = FastAPI()

# dirs
DATA_DIR = "data"
PDF_DIR = os.path.join(DATA_DIR, "pdfs")
CHUNK_DIR = os.path.join(DATA_DIR, "chunks")
os.makedirs(PDF_DIR, exist_ok=True)
os.makedirs(CHUNK_DIR, exist_ok=True)

USER_AGENT = "MySansadScraper/1.0 (+youremail@example.com)"
HEADERS = {"User-Agent": USER_AGENT}

class IngestRequest(BaseModel):
    pdf_url: str
    source: str | None = None

def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

def download_pdf(url: str, timeout=30):
    """
    Robust download using retries for transient errors.
    Note: DNS errors will still fail here.
    """
    session = requests.Session()
    retries = Retry(total=3, backoff_factor=0.8, status_forcelist=[429,500,502,503,504])
    adapter = HTTPAdapter(max_retries=retries)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    resp = session.get(url, headers=HEADERS, timeout=timeout)
    resp.raise_for_status()
    return resp.content

def extract_text_and_meta(pdf_bytes: bytes):
    """
    Try to extract selectable text and metadata using pdfplumber/PyPDF2.
    (No OCR here; that is optional later.)
    """
    text = ""
    meta = {}
    try:
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            reader = PdfReader(io.BytesIO(pdf_bytes))
            meta_raw = reader.metadata or {}
            for k, v in meta_raw.items():
                meta[k] = str(v)
            pages = []
            for p in pdf.pages:
                pages.append(p.extract_text() or "")
            text = "\n\n".join(pages).strip()
    except Exception as e:
        # extraction failure -> return empty text and meta possibly partial
        print("pdfplumber error:", e)
    return text, meta

def chunk_text(text: str, target_words=400, overlap=0.2):
    if not text:
        return []
    words = text.split()
    chunks = []
    step = int(target_words * (1 - overlap)) or target_words
    i = 0
    while i < len(words):
        chunk_words = words[i:i+target_words]
        chunks.append(" ".join(chunk_words))
        i += step
    return chunks

def _process_pdf_bytes(pdf_bytes: bytes, source: str | None = None):
    """
    Save PDF, extract text/meta, chunk, write chunk previews, return result dict.
    """
    checksum = sha256_bytes(pdf_bytes)
    fname = f"{int(time.time())}_{uuid.uuid4().hex[:8]}.pdf"
    pdf_path = os.path.join(PDF_DIR, fname)
    with open(pdf_path, "wb") as f:
        f.write(pdf_bytes)

    text, meta = extract_text_and_meta(pdf_bytes)
    needs_ocr = False
    if not text or len(text) < 100:
        needs_ocr = True

    language = None
    if text and len(text) > 30:
        try:
            language = detect(text)
        except Exception:
            language = "unknown"

    chunks = chunk_text(text, target_words=400, overlap=0.2)
    chunk_ids = []
    for i, c in enumerate(chunks):
        cid = f"{fname}.chunk{i}.txt"
        with open(os.path.join(CHUNK_DIR, cid), "w", encoding="utf-8") as fh:
            fh.write(c[:2000])  # store preview
        chunk_ids.append(cid)

    result = {
        "pdf_filename": fname,
        "sha256": checksum,
        "needs_ocr": needs_ocr,
        "language": language,
        "num_chunks": len(chunks),
        "chunk_ids": chunk_ids,
        "meta": meta,
        "pdf_path": pdf_path,
        "source": source or ""
    }
    return result

@app.post("/ingest")
async def ingest(req: IngestRequest):
    try:
        pdf_bytes = download_pdf(req.pdf_url)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Download failed: {e}")

    result = _process_pdf_bytes(pdf_bytes, source=req.source)
    return result

@app.post("/ingest-file")
async def ingest_file(
    source: str | None = Form(None),            # accept 'source' from multipart form-data
    file: UploadFile = File(...),               # file from multipart
):
    """
    Accepts multipart/form-data with form field 'source' and binary field 'file'.
    Example via curl:
      curl -X POST "http://127.0.0.1:8000/ingest-file" \
        -F "source=n8n_test" -F "file=@/path/to/file.pdf"
    """
    try:
        pdf_bytes = await file.read()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read uploaded file: {e}")

    result = _process_pdf_bytes(pdf_bytes, source=source)
    return result

# run with:
# uvicorn ingest_service:app --reload --port 8000
