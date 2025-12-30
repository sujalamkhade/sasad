uvicorn ingest_service:app --reload --port 8000


Sansad PDF / Dummy PDF
        ↓
n8n HTTP node (multipart upload)
        ↓
POST /ingest-file
        ↓
ingest_service.py
        ↓
PDF saved → text extracted → chunks created
        ↓
JSON returned to n8n
        ↓
Google Sheets
