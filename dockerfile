FROM python:3.11-slim

# System dependencies:
#   tesseract-ocr / libtesseract-dev  — OCR fallback for scanned PDFs
#   libgl1 / libglib2.0-0             — Required by PyMuPDF (fitz) on slim images
#   libgomp1                          — OpenMP runtime for numpy / PyMuPDF
#   poppler-utils                     — PDF utilities used by some langchain loaders
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    libtesseract-dev \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Default data dir — override with DATA_DIR=/data env var when a Render Disk
# is mounted at /data to make sessions/reports persist across restarts.
ENV DATA_DIR=/app/data
RUN mkdir -p /app/data/temp_uploads \
             /app/data/audit_reports \
             /app/data/rules_docs \
             /app/data/rules_chromadb \
             /app/data/sessions \
             /app/data/doc_cache

EXPOSE 8000

CMD ["uvicorn", "Main:app", "--host", "0.0.0.0", "--port", "8000"]
