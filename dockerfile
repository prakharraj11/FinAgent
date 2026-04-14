FROM python:3.11-slim

# System dependencies
# - tesseract-ocr / libtesseract-dev : OCR fallback for scanned PDFs
# - libgl1 libglib2.0-0              : Required by PyMuPDF (fitz) on slim images
# - libmupdf-dev                     : MuPDF C library used by PyMuPDF wheels
# - poppler-utils                    : PDF utilities used by some langchain loaders
# - libgomp1                         : OpenMP runtime (numpy / PyMuPDF parallelism)
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    libtesseract-dev \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Create persistent directories so they survive within a container run
# (mount a Render Disk to /app/data for true persistence across deploys)
RUN mkdir -p temp_uploads audit_reports rules_docs rules_chromadb sessions doc_cache

EXPOSE 8000

CMD ["uvicorn", "Main:app", "--host", "0.0.0.0", "--port", "8000"]
