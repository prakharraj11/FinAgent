FROM python:3.11-slim

# Install system dependencies, specifically Tesseract OCR
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    libtesseract-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code
COPY . .

# Ensure the required directories exist
RUN mkdir -p temp_uploads audit_reports rules_docs sessions

# Expose the port
EXPOSE 8000

# Start the FastAPI server using Railway's dynamic PORT
CMD ["sh", "-c", "uvicorn Main:app --host 0.0.0.0 --port ${PORT:-8000}"]
