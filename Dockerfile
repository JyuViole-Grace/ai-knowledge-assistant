FROM python:3.12-slim

WORKDIR /app

# CPU-only PyTorch keeps the image far smaller than the default CUDA build.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY rag_assistant.py .

ENV OLLAMA_URL=http://ollama:11434 \
    RAG_DOCS_ROOT=/data/documents \
    HF_HOME=/models

EXPOSE 8000

CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
