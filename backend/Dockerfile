# Hugging Face Spaces — Docker SDK
# Space settings: SDK = Docker, this file at repo root of the Space (backend/ folder)
FROM python:3.11-slim

WORKDIR /app

# System deps for lxml / bs4
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# HF Spaces expects the container to listen on 7860
ENV PORT=7860
EXPOSE 7860

# data/ holds the SQLite file. On free HF Spaces this disk is EPHEMERAL —
# it resets on every Space rebuild/sleep unless you enable "Persistent Storage"
# (paid) in Space settings. See README.md "Known gaps" for the honest caveat.
RUN mkdir -p /app/data

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
