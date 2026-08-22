# NOT USED in the default deploy path (see README.md — that path uses
# sdk: gradio, which HF's free ZeroGPU exception covers for personal
# accounts). Docker Spaces require a paid HF plan as of mid-2026.
#
# Kept here for two cases: (1) you're on a paid HF plan and prefer
# Docker SDK's full control, or (2) you're deploying this backend to
# any other Docker-based host (Render, Railway, Fly.io, etc.) instead.
# If using this, remove/ignore the `if __name__ == "__main__"` uvicorn
# call at the bottom of app.py and run uvicorn from CMD as before, since
# a Docker host won't need the Gradio-mounting shim.
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
