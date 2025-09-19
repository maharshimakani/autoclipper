FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 
    PIP_NO_CACHE_DIR=1 
    PYTHONDONTWRITEBYTECODE=1

# System deps for ffmpeg, OpenCV, and librosa/soundfile
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libsndfile1 \
    build-essential \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Python deps
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# App
COPY app.py ./app.py
# If you add a Streamlit config later, also:
# COPY .streamlit ./.streamlit

# Streamlit server
ENV PORT=8501
EXPOSE 8501
CMD streamlit run app.py --server.port  --server.address 0.0.0.0 --server.headless true
