FROM python:3.7-slim-buster

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        mesa-utils \
        libgl1-mesa-glx \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender-dev

# Copy the Flask app code and requirements
COPY app.py .
COPY config.py .
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy controllers and services
COPY controllers/ ./controllers/
COPY services/ ./services/
COPY .env .env

# Copy the PyTorch model weight file and directories
COPY Net/ ./Net/
COPY mrcnn/ ./mrcnn/
COPY PredictResult/ ./PredictResult/
COPY ReceivedFile/ ./ReceivedFile/
COPY CloudinaryFile/ ./CloudinaryFile/

# Copy test folder
# COPY tests/ ./tests/
# RUN coverage run -m unittest discover tests

CMD flask run -h 0.0.0.0 -p 5000