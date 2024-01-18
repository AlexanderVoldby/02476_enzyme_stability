# Base image
FROM python:3.11-slim

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements.txt
COPY pyproject.toml pyproject.toml
COPY mlops_enzyme_stability/ mlops_enzyme_stability/
COPY data/ data/
COPY config.yaml config.yaml

WORKDIR /
RUN pip install -r requirements.txt --no-cache-dir
RUN pip install . --no-deps --no-cache-dir
RUN mlops_enzyme_stability/data/download_BERT.py

# WORKDIR /mlops_enzyme_stability/

EXPOSE 8080

CMD ["uvicorn", "mlops_enzyme_stability.predict_sequence:app", "--host", "0.0.0.0", "--port", "8080"]