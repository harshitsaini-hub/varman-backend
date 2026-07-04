FROM python:3.10-slim

WORKDIR /code

ENV TORCH_HOME=/tmp/.cache/torch
ENV HF_HOME=/tmp/.cache/huggingface

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY ./requirements.txt /code/requirements.txt

# Explicitly install the CPU-only version of PyTorch to minimize image layer sizes
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu -r /code/requirements.txt

COPY . .

# Permissions configuration required for Hugging Face container runtimes
RUN chmod -R 777 /code

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
