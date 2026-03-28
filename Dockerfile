FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libmagic1 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml .
RUN pip install --no-cache-dir .

COPY src/ src/
RUN pip install --no-cache-dir .

EXPOSE 8000

CMD ["uvicorn", "docmind.app:app", "--host", "0.0.0.0", "--port", "8000"]
