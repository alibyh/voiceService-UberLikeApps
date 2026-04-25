FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Build indexes at image-build time so cold starts are fast.
# If you'd rather skip the heavy semantic index, change this to:
#   RUN python -m matcher.index_build --skip-semantic
RUN python -m matcher.index_build

EXPOSE 8000
CMD ["uvicorn", "matcher.api:app", "--host", "0.0.0.0", "--port", "8000"]
