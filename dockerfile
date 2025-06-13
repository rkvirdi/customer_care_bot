FROM python:3.10-slim

WORKDIR /app

# 1) Copy requirements.txt into the image
COPY requirements.txt .

# 2) Install OS deps needed for packages like psycopg2
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      build-essential \
      gcc \
      libpq-dev \
 && rm -rf /var/lib/apt/lists/*

# 3) Install Python deps
RUN pip install --no-cache-dir -r requirements.txt

# 4) Copy the rest of your code
COPY . .

EXPOSE 8000
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
