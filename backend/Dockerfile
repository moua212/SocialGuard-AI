FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/app:/app/backend SG_ROOT=/app
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY backend/requirements.txt backend/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt
COPY ml ml
COPY backend backend
COPY scripts scripts
COPY docs docs
# Train a model at build time if no registered model is shipped (reproducible, seed=42)
RUN python -c "from ml.models.registry import read_registry; import sys; sys.exit(0 if read_registry()['active'] else 1)" \
    || python -m ml.training.train --n-accounts 2000
RUN useradd -m app && chown -R app /app
USER app
WORKDIR /app/backend
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
