FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    POETRY_VIRTUALENVS_CREATE=false \
    POETRY_NO_INTERACTION=1

WORKDIR /app

# Install dependencies first so this layer is cached until they change.
# poetry.lock was generated with Poetry 2.1.3.
COPY pyproject.toml poetry.lock ./
RUN pip install "poetry==2.1.3" \
    && poetry install --only main --no-root

# Copy the application code (see .dockerignore for what is left out)
COPY . .

# Run as a non-root user. The app writes generated code to ./logs.
RUN useradd --create-home appuser \
    && mkdir -p /app/logs \
    && chown -R appuser:appuser /app
USER appuser

# OPENAI_API_KEY is NOT baked into the image: pass it at run time
EXPOSE 8501

HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')" || exit 1

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
