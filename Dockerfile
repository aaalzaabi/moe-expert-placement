# CPU-only image for the planner API, the dashboard and the test suite.
# It does not contain the OLMoE weights or torch; 'Try your own prompt' reports that the model is unavailable.
FROM python:3.11.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MOE_PROJECT_ROOT=/app

WORKDIR /app

COPY requirements-ci.txt requirements-serve.txt ./
RUN pip install -r requirements-serve.txt

# Source, saved measurements/results and the two small checkpoint metadata files (see .dockerignore)
COPY . .

# Run as an unprivileged user; data/ holds the SQLite database when no MOE_DATABASE_URL is given
# Some source files are mode 600 on the author's machine; make the tree world-readable, data/ writable by app
RUN chmod -R a+rX /app && useradd --create-home --uid 1000 app && mkdir -p /app/data && chown -R app:app /app/data
USER app

EXPOSE 8000 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health', timeout=4).status == 200 else 1)"

CMD ["uvicorn", "service.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
