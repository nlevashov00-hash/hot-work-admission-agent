FROM python:3.11-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

# Git is needed by the autonomous git-check-ignore test.
RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .gitignore REQUIREMENTS.md run_tests.sh ./
COPY app/ ./app/
COPY eval/ ./eval/
COPY tests/ ./tests/

CMD ["python", "-m", "app.pipeline"]
