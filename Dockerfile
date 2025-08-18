FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY prompts ./prompts
COPY src ./src

ENV PYTHONPATH=/app/src

# Create data dir for SQLite
RUN mkdir -p /data

EXPOSE 8000

# entrypoint selects CLI or API
ENTRYPOINT ["python", "-m", "agent.cli"]
CMD ["--help"]