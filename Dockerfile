FROM python:3.11-slim AS base
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
# Validate OpenEnv spec
RUN pip install openenv && openenv validate openenv.yaml
ENV PYTHONUNBUFFERED=1
ENTRYPOINT ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "7860"]
