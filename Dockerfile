FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml ./
RUN pip install --no-cache-dir "fastapi>=0.111.0" "uvicorn[standard]>=0.29.0" "anthropic>=0.28.0" "httpx>=0.27.0" "python-dotenv>=1.0.0"

COPY . .

EXPOSE 10000

CMD sh -c "uvicorn src.michael_doors_bot.main:app --host 0.0.0.0 --port ${PORT:-10000}"
