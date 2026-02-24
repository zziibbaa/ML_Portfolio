FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN apt-get update && \
    apt-get install -y gcc && \
    pip install --no-cache-dir -r requirements.txt && \
    apt-get purge -y gcc && \
    apt-get autoremove -y && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*


COPY . .

EXPOSE 8000

ENTRYPOINT ["uvicorn", "fast_api:app"]
CMD ["--host", "0.0.0.0", "--port", "8000"]
