FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir "fastapi>=0.115" "uvicorn[standard]>=0.30"
COPY milton/ ./milton/
COPY scripts/ ./scripts/
ENV PYTHONPATH=/app
EXPOSE 8200
CMD ["uvicorn", "milton.web:app", "--host", "0.0.0.0", "--port", "8200"]
