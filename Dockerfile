FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MODEL_PATH=/app/models/heart_disease_pipeline.joblib \
    PYTHONPATH=/app/src

WORKDIR /app
RUN pip install --no-cache-dir fastapi joblib pandas scikit-learn "uvicorn[standard]" prometheus-client
COPY src/ ./src/
COPY models/heart_disease_pipeline.joblib ./models/heart_disease_pipeline.joblib
RUN useradd --create-home --uid 10001 appuser \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000
CMD ["uvicorn", "heart_disease.api:app", "--host", "0.0.0.0", "--port", "8000"]