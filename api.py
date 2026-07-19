"""FastAPI-сервис для модели (День 7).

Запуск:
    uvicorn api:app --reload

Документация: http://127.0.0.1:8000/docs

Пример запроса:
    curl -X POST http://127.0.0.1:8000/predict \
         -H "Content-Type: application/json" \
         -d '{"text": "This movie was great!"}'
"""

from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from predictor import predict

app = FastAPI(
    title="Sentiment Analysis API",
    description="DistilBERT, дообученный на SST-2 (macro F1 = 0.92)",
    version="1.0.0",
)


class PredictionRequest(BaseModel):
    text: str = Field(..., min_length=1, examples=["This movie was great!"])


class BatchRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1)


@app.get("/health")
def health() -> dict:
    """Проверка, что сервис жив."""
    return {"status": "ok"}


@app.post("/predict")
def predict_one(request: PredictionRequest) -> dict:
    """Определяет тональность одного текста."""
    try:
        return predict(request.text)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/predict_batch")
def predict_batch(request: BatchRequest) -> dict:
    """Определяет тональность списка текстов."""
    try:
        return {"results": [predict(t) for t in request.texts]}
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
