"""Общая логика инференса для демо-приложений (День 7).

Используется и Gradio-приложением (app.py), и FastAPI-сервисом (api.py),
чтобы предсказание было ровно одним и тем же в обоих.
"""

from __future__ import annotations

from functools import lru_cache

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_PATH = "./fine_tuned_model"

# Модель обучалась на SST-2 — классов ДВА, а не три.
# В шаблоне задания стояла карта {0: Negative, 1: Neutral, 2: Positive},
# с которой наш класс 1 (positive) отображался бы как "Neutral".
LABEL_MAP = {0: "Negative", 1: "Positive"}


@lru_cache(maxsize=1)
def load() -> tuple:
    """Загружает модель и токенизатор один раз на процесс."""
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model.eval()
    return model, tokenizer


def predict(text: str, max_length: int = 128) -> dict:
    """Определяет тональность текста.

    Args:
        text: входной текст.
        max_length: максимальная длина в токенах.

    Returns:
        Словарь вида
        {"label": "Positive", "confidence": 0.99, "probabilities": {...}}.

    Raises:
        ValueError: если текст пустой.
    """
    if not text or not text.strip():
        raise ValueError("Текст пустой")

    model, tokenizer = load()

    inputs = tokenizer(
        text, return_tensors="pt", truncation=True, max_length=max_length
    )

    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.nn.functional.softmax(outputs.logits, dim=1)[0]
    pred = int(torch.argmax(probs).item())

    return {
        "label": LABEL_MAP[pred],
        "confidence": float(probs[pred]),
        "probabilities": {
            LABEL_MAP[i]: float(p) for i, p in enumerate(probs)
        },
    }
