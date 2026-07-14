"""Переиспользуемые утилиты для эмбеддингов / hidden states (День 2).

Опирается на токенизацию из Дня 1 ([[tokenization_utils]]) и добавляет:
загрузку модели, извлечение CLS-эмбеддингов батчами и косинусное сходство.

Пример:
    from tokenization_utils import load_tokenizer
    from embeddings_utils import load_model, get_embeddings, similarity

    tokenizer = load_tokenizer("distilbert-base-uncased")
    model = load_model("distilbert-base-uncased")

    emb = get_embeddings(["Great movie!", "Terrible film!"], tokenizer, model)
    print(emb.shape)  # (2, 768) для DistilBERT

    print(similarity("Great movie!", "Amazing film!", tokenizer, model))

Этот код понадобится в День 3 (визуализация) и День 4 (классификация).
"""

from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModel, PreTrainedModel, PreTrainedTokenizerBase

from tokenization_utils import DEFAULT_MODEL, tokenize_texts


def load_model(model_name: str = DEFAULT_MODEL) -> PreTrainedModel:
    """Загружает модель и переводит её в режим оценки (eval).

    В режиме eval отключается dropout и т.п. — при извлечении эмбеддингов
    градиенты и обучающее поведение нам не нужны.

    Args:
        model_name: имя модели на Hugging Face Hub.

    Returns:
        Модель в режиме eval.
    """
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    return model


def get_embeddings(
    texts: list[str],
    tokenizer: PreTrainedTokenizerBase,
    model: PreTrainedModel,
    batch_size: int = 32,
    max_length: int = 128,
) -> np.ndarray:
    """Возвращает CLS-эмбеддинги для списка текстов.

    Тексты обрабатываются батчами: каждый батч токенизируется
    (через tokenize_texts из Дня 1), прогоняется через модель без
    градиентов, после чего берётся вектор CLS-токена (позиция 0).

    Args:
        texts: список строк.
        tokenizer: загруженный токенизатор.
        model: загруженная модель.
        batch_size: размер батча.
        max_length: максимальная длина последовательности.

    Returns:
        Массив numpy формы (len(texts), hidden_size).
    """
    all_embeddings = []

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]

        # Токенизируем батч функцией из Дня 1.
        tokens = tokenize_texts(batch_texts, tokenizer, max_length=max_length)

        # Прогоняем через модель без построения графа градиентов.
        with torch.no_grad():
            outputs = model(**tokens)

        # CLS-токен — первый в последовательности: [batch, 0, hidden].
        cls_embeddings = outputs.last_hidden_state[:, 0, :]
        all_embeddings.append(cls_embeddings.cpu().numpy())

    return np.vstack(all_embeddings)


def similarity(
    text1: str,
    text2: str,
    tokenizer: PreTrainedTokenizerBase,
    model: PreTrainedModel,
) -> float:
    """Косинусное сходство CLS-эмбеддингов двух текстов.

    Returns:
        Число в диапазоне [-1, 1]; ближе к 1 — тексты похожи.
    """
    emb = get_embeddings([text1, text2], tokenizer, model)
    sim = cosine_similarity(emb[0:1], emb[1:2])[0][0]
    return float(sim)


if __name__ == "__main__":
    # Небольшая демонстрация при запуске файла напрямую.
    from tokenization_utils import load_tokenizer

    tokenizer = load_tokenizer()
    model = load_model()

    texts = [
        "This movie was absolutely amazing!",
        "Terrible movie, waste of time.",
        "Pretty good, I liked it.",
        "Boring and too long.",
    ]
    embeddings = get_embeddings(texts, tokenizer, model)
    print(f"Embeddings shape: {embeddings.shape}")

    sim1 = similarity("Great movie!", "Amazing film!", tokenizer, model)
    sim2 = similarity("Great movie!", "Terrible film!", tokenizer, model)
    print(f"Сходство похожих: {sim1:.3f}")
    print(f"Сходство разных: {sim2:.3f}")
