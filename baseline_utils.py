"""Baseline-классификация на CLS-эмбеддингах без дообучения трансформера (День 4).

Идея baseline: трансформер используется только как «замороженный» экстрактор
признаков (Дни 1–2), а учится поверх него простая логистическая регрессия.
Это даёт точку отсчёта, с которой потом сравнивают полноценный fine-tuning.

Пример:
    from tokenization_utils import load_tokenizer
    from embeddings_utils import load_model
    from baseline_utils import load_sst2, get_cls_embeddings, train_baseline, save_results

    df = load_sst2(n_samples=3000)
    tokenizer, model = load_tokenizer(), load_model()

    X = get_cls_embeddings(df["text"].tolist(), tokenizer, model)
    result = train_baseline(X, df["label"].tolist())
    save_results("baseline_results.txt", result)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from embeddings_utils import get_embeddings

# Метки SST-2: бинарная тональность.
SST2_LABEL_NAMES = {0: "negative", 1: "positive"}


def load_sst2(n_samples: int | None = 3000, random_state: int = 42) -> pd.DataFrame:
    """Загружает SST-2 и возвращает DataFrame с колонками text/label.

    Берётся стратифицированная подвыборка, чтобы инференс на CPU
    занимал разумное время — полный train это 67k предложений.

    Args:
        n_samples: сколько примеров оставить; None — весь train.
        random_state: seed для воспроизводимости.

    Returns:
        DataFrame с колонками "text" (str) и "label" (int: 0/1).
    """
    from datasets import load_dataset

    ds = load_dataset("stanfordnlp/sst2", split="train")
    df = ds.to_pandas()[["sentence", "label"]].rename(columns={"sentence": "text"})
    df["text"] = df["text"].str.strip()

    if n_samples is not None and n_samples < len(df):
        # stratify сохраняет исходный баланс классов в подвыборке.
        df, _ = train_test_split(
            df,
            train_size=n_samples,
            stratify=df["label"],
            random_state=random_state,
        )
        df = df.reset_index(drop=True)

    return df


def get_cls_embeddings(
    texts: list[str],
    tokenizer: PreTrainedTokenizerBase,
    model: PreTrainedModel,
    batch_size: int = 32,
    max_length: int = 128,
) -> np.ndarray:
    """CLS-эмбеддинги для списка текстов.

    Тонкая обёртка над get_embeddings из Дня 2 — логика та же
    (батчи → torch.no_grad → last_hidden_state[:, 0, :] → np.vstack),
    имя оставлено ради читаемости в контексте Дня 4.
    """
    return get_embeddings(
        texts, tokenizer, model, batch_size=batch_size, max_length=max_length
    )


@dataclass
class BaselineResult:
    """Результат обучения baseline-классификатора."""

    model: LogisticRegression
    f1_macro: float
    report: str
    n_train: int
    n_test: int


def train_baseline(
    X: np.ndarray,
    y: list[int] | np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42,
    max_iter: int = 1000,
) -> BaselineResult:
    """Обучает логистическую регрессию на эмбеддингах и считает метрики.

    Args:
        X: матрица эмбеддингов (n_samples, hidden_size).
        y: метки классов.
        test_size: доля тестовой выборки.
        random_state: seed для воспроизводимости разбиения.
        max_iter: максимум итераций солвера.

    Returns:
        BaselineResult с обученной моделью, macro F1 и текстовым отчётом.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    clf = LogisticRegression(max_iter=max_iter, n_jobs=-1)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    target_names = [SST2_LABEL_NAMES[c] for c in sorted(set(y))]
    report = classification_report(y_test, y_pred, target_names=target_names)
    f1 = f1_score(y_test, y_pred, average="macro")

    return BaselineResult(
        model=clf,
        f1_macro=float(f1),
        report=report,
        n_train=len(X_train),
        n_test=len(X_test),
    )


def save_results(
    path: str,
    result: BaselineResult,
    model_name: str = "distilbert-base-uncased",
    dataset_name: str = "SST-2",
) -> None:
    """Сохраняет метрики baseline в текстовый файл."""
    lines = [
        "Baseline без обучения трансформера (День 4)",
        "=" * 45,
        f"Модель (заморожена):  {model_name}",
        f"Признаки:             CLS-эмбеддинги, last_hidden_state[:, 0, :]",
        f"Классификатор:        LogisticRegression(max_iter=1000)",
        f"Датасет:              {dataset_name}",
        f"Train / test:         {result.n_train} / {result.n_test}",
        "",
        f"macro F1: {result.f1_macro:.4f}",
        "",
        "Classification report:",
        result.report,
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    from embeddings_utils import load_model
    from tokenization_utils import load_tokenizer

    df = load_sst2(n_samples=3000)
    print(f"Датасет: {df.shape}, баланс классов:\n{df['label'].value_counts()}\n")

    tokenizer = load_tokenizer()
    model = load_model()

    print("Извлекаем CLS-эмбеддинги...")
    X = get_cls_embeddings(df["text"].tolist(), tokenizer, model)
    print(f"Embeddings shape: {X.shape}\n")

    result = train_baseline(X, df["label"].tolist())
    print(result.report)
    print(f"macro F1: {result.f1_macro:.4f}")

    save_results("baseline_results.txt", result)
    print("\nСохранено: baseline_results.txt")
