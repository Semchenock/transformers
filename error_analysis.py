"""Анализ ошибок модели (День 7).

Разбирает, на чём именно ошибается дообученная модель: False Positive
и False Negative, зависимость от длины текста, уверенность на ошибках,
а также какие примеры оказались трудными сразу для всех трёх моделей
из Дня 6.

Пример:
    from error_analysis import build_error_frame, analyze_errors, save_analysis

    df = build_error_frame(texts, true_labels, preds, probabilities)
    stats = analyze_errors(df)
    save_analysis(df, stats, "error_analysis.txt")
"""

from __future__ import annotations

import numpy as np
import pandas as pd

LABEL_NAMES = {0: "negative", 1: "positive"}


def build_error_frame(
    texts: list[str],
    true_labels: list[int],
    pred_labels: list[int],
    probabilities: list[np.ndarray] | None = None,
) -> pd.DataFrame:
    """Собирает DataFrame с предсказаниями, ошибками и уверенностью.

    Args:
        texts: тексты тестовой выборки.
        true_labels: истинные метки.
        pred_labels: предсказания модели.
        probabilities: массивы вероятностей по классам (опционально).

    Returns:
        DataFrame с колонками text, true_label, pred_label, is_error,
        text_length и (если переданы вероятности) confidence.
    """
    df = pd.DataFrame(
        {
            "text": texts,
            "true_label": true_labels,
            "pred_label": pred_labels,
        }
    )
    df["is_error"] = df["true_label"] != df["pred_label"]
    df["text_length"] = df["text"].str.len()
    df["n_words"] = df["text"].str.split().str.len()

    if probabilities is not None:
        df["confidence"] = [float(np.max(p)) for p in probabilities]

    return df


def analyze_errors(df: pd.DataFrame) -> dict:
    """Считает сводную статистику по ошибкам.

    Returns:
        Словарь со счётчиками FP/FN, средними длинами и (если есть
        колонка confidence) средней уверенностью на верных и ошибочных
        предсказаниях.
    """
    errors = df[df["is_error"]]

    # FP: предсказали positive (1), а было negative (0).
    fp = errors[(errors["pred_label"] == 1) & (errors["true_label"] == 0)]
    fn = errors[(errors["pred_label"] == 0) & (errors["true_label"] == 1)]

    stats = {
        "total": len(df),
        "n_errors": len(errors),
        "n_fp": len(fp),
        "n_fn": len(fn),
        "accuracy": 1 - len(errors) / len(df),
        "mean_len_errors": float(errors["text_length"].mean()),
        "mean_len_all": float(df["text_length"].mean()),
        "mean_words_errors": float(errors["n_words"].mean()),
        "mean_words_all": float(df["n_words"].mean()),
        "fp": fp,
        "fn": fn,
        "errors": errors,
    }

    if "confidence" in df.columns:
        correct = df[~df["is_error"]]
        stats["conf_correct"] = float(correct["confidence"].mean())
        stats["conf_errors"] = float(errors["confidence"].mean())
        # Ошибки с очень высокой уверенностью — самые проблемные.
        stats["n_confident_errors"] = int((errors["confidence"] > 0.9).sum())

    return stats


def length_buckets(df: pd.DataFrame, bins: tuple[int, ...] = (0, 5, 10, 15, 25, 100)):
    """Считает долю ошибок в зависимости от длины текста в словах."""
    labels = [f"{bins[i]}-{bins[i+1]}" for i in range(len(bins) - 1)]
    buckets = pd.cut(df["n_words"], bins=list(bins), labels=labels, right=False)

    return (
        df.groupby(buckets, observed=False)
        .agg(n=("is_error", "size"), errors=("is_error", "sum"))
        .assign(error_rate=lambda d: d["errors"] / d["n"])
    )


def find_hard_examples(
    df: pd.DataFrame, other_predictions: dict[str, list[int]]
) -> pd.DataFrame:
    """Находит примеры, на которых ошиблись все модели сразу.

    Такие примеры обычно указывают не на слабость конкретной модели,
    а на объективно трудный или спорно размеченный текст.

    Args:
        df: результат build_error_frame для основной модели.
        other_predictions: {название модели: предсказания} остальных.

    Returns:
        Подмножество df с примерами, где ошиблись все модели.
    """
    mask = df["is_error"].to_numpy()
    for preds in other_predictions.values():
        mask = mask & (np.array(preds) != df["true_label"].to_numpy())
    return df[mask]


def save_analysis(
    df: pd.DataFrame,
    stats: dict,
    path: str = "error_analysis.txt",
    n_examples: int = 5,
    hard_examples: pd.DataFrame | None = None,
    observations: list[str] | None = None,
) -> None:
    """Сохраняет анализ ошибок в текстовый файл."""
    lines = [
        "=== АНАЛИЗ ОШИБОК ===",
        "",
        f"Всего примеров: {stats['total']}",
        f"Всего ошибок: {stats['n_errors']} ({stats['n_errors']/stats['total']:.1%})",
        f"False Positives (сказали positive, было negative): {stats['n_fp']}",
        f"False Negatives (сказали negative, было positive): {stats['n_fn']}",
        "",
        "--- Длина текстов ---",
        f"Средняя длина ошибочных текстов: {stats['mean_len_errors']:.0f} симв. "
        f"({stats['mean_words_errors']:.1f} слов)",
        f"Средняя длина всех текстов:      {stats['mean_len_all']:.0f} симв. "
        f"({stats['mean_words_all']:.1f} слов)",
    ]

    if "conf_correct" in stats:
        lines += [
            "",
            "--- Уверенность модели ---",
            f"Средняя уверенность на верных предсказаниях:    {stats['conf_correct']:.3f}",
            f"Средняя уверенность на ошибочных предсказаниях: {stats['conf_errors']:.3f}",
            f"Ошибок с уверенностью > 0.9: {stats['n_confident_errors']} "
            f"из {stats['n_errors']}",
        ]

    for title, subset in [
        ("=== ПРИМЕРЫ FALSE POSITIVES (сказали good, а было bad) ===", stats["fp"]),
        ("=== ПРИМЕРЫ FALSE NEGATIVES (сказали bad, а было good) ===", stats["fn"]),
    ]:
        lines += ["", title]
        for _, row in subset.head(n_examples).iterrows():
            conf = f", уверенность {row['confidence']:.3f}" if "confidence" in row else ""
            lines.append(f"\nТекст: {row['text']}")
            lines.append(
                f"Истинный: {LABEL_NAMES[row['true_label']]}, "
                f"Предсказан: {LABEL_NAMES[row['pred_label']]}{conf}"
            )

    if hard_examples is not None and len(hard_examples):
        lines += [
            "",
            "",
            f"=== ТРУДНЫЕ ПРИМЕРЫ (ошиблись все модели): {len(hard_examples)} ===",
        ]
        for _, row in hard_examples.head(n_examples).iterrows():
            lines.append(f"\nТекст: {row['text']}")
            lines.append(f"Истинный: {LABEL_NAMES[row['true_label']]}")

    if observations:
        lines += ["", "", "=== НАБЛЮДЕНИЯ ==="]
        lines += [f"{i}. {obs}" for i, obs in enumerate(observations, 1)]

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    from compare_utils import (
        get_test_split,
        load_baseline,
        load_fine_tuned,
        load_tfidf,
        predict_baseline,
        predict_fine_tuned,
        predict_tfidf,
    )
    from embeddings_utils import load_model
    from tokenization_utils import load_tokenizer

    texts, labels = get_test_split()

    model_ft, tokenizer_ft = load_fine_tuned()
    preds_ft = predict_fine_tuned(texts, model_ft, tokenizer_ft)
    y_pred_ft = [p["prediction"] for p in preds_ft]
    probs_ft = [p["probabilities"] for p in preds_ft]

    df = build_error_frame(texts, labels, y_pred_ft, probs_ft)
    stats = analyze_errors(df)

    # Предсказания остальных моделей — чтобы найти общие трудные примеры.
    clf_base = load_baseline()
    tok_enc, encoder = load_tokenizer(), load_model()
    y_pred_base = [
        p["prediction"] for p in predict_baseline(texts, clf_base, tok_enc, encoder)
    ]
    clf_tfidf, vectorizer = load_tfidf()
    y_pred_tfidf = [
        p["prediction"] for p in predict_tfidf(texts, clf_tfidf, vectorizer)
    ]

    hard = find_hard_examples(
        df, {"baseline": y_pred_base, "tfidf": y_pred_tfidf}
    )

    print(f"Всего ошибок: {stats['n_errors']} из {stats['total']}")
    print(f"FP: {stats['n_fp']}, FN: {stats['n_fn']}")
    print(f"Уверенность: верные {stats['conf_correct']:.3f}, "
          f"ошибки {stats['conf_errors']:.3f}")
    print(f"Ошибок с уверенностью > 0.9: {stats['n_confident_errors']}")
    print(f"Трудных примеров (ошиблись все 3 модели): {len(hard)}")
    print()
    print(length_buckets(df))

    save_analysis(df, stats, hard_examples=hard)
    print("\nСохранено: error_analysis.txt")
