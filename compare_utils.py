"""Инференс и сравнение baseline vs fine-tuned (День 6).

Важное отличие от шаблона задания: там baseline предполагался как
TF-IDF vectorizer + sklearn-классификатор. Наш baseline из Дня 4 устроен
иначе — это замороженный DistilBERT (экстрактор CLS-признаков) плюс
LogisticRegression. Поэтому вместо `vectorizer.transform(texts)`
используется `get_cls_embeddings(texts, ...)`; роль «векторизатора»
играет сам трансформер.

Обе модели оцениваются на одной и той же валидационной выборке
(600 примеров, тот же random_state=42, что в Днях 4 и 5).
"""

from __future__ import annotations

import joblib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from baseline_utils import SST2_LABEL_NAMES, get_cls_embeddings, load_sst2
from embeddings_utils import load_model

LABEL_NAMES = [SST2_LABEL_NAMES[0], SST2_LABEL_NAMES[1]]  # ["negative", "positive"]
FINE_TUNED_PATH = "./fine_tuned_model"
BASELINE_PATH = "baseline_model.pkl"
TFIDF_MODEL_PATH = "tfidf_model.pkl"
TFIDF_VECTORIZER_PATH = "tfidf_vectorizer.pkl"


# --------------------------------------------------------------------------
# Данные: та же выборка, что в Днях 4 и 5
# --------------------------------------------------------------------------

def get_test_split(
    n_samples: int = 3000, test_size: float = 0.2, random_state: int = 42
) -> tuple[list[str], list[int]]:
    """Возвращает (тексты, метки) валидационной выборки.

    Параметры совпадают с Днями 4–5, поэтому это ровно те же 600
    примеров, на которых замерялись обе модели.
    """
    df = load_sst2(n_samples=n_samples, random_state=random_state)
    texts, labels = df["text"].tolist(), df["label"].tolist()

    _, test_texts, _, test_labels = train_test_split(
        texts, labels, test_size=test_size, random_state=random_state, stratify=labels
    )
    return test_texts, test_labels


# --------------------------------------------------------------------------
# Сохранение / загрузка baseline
# --------------------------------------------------------------------------

def fit_and_save_baseline(
    path: str = BASELINE_PATH,
    n_samples: int = 3000,
    random_state: int = 42,
) -> LogisticRegression:
    """Обучает baseline из Дня 4 и сохраняет его через joblib.

    В Дне 4 модель не сохранялась на диск — здесь восполняем это,
    чтобы День 6 мог её просто загрузить.
    """
    from tokenization_utils import load_tokenizer

    df = load_sst2(n_samples=n_samples, random_state=random_state)
    tokenizer, encoder = load_tokenizer(), load_model()

    X = get_cls_embeddings(df["text"].tolist(), tokenizer, encoder)
    y = df["label"].tolist()

    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=random_state
    )

    clf = LogisticRegression(max_iter=1000, n_jobs=-1)
    clf.fit(X_train, y_train)

    joblib.dump(clf, path)
    return clf


def load_baseline(path: str = BASELINE_PATH) -> LogisticRegression:
    """Загружает сохранённый baseline-классификатор."""
    return joblib.load(path)


def fit_and_save_tfidf(
    model_path: str = TFIDF_MODEL_PATH,
    vectorizer_path: str = TFIDF_VECTORIZER_PATH,
    n_samples: int = 3000,
    random_state: int = 42,
) -> tuple[LogisticRegression, TfidfVectorizer]:
    """Обучает классический TF-IDF + LogisticRegression и сохраняет его.

    Это тот baseline, который подразумевался в шаблоне Дня 6 — никакого
    трансформера, только частоты слов и биграмм.

    Важно: векторизатор обучается ТОЛЬКО на train. Если вызвать
    fit_transform на всех данных, словарь и IDF-веса увидят тестовую
    выборку — это утечка, и метрики окажутся завышенными.
    """
    df = load_sst2(n_samples=n_samples, random_state=random_state)
    texts, y = df["text"].tolist(), df["label"].tolist()

    train_texts, _, y_train, _ = train_test_split(
        texts, y, test_size=0.2, random_state=random_state, stratify=y
    )

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),  # униграммы + биграммы: ловит "not good"
        min_df=2,            # выбрасываем слова, встретившиеся один раз
        sublinear_tf=True,
    )
    X_train = vectorizer.fit_transform(train_texts)

    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)

    joblib.dump(clf, model_path)
    joblib.dump(vectorizer, vectorizer_path)
    return clf, vectorizer


def load_tfidf(
    model_path: str = TFIDF_MODEL_PATH, vectorizer_path: str = TFIDF_VECTORIZER_PATH
) -> tuple[LogisticRegression, TfidfVectorizer]:
    """Загружает сохранённые TF-IDF модель и векторизатор."""
    return joblib.load(model_path), joblib.load(vectorizer_path)


def predict_tfidf(
    texts: str | list[str],
    clf: LogisticRegression,
    vectorizer: TfidfVectorizer,
) -> list[dict]:
    """Предсказания TF-IDF модели.

    Это ровно та функция predict_baseline, что была в шаблоне задания —
    с настоящим vectorizer.transform().
    """
    if isinstance(texts, str):
        texts = [texts]

    X = vectorizer.transform(texts)
    predictions = clf.predict(X)
    probs = clf.predict_proba(X) if hasattr(clf, "predict_proba") else None

    return [
        {
            "text": text,
            "prediction": int(predictions[i]),
            "probabilities": probs[i] if probs is not None else None,
        }
        for i, text in enumerate(texts)
    ]


def load_fine_tuned(
    path: str = FINE_TUNED_PATH,
) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """Загружает дообученную модель и её токенизатор."""
    model = AutoModelForSequenceClassification.from_pretrained(path)
    tokenizer = AutoTokenizer.from_pretrained(path)
    model.eval()
    return model, tokenizer


# --------------------------------------------------------------------------
# Предсказания
# --------------------------------------------------------------------------

def predict_fine_tuned(
    texts: str | list[str],
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    batch_size: int = 32,
    max_length: int = 128,
) -> list[dict]:
    """Предсказания дообученной модели.

    В отличие от шаблона задания обрабатывает тексты батчами, а не по
    одному — на 600 примерах это заметно быстрее при том же результате.

    Returns:
        Список словарей с ключами text / prediction / probabilities.
    """
    if isinstance(texts, str):
        texts = [texts]

    results: list[dict] = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )

        with torch.no_grad():
            outputs = model(**inputs)

        probs = torch.nn.functional.softmax(outputs.logits, dim=1)
        preds = torch.argmax(probs, dim=1)

        for text, pred, prob in zip(batch, preds, probs):
            results.append(
                {
                    "text": text,
                    "prediction": int(pred.item()),
                    "probabilities": prob.cpu().numpy(),
                }
            )

    return results


def predict_baseline(
    texts: str | list[str],
    clf: LogisticRegression,
    tokenizer: PreTrainedTokenizerBase,
    encoder: PreTrainedModel,
    batch_size: int = 32,
) -> list[dict]:
    """Предсказания baseline-модели (замороженный энкодер + LogReg).

    Роль «векторизатора» из шаблона задания здесь играет сам DistilBERT:
    тексты превращаются в CLS-эмбеддинги, а уже по ним предсказывает
    логистическая регрессия.
    """
    if isinstance(texts, str):
        texts = [texts]

    X = get_cls_embeddings(texts, tokenizer, encoder, batch_size=batch_size)
    predictions = clf.predict(X)
    probs = clf.predict_proba(X) if hasattr(clf, "predict_proba") else None

    return [
        {
            "text": text,
            "prediction": int(predictions[i]),
            "probabilities": probs[i] if probs is not None else None,
        }
        for i, text in enumerate(texts)
    ]


# --------------------------------------------------------------------------
# Метрики и визуализация
# --------------------------------------------------------------------------

def plot_confusion_matrix(
    y_true: list[int],
    y_pred: list[int],
    title: str,
    save_path: str | None = None,
    show: bool = True,
    ax=None,
):
    """Рисует confusion matrix с подписями классов."""
    cm = confusion_matrix(y_true, y_pred)

    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(7, 5.5))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=LABEL_NAMES,
        yticklabels=LABEL_NAMES,
        cbar=False,
        ax=ax,
    )
    ax.set_title(title)
    ax.set_ylabel("True Label")
    ax.set_xlabel("Predicted Label")

    if own_fig:
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=100)
        if show:
            plt.show()
        else:
            plt.close(fig)

    return cm


def compare_models(y_true: list[int], predictions: dict[str, list[int]]) -> dict:
    """Считает метрики для произвольного числа моделей.

    Args:
        y_true: истинные метки.
        predictions: словарь {название модели: предсказания}.

    Returns:
        Словарь с метриками каждой модели, размером выборки и
        попарным согласием моделей между собой.
    """
    n = len(y_true)
    models = {
        name: {
            "f1": float(f1_score(y_true, y_pred, average="macro")),
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "errors": int(sum(p != t for p, t in zip(y_pred, y_true))),
            "report": classification_report(y_true, y_pred, target_names=LABEL_NAMES),
        }
        for name, y_pred in predictions.items()
    }

    names = list(predictions)
    agreement = {
        f"{a} vs {b}": float(
            np.mean(np.array(predictions[a]) == np.array(predictions[b]))
        )
        for i, a in enumerate(names)
        for b in names[i + 1 :]
    }

    return {"models": models, "n_test": n, "agreement": agreement}


def summary_table(results: dict, baseline_name: str | None = None) -> str:
    """Собирает текстовую таблицу сравнения моделей.

    Args:
        baseline_name: относительно какой модели считать прирост;
            по умолчанию — первая в словаре.
    """
    models = results["models"]
    names = list(models)
    ref = baseline_name or names[0]
    ref_f1 = models[ref]["f1"]

    lines = [
        f"{'Модель':<28}{'macro F1':>10}{'Accuracy':>10}{'Ошибок':>9}{'Δ F1':>10}",
        "-" * 67,
    ]
    for name, m in models.items():
        delta = "—" if name == ref else f"{m['f1'] - ref_f1:+.4f}"
        lines.append(
            f"{name:<28}{m['f1']:>10.4f}{m['accuracy']:>10.4f}"
            f"{m['errors']:>9}{delta:>10}"
        )
    lines.append(f"(Δ F1 — прирост относительно «{ref}»)")
    return "\n".join(lines)


def save_comparison(
    results: dict,
    path: str = "comparison_results.txt",
    baseline_name: str | None = None,
) -> None:
    """Сохраняет сравнение моделей в текстовый файл."""
    lines = [
        "=== Сравнение моделей (День 6) ===",
        "",
        f"Тестовая выборка: {results['n_test']} примеров (SST-2, та же в Днях 4-6)",
        "",
        summary_table(results, baseline_name),
        "",
        "Согласие моделей между собой:",
    ]
    for pair, value in results["agreement"].items():
        lines.append(f"  {pair}: {value:.1%}")

    for name, m in results["models"].items():
        lines += ["", f"--- {name}: classification report ---", m["report"]]

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    from tokenization_utils import load_tokenizer

    test_texts, test_labels = get_test_split()
    print(f"Тестовая выборка: {len(test_texts)} примеров")

    print("Обучаем и сохраняем baseline...")
    clf = fit_and_save_baseline()

    tokenizer_enc, encoder = load_tokenizer(), load_model()
    model_ft, tokenizer_ft = load_fine_tuned()

    print("Обучаем и сохраняем TF-IDF...")
    clf_tfidf, vectorizer = fit_and_save_tfidf()

    print("Предсказания TF-IDF...")
    y_pred_tfidf = [
        p["prediction"] for p in predict_tfidf(test_texts, clf_tfidf, vectorizer)
    ]
    print("Предсказания baseline...")
    y_pred_base = [
        p["prediction"]
        for p in predict_baseline(test_texts, clf, tokenizer_enc, encoder)
    ]
    print("Предсказания fine-tuned...")
    y_pred_ft = [
        p["prediction"] for p in predict_fine_tuned(test_texts, model_ft, tokenizer_ft)
    ]

    results = compare_models(
        test_labels,
        {
            "TF-IDF + LogReg": y_pred_tfidf,
            "Заморожен BERT + LogReg": y_pred_base,
            "Fine-tuned BERT": y_pred_ft,
        },
    )
    save_comparison(results)

    print()
    print(summary_table(results))
    print("\nСогласие моделей:")
    for pair, value in results["agreement"].items():
        print(f"  {pair}: {value:.1%}")
    print("\nСохранено: comparison_results.txt")
