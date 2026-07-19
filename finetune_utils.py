"""Fine-tuning трансформера для классификации (День 5).

В отличие от Дня 4, где трансформер был заморожен и учился только
линейный классификатор поверх CLS, здесь обучается вся модель целиком:
градиенты проходят через все 6 слоёв DistilBERT.

Разбиение данных намеренно совпадает с Днём 4 (те же n_samples,
test_size и random_state), чтобы в День 6 сравнение baseline и
fine-tuned модели было честным — на одной и той же валидации.

Пример:
    from tokenization_utils import load_tokenizer
    from finetune_utils import *

    tokenizer = load_tokenizer()
    train_ds, val_ds = build_datasets(tokenizer)
    model, history = run_training(train_ds, val_ds, num_epochs=3)
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from baseline_utils import load_sst2
from tokenization_utils import DEFAULT_MODEL

# У SST-2 в нашей подвыборке максимум ровно 64 токена (медиана 8),
# поэтому 64 ничего не обрезает, но вдвое быстрее, чем 128.
MAX_LENGTH = 64


def set_seed(seed: int = 42) -> None:
    """Фиксирует seed для воспроизводимости обучения."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """Возвращает cuda, если доступна, иначе cpu."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SentimentDataset(Dataset):
    """Датасет текстов с метками для дообучения классификатора."""

    def __init__(
        self,
        texts: list[str],
        labels: list[int],
        tokenizer: PreTrainedTokenizerBase,
        max_length: int = MAX_LENGTH,
    ):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        text = self.texts[idx]
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        # flatten убирает лишнюю batch-размерность: [1, L] -> [L].
        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(label, dtype=torch.long),
        }


def build_datasets(
    tokenizer: PreTrainedTokenizerBase,
    n_samples: int = 3000,
    test_size: float = 0.2,
    random_state: int = 42,
    max_length: int = MAX_LENGTH,
) -> tuple[SentimentDataset, SentimentDataset]:
    """Готовит train/val датасеты из SST-2.

    Параметры разбиения совпадают с Днём 4, поэтому валидационная
    выборка здесь — это ровно та же выборка, на которой замерялся
    baseline.
    """
    df = load_sst2(n_samples=n_samples, random_state=random_state)
    texts, labels = df["text"].tolist(), df["label"].tolist()

    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=test_size, random_state=random_state, stratify=labels
    )

    train_ds = SentimentDataset(train_texts, train_labels, tokenizer, max_length)
    val_ds = SentimentDataset(val_texts, val_labels, tokenizer, max_length)
    return train_ds, val_ds


def load_classifier(
    model_name: str = DEFAULT_MODEL, num_labels: int = 2
) -> PreTrainedModel:
    """Загружает модель с головой классификации.

    Голова (classifier / pre_classifier) инициализируется случайно —
    предупреждение об этом при загрузке ожидаемо, её и предстоит обучить.
    """
    return AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=num_labels
    )


def train_epoch(
    model: PreTrainedModel,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    log_every: int | None = None,
) -> float:
    """Обучает модель одну эпоху. Возвращает средний loss."""
    model.train()
    total_loss = 0.0

    for step, batch in enumerate(dataloader):
        optimizer.zero_grad()

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        # Передаём labels — модель сама посчитает loss (CrossEntropy).
        outputs = model(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels
        )

        loss = outputs.loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if log_every and (step + 1) % log_every == 0:
            print(f"    step {step + 1}/{len(dataloader)}, loss={loss.item():.4f}")

    return total_loss / len(dataloader)


def evaluate(
    model: PreTrainedModel, dataloader: DataLoader, device: torch.device
) -> tuple[float, float]:
    """Оценивает модель. Возвращает (accuracy, macro F1)."""
    model.eval()
    predictions: list[int] = []
    true_labels: list[int] = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=1)

            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(true_labels, predictions)
    f1 = f1_score(true_labels, predictions, average="macro")
    return float(accuracy), float(f1)


@dataclass
class TrainingHistory:
    """История обучения по эпохам."""

    train_loss: list[float] = field(default_factory=list)
    val_accuracy: list[float] = field(default_factory=list)
    val_f1: list[float] = field(default_factory=list)

    @property
    def best_f1(self) -> float:
        return max(self.val_f1) if self.val_f1 else 0.0

    @property
    def final_f1(self) -> float:
        return self.val_f1[-1] if self.val_f1 else 0.0

    @property
    def final_accuracy(self) -> float:
        return self.val_accuracy[-1] if self.val_accuracy else 0.0


def run_training(
    train_ds: SentimentDataset,
    val_ds: SentimentDataset,
    model: PreTrainedModel | None = None,
    num_epochs: int = 3,
    batch_size: int = 16,
    lr: float = 2e-5,
    device: torch.device | None = None,
    seed: int = 42,
) -> tuple[PreTrainedModel, TrainingHistory]:
    """Полный цикл обучения: DataLoader'ы, оптимизатор, эпохи, метрики.

    Note:
        Используется torch.optim.AdamW. Раньше AdamW импортировали как
        `from transformers import AdamW`, но в transformers 5.x он удалён.
    """
    set_seed(seed)
    device = device or get_device()
    model = model or load_classifier()
    model.to(device)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    history = TrainingHistory()

    for epoch in range(num_epochs):
        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_acc, val_f1 = evaluate(model, val_loader, device)

        history.train_loss.append(train_loss)
        history.val_accuracy.append(val_acc)
        history.val_f1.append(val_f1)

        print(f"Epoch {epoch + 1}/{num_epochs}")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Accuracy: {val_acc:.4f}")
        print(f"Val F1: {val_f1:.4f}")
        print("-" * 50)

    return model, history


def save_model(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    path: str = "./fine_tuned_model",
) -> None:
    """Сохраняет модель и токенизатор в папку (~250 МБ)."""
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)


def save_metrics(
    history: TrainingHistory,
    path: str = "fine_tuned_results.txt",
    baseline_f1: float | None = None,
) -> None:
    """Сохраняет метрики дообучения в текстовый файл."""
    lines = [
        "Fine-tuning DistilBERT (День 5)",
        "=" * 45,
        f"Final Validation F1: {history.final_f1:.4f}",
        f"Final Validation Accuracy: {history.final_accuracy:.4f}",
        f"Best Validation F1: {history.best_f1:.4f}",
        "",
        "По эпохам:",
    ]
    for i, (loss, acc, f1) in enumerate(
        zip(history.train_loss, history.val_accuracy, history.val_f1), start=1
    ):
        lines.append(f"  epoch {i}: loss={loss:.4f}  acc={acc:.4f}  f1={f1:.4f}")

    if baseline_f1 is not None:
        delta = history.final_f1 - baseline_f1
        lines += [
            "",
            f"Baseline (День 4, замороженная модель): {baseline_f1:.4f}",
            f"Прирост от fine-tuning: {delta:+.4f}",
        ]

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    from tokenization_utils import load_tokenizer

    tokenizer = load_tokenizer()
    train_ds, val_ds = build_datasets(tokenizer)
    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")
    print(f"Device: {get_device()}\n")

    model, history = run_training(train_ds, val_ds, num_epochs=3)

    save_model(model, tokenizer)
    save_metrics(history, baseline_f1=0.8630)
    print("Сохранено: ./fine_tuned_model, fine_tuned_results.txt")
