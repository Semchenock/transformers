"""Переиспользуемые утилиты для токенизации (День 1).

Этот модуль собирает всё, что пригодится в следующие дни:
загрузку токенизатора, токенизацию батчей и разбор токенизации текста.

Пример:
    from tokenization_utils import load_tokenizer, tokenize_texts, explain_tokenization

    tokenizer = load_tokenizer("distilbert-base-uncased")
    batch = tokenize_texts(["Great movie!", "Waste of time."], tokenizer)
    explain_tokenization("Transformers are amazing!", tokenizer)
"""

from __future__ import annotations

from transformers import AutoTokenizer, PreTrainedTokenizerBase

DEFAULT_MODEL = "distilbert-base-uncased"


def load_tokenizer(model_name: str = DEFAULT_MODEL) -> PreTrainedTokenizerBase:
    """Загружает токенизатор для указанной модели.

    Args:
        model_name: имя модели на Hugging Face Hub.

    Returns:
        Загруженный токенизатор.
    """
    return AutoTokenizer.from_pretrained(model_name)


def tokenize_texts(
    texts: list[str],
    tokenizer: PreTrainedTokenizerBase,
    max_length: int = 128,
):
    """Токенизирует список текстов в батч тензоров PyTorch.

    Padding и truncation включены, поэтому на выходе получается
    прямоугольный тензор, готовый для подачи в модель.

    Args:
        texts: список строк для токенизации.
        tokenizer: загруженный токенизатор.
        max_length: максимальная длина последовательности.

    Returns:
        BatchEncoding с ключами input_ids и attention_mask (тензоры "pt").
    """
    return tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )


def explain_tokenization(text: str, tokenizer: PreTrainedTokenizerBase) -> dict:
    """Показывает, как текст разбивается на токены, и печатает разбор.

    Args:
        text: исходный текст.
        tokenizer: загруженный токенизатор.

    Returns:
        Словарь с токенами, их ID и количеством — на случай, если
        результат нужен программно, а не только для печати.
    """
    tokens = tokenizer.tokenize(text)
    ids = tokenizer.convert_tokens_to_ids(tokens)

    print(f"Исходный текст: {text}")
    print(f"Токены: {tokens}")
    print(f"IDs: {ids}")
    print(f"Количество: {len(tokens)}")

    return {"tokens": tokens, "ids": ids, "count": len(tokens)}


def describe_special_tokens(tokenizer: PreTrainedTokenizerBase) -> None:
    """Печатает специальные токены токенизатора (CLS/SEP/PAD)."""
    print(f"CLS token: {tokenizer.cls_token} (ID: {tokenizer.cls_token_id})")
    print(f"SEP token: {tokenizer.sep_token} (ID: {tokenizer.sep_token_id})")
    print(f"PAD token: {tokenizer.pad_token} (ID: {tokenizer.pad_token_id})")


if __name__ == "__main__":
    # Небольшая демонстрация при запуске файла напрямую.
    tok = load_tokenizer()
    print(f"vocab_size: {tok.vocab_size}")
    print(f"model_max_length: {tok.model_max_length}\n")

    describe_special_tokens(tok)
    print()
    explain_tokenization("Transformers are amazing!", tok)
