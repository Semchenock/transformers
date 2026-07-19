"""Переиспользуемые утилиты для attention-матриц и их визуализации (День 3).

Опирается на Дни 1–2 и добавляет: загрузку модели с output_attentions,
heatmap одной головы, сетку по всем головам слоя и разбор внимания
к конкретному слову.

Пример:
    from tokenization_utils import load_tokenizer
    from attention_utils import load_model_with_attention, get_attentions, visualize_attention

    tokenizer = load_tokenizer()
    model = load_model_with_attention()

    tokens, attentions = get_attentions("The amazing movie won many awards", tokenizer, model)
    visualize_attention(tokens, attentions, tokenizer, layer=0, head=0)
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import seaborn as sns
import torch
from transformers import AutoModel, PreTrainedModel, PreTrainedTokenizerBase

from tokenization_utils import DEFAULT_MODEL


def load_model_with_attention(model_name: str = DEFAULT_MODEL) -> PreTrainedModel:
    """Загружает модель, которая возвращает attention-веса, в режиме eval.

    Флаг output_attentions=True обязателен — без него outputs.attentions
    будет None.
    """
    model = AutoModel.from_pretrained(model_name, output_attentions=True)
    model.eval()
    return model


def get_attentions(
    text: str,
    tokenizer: PreTrainedTokenizerBase,
    model: PreTrainedModel,
):
    """Прогоняет текст через модель и возвращает (tokens, attentions).

    Args:
        text: исходная строка.
        tokenizer: загруженный токенизатор.
        model: модель, загруженная с output_attentions=True.

    Returns:
        Кортеж (tokens, attentions), где attentions — кортеж тензоров
        по одному на слой, каждый формы [batch, num_heads, seq_len, seq_len].
    """
    tokens = tokenizer(text, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**tokens)

    return tokens, outputs.attentions


def get_token_list(tokens, tokenizer: PreTrainedTokenizerBase) -> list[str]:
    """Возвращает список токенов-строк для подписей на графике."""
    return tokenizer.convert_ids_to_tokens(tokens["input_ids"][0])


def visualize_attention(
    tokens,
    attention,
    tokenizer: PreTrainedTokenizerBase,
    layer: int = 0,
    head: int = 0,
    save_path: str | None = None,
    show: bool = True,
):
    """Рисует heatmap attention для одной головы одного слоя.

    Строки (Queries) — «кто смотрит», столбцы (Keys) — «на кого смотрят».
    Каждая строка суммируется в 1 (softmax по ключам).

    Args:
        tokens: результат токенизации (BatchEncoding).
        attention: кортеж attention-тензоров по слоям.
        tokenizer: токенизатор, нужен для подписей осей.
        layer: номер слоя.
        head: номер головы.
        save_path: если указан — сохранить картинку в этот файл.
        show: вызывать ли plt.show().

    Returns:
        Объект matplotlib Figure.
    """
    attn = attention[layer][0, head]  # [seq_len, seq_len]
    token_list = get_token_list(tokens, tokenizer)

    fig = plt.figure(figsize=(10, 8))
    sns.heatmap(
        attn.cpu().numpy(),
        xticklabels=token_list,
        yticklabels=token_list,
        cmap="viridis",
        cbar=True,
    )
    plt.title(f"Attention - Layer {layer}, Head {head}")
    plt.xlabel("Keys")
    plt.ylabel("Queries")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=100)
    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig


def visualize_all_heads(
    tokens,
    attention,
    tokenizer: PreTrainedTokenizerBase,
    layer: int = 0,
    ncols: int = 4,
    save_path: str | None = None,
    show: bool = True,
):
    """Рисует все головы одного слоя одной сеткой heatmap-ов.

    Компактнее, чем вызывать visualize_attention в цикле: удобно
    сравнивать головы между собой на одном экране.
    """
    num_heads = attention[layer].shape[1]
    nrows = (num_heads + ncols - 1) // ncols
    token_list = get_token_list(tokens, tokenizer)

    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.5 * nrows))
    axes = axes.flatten()

    for head in range(num_heads):
        sns.heatmap(
            attention[layer][0, head].cpu().numpy(),
            xticklabels=token_list,
            yticklabels=token_list,
            cmap="viridis",
            cbar=False,
            ax=axes[head],
        )
        axes[head].set_title(f"Head {head}", fontsize=10)
        axes[head].tick_params(labelsize=7)

    # Прячем лишние оси, если голов меньше, чем ячеек сетки.
    for extra in range(num_heads, len(axes)):
        axes[extra].axis("off")

    fig.suptitle(f"Attention - Layer {layer}, все головы", fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=100)
    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig


def attention_to_word(
    tokens,
    attention,
    tokenizer: PreTrainedTokenizerBase,
    word: str,
    layer: int = -1,
    top_k: int = 5,
) -> list[tuple[str, float]]:
    """Показывает, на какие токены сильнее всего смотрит заданное слово.

    Внимание усредняется по всем головам указанного слоя, затем берётся
    строка запроса (query), соответствующая искомому слову.

    Args:
        word: искомый токен (регистр не важен).
        layer: номер слоя; -1 — последний.
        top_k: сколько верхних токенов вернуть.

    Returns:
        Список пар (токен, вес), отсортированный по убыванию веса.

    Raises:
        ValueError: если слово не найдено среди токенов.
    """
    token_list = get_token_list(tokens, tokenizer)
    word_lower = word.lower()

    if word_lower not in token_list:
        raise ValueError(
            f"Токен {word!r} не найден. Доступные токены: {token_list}"
        )

    idx = token_list.index(word_lower)

    # Усредняем по головам: [num_heads, seq, seq] -> [seq, seq]
    attn_mean = attention[layer][0].mean(dim=0)
    weights = attn_mean[idx].cpu().numpy()

    pairs = sorted(zip(token_list, weights), key=lambda p: p[1], reverse=True)
    return [(tok, float(w)) for tok, w in pairs[:top_k]]


if __name__ == "__main__":
    # Демонстрация при запуске файла напрямую (без показа окон).
    from tokenization_utils import load_tokenizer

    tokenizer = load_tokenizer()
    model = load_model_with_attention()

    text = "The amazing movie won many awards"
    tokens, attentions = get_attentions(text, tokenizer, model)

    print(f"Количество слоёв: {len(attentions)}")
    print(f"Форма attention для слоя 0: {attentions[0].shape}")

    visualize_attention(
        tokens, attentions, tokenizer, layer=0, head=0,
        save_path="attention_layer0_head0.png", show=False,
    )
    print("Сохранено: attention_layer0_head0.png")

    text2 = "This movie was absolutely terrible and I hated it"
    tokens2, attentions2 = get_attentions(text2, tokenizer, model)
    print("\nВнимание слова 'terrible' (последний слой):")
    for tok, weight in attention_to_word(tokens2, attentions2, tokenizer, "terrible"):
        print(f"  {tok:15s} {weight:.3f}")
