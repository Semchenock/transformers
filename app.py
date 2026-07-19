"""Gradio-приложение для демонстрации модели (День 7).

Запуск:
    python app.py

Затем откройте http://127.0.0.1:7860
"""

import gradio as gr

from predictor import LABEL_MAP, predict

EXAMPLES = [
    ["This movie was absolutely fantastic!"],
    ["Terrible, waste of my time."],
    ["It was okay, nothing special."],
    ["too good to be bad"],
    ["that most frightening of all movies"],
]


def predict_sentiment(text: str):
    """Обработчик для Gradio.

    Returns:
        Кортеж (метка, словарь вероятностей для gr.Label).
    """
    if not text or not text.strip():
        return "—", {}

    result = predict(text)
    verdict = f"{result['label']} ({result['confidence']:.1%})"
    return verdict, result["probabilities"]


demo = gr.Interface(
    fn=predict_sentiment,
    inputs=gr.Textbox(lines=3, placeholder="Введите текст для анализа...", label="Текст"),
    outputs=[
        gr.Textbox(label="Результат"),
        gr.Label(label="Вероятности", num_top_classes=len(LABEL_MAP)),
    ],
    title="Sentiment Analysis с DistilBERT",
    description=(
        "Модель дообучена на SST-2 (рецензии на фильмы), macro F1 = 0.92. "
        "Классов два: Negative и Positive — нейтрального класса в SST-2 нет, "
        "поэтому нейтральные фразы модель всё равно отнесёт к одному из двух."
    ),
    examples=EXAMPLES,
    flagging_mode="never",
)


if __name__ == "__main__":
    demo.launch()
