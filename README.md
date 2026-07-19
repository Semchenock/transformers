# Sentiment Analysis с трансформерами

Недельный учебный проект: от токенизации до дообученной модели и демо-приложения.
Задача — бинарная классификация тональности на **SST-2** (рецензии на фильмы).

## Результаты

Все три модели оценены на **одной и той же** валидационной выборке (600 примеров,
`random_state=42`), поэтому цифры сравнимы напрямую.

| Модель | macro F1 | Accuracy | Ошибок из 600 |
|---|---|---|---|
| TF-IDF + LogisticRegression | 0.7215 | 0.7367 | 158 |
| Замороженный DistilBERT + LogReg | 0.8630 | 0.8650 | 81 |
| **Fine-tuned DistilBERT** | **0.9204** | **0.9217** | **47** |

Улучшение fine-tuning над замороженным baseline: **+0.0574 F1** (+6.65% относительных).
Каждый шаг примерно вдвое сокращает число ошибок: 158 → 81 → 47.

> Обучение шло на подвыборке в 3000 примеров SST-2 (из 67 349) — чтобы инференс и
> дообучение укладывались в разумное время на CPU.

## Структура проекта

### Переиспользуемые модули

| Файл | Назначение |
|---|---|
| `tokenization_utils.py` | Загрузка токенизатора, токенизация батчей, разбор токенов |
| `embeddings_utils.py` | CLS-эмбеддинги, косинусное сходство |
| `attention_utils.py` | Attention-матрицы, heatmap-визуализация |
| `baseline_utils.py` | Загрузка SST-2, baseline на замороженных эмбеддингах |
| `finetune_utils.py` | `Dataset`, цикл обучения, оценка, сохранение модели |
| `compare_utils.py` | Инференс всех трёх моделей, сравнение метрик |
| `error_analysis.py` | Анализ FP/FN, уверенности, трудных примеров |
| `predictor.py` | Общая логика инференса для демо-приложений |

### Ноутбуки по дням

| День | Ноутбук | Тема |
|---|---|---|
| 1 | `transformers_day01.ipynb` | Архитектура и токенизация |
| 2 | `transformers_day02.ipynb` | Эмбеддинги (hidden states) |
| 3 | `transformers_day03.ipynb` | Attention-матрицы и визуализация |
| 4 | `transformers_day04.ipynb` | Baseline без обучения трансформера |
| 5 | `transformers_day05.ipynb` | Fine-tuning |
| 6 | `transformers_day06.ipynb` | Инференс и сравнение моделей |
| 7 | `transformers_day07.ipynb` | Анализ ошибок и демо |

### Артефакты

- `fine_tuned_model/` — дообученная модель (~250 МБ, в git не коммитится)
- `baseline_results.txt` — метрики baseline (День 4)
- `fine_tuned_results.txt` — метрики дообучения по эпохам (День 5)
- `comparison_results.txt` — сравнение трёх моделей (День 6)
- `error_analysis.txt` — анализ ошибок (День 7)

## Установка

```bash
pip install -r requirements.txt
```

### Windows: если `python` не запускается

Если в PowerShell вы видите

```
Сбой выполнения программы python.exe: Системе не удается найти указанный путь
```

то `python` резолвится в заглушку Microsoft Store
(`...\Microsoft\WindowsApps\python.exe`) — это не Python, а файл-псевдоним,
который падает, когда Store-версия не установлена.

Быстрое решение — использовать лаунчер `py`, который всегда указывает на
настоящую установку:

```powershell
py app.py
py -m pip install -r requirements.txt
py -m uvicorn api:app --reload
```

Постоянное решение: **Параметры → Приложения → Дополнительные параметры
приложений → Псевдонимы выполнения приложения** → выключить `python.exe`
и `python3.exe`, затем перезапустить терминал.

Проверить, куда всё указывает: `Get-Command python -All`

## Воспроизведение результатов

```bash
python baseline_utils.py    # baseline на замороженных эмбеддингах  (~2 мин)
python finetune_utils.py    # fine-tuning, 3 эпохи                  (~15 мин на CPU)
python compare_utils.py     # сравнение трёх моделей                (~3 мин)
python error_analysis.py    # анализ ошибок                          (~3 мин)
```

Fine-tuning должен отработать первым — остальные шаги используют `fine_tuned_model/`.

## Запуск демо

### Gradio

```bash
pip install gradio
python app.py
```

Откройте http://127.0.0.1:7860

### FastAPI

```bash
pip install fastapi uvicorn
uvicorn api:app --reload
```

Документация: http://127.0.0.1:8000/docs

```bash
curl -X POST http://127.0.0.1:8000/predict \
     -H "Content-Type: application/json" \
     -d '{"text": "This movie was great!"}'
```

```json
{
  "label": "Positive",
  "confidence": 0.9947,
  "probabilities": {"Negative": 0.0053, "Positive": 0.9947}
}
```

Есть также `POST /predict_batch` для списка текстов и `GET /health`.

## Использование в коде

```python
from predictor import predict

result = predict("This movie was great!")
print(result["label"])       # Positive
print(result["confidence"])  # 0.9947
```

Или напрямую через transformers:

```python
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

model = AutoModelForSequenceClassification.from_pretrained('./fine_tuned_model')
tokenizer = AutoTokenizer.from_pretrained('./fine_tuned_model')
model.eval()

inputs = tokenizer("Your text here", return_tensors="pt", truncation=True, max_length=128)
with torch.no_grad():
    outputs = model(**inputs)

probs = torch.nn.functional.softmax(outputs.logits, dim=1)
pred = torch.argmax(probs, dim=1).item()  # 0 = Negative, 1 = Positive
```

## Ограничения модели

Стоит знать до того, как применять её к своим данным.

- **Только два класса.** В SST-2 нет нейтрального класса, поэтому нейтральный текст
  вроде «It was okay, nothing special» модель обязана отнести к Negative или Positive.
- **Модель переуверена.** 27 из 47 ошибок сделаны с уверенностью выше 0.9. Средняя
  уверенность на верных ответах 0.953, на ошибочных — 0.870: разрыв слишком мал,
  чтобы отсекать ошибки по порогу. Если вероятности нужны как оценка риска,
  модель требует калибровки (temperature scaling).
- **Домен — рецензии на фильмы.** На отзывах о товарах или технических текстах
  качество будет ниже.
- **Английский язык.** `distilbert-base-uncased` не работает с русским.

## Требования

- Python 3.8+
- transformers, torch, scikit-learn, datasets, pandas, numpy
- matplotlib, seaborn (визуализация)
- gradio (Gradio-демо), fastapi + uvicorn (API)

Точные версии — в `requirements.txt`.
