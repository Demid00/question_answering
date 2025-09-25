# QA Evaluator - Оценка качества вопросно-ответных систем

Простой и эффективный инструмент для оценки качества моделей вопросно-ответных систем (Question-Answering) на русском языке.

## 🚀 Быстрый старт

### Установка зависимостей

```bash
pip install transformers torch
```

### Запуск демо

```bash
python qa_evaluator.py
```

## 📁 Что реализовано

### Основной функционал
- **Класс `QAEvaluator`** - центральный компонент для работы с QA-моделями
- **Автоматическая оценка** точности ответов модели
- **Поддержка русского языка** через предобученные модели
- **Гибкая настройка** контекста и вопросов

### Ключевые возможности
- Загрузка предобученных моделей HuggingFace
- Оценка качества ответов с метриками точности
- Детальная статистика по уверенности модели
- Обработка ошибок и логирование

## 💻 Использование

### Базовый пример

```python
from qa_evaluator import QAEvaluator, create_sample_test

# Инициализация
evaluator = QAEvaluator()
evaluator.load_model()

# Тестовые данные
context, questions, answers = create_sample_test()
evaluator.set_context(context)
evaluator.set_questions(questions, answers)

# Запуск оценки
results = evaluator.evaluate_answers()
evaluator.print_evaluation_report(results)
```

### Свои данные

```python
evaluator = QAEvaluator()

# Ваш контекст
context = "Ваш текст контекста..."
questions = ["Вопрос 1?", "Вопрос 2?"]
answers = {"Вопрос 1?": "Правильный ответ"}

evaluator.set_context(context)
evaluator.set_questions(questions, answers)

results = evaluator.evaluate_answers()
```

## 🏗️ Структура проекта

```
qa_evaluator.py     # Основной модуль с классом QAEvaluator
```

### Основные компоненты

**Класс QAEvaluator:**
- `load_model()` - загрузка модели
- `set_context()` - установка контекста
- `set_questions()` - добавление вопросов
- `evaluate_answers()` - запуск оценки
- `print_evaluation_report()` - вывод результатов

**Вспомогательные функции:**
- `create_sample_test()` - готовый тестовый пример

## ⚙️ Настройки

### Параметры инициализации
```python
evaluator = QAEvaluator(
    model_name="AlexKay/xlm-roberta-large-qa-multilingual-finedtuned-ru"
)
```

### Метод evaluate_answers()
```python
results = evaluator.evaluate_answers(
    threshold=0.5  # порог уверенности ответа
)
```

## 📊 Пример вывода

```
ОТЧЕТ ОБ ОЦЕНКЕ КАЧЕСТВА QUESTION-ANSWERING
============================================

Модель: AlexKay/xlm-roberta-large-qa-multilingual-finedtuned-ru
Длина контекста: 342 символов
Количество вопросов: 3

1. Вопрос: Кто написал Войну и мир?
   Ответ: 'Льва Николаевича Толстого'
   Score: 0.9633
   ✅ ПРАВИЛЬНО

Точность: 100.0%
Средний score: 72.8%
```

## 🔧 Требования

- Python 3.7+
- transformers >= 4.20.0
- torch >= 1.9.0

## 🐛 Обработка ошибок

Код включает обработку основных ошибок:
- Проверка загрузки модели
- Валидация входных параметров
- Обработка исключений при запросах к модели

---

**Примечание:** Для первого запуска потребуется время на загрузку модели (≈500 МБ). Последующие запуски будут быстрее.
