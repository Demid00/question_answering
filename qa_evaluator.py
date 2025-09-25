import logging
from typing import List, Dict, Tuple, Optional
from transformers import pipeline, Pipeline
import sys

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class QAEvaluator:
    """
    Класс для оценки качества question-answering модели.
    
    Attributes:
        model_name (str): Название предобученной модели
        context (str): Текст контекста для вопросов
        questions (List[str]): Список вопросов
        correct_answers (Dict[str, str]): Словарь правильных ответов
        qa_pipeline (Pipeline): Загруженная модель QA
    """
    
    def __init__(self, model_name: str = "AlexKay/xlm-roberta-large-qa-multilingual-finedtuned-ru"):
        """
        Инициализация оценщика QA.
        
        Args:
            model_name: Название модели HuggingFace для QA
        """
        self.model_name = model_name
        self.context = ""
        self.questions = []
        self.correct_answers = {}
        self.qa_pipeline = None
        
    def load_model(self) -> bool:
        """
        Загрузка модели QA pipeline.
        
        Returns:
            bool: Успешность загрузки модели
        """
        try:
            logger.info(f"Загрузка модели: {self.model_name}")
            self.qa_pipeline = pipeline(
                "question-answering", 
                model=self.model_name,
                tokenizer=self.model_name
            )
            logger.info("Модель успешно загружена")
            return True
        except Exception as e:
            logger.error(f"Ошибка загрузки модели: {e}")
            return False
    
    def set_context(self, context: str) -> None:
        """
        Установка контекста для вопросов.
        
        Args:
            context: Текст контекста
        """
        if not context or not isinstance(context, str):
            raise ValueError("Контекст должен быть непустой строкой")
        
        self.context = context.strip()
        logger.info(f"Установлен контекст длиной {len(self.context)} символов")
    
    def set_questions(self, questions: List[str], correct_answers: Optional[Dict[str, str]] = None) -> None:
        """
        Установка вопросов и правильных ответов.
        
        Args:
            questions: Список вопросов
            correct_answers: Словарь вопрос-ответ для оценки
        """
        if not questions or not isinstance(questions, list):
            raise ValueError("Questions должен быть непустым списком")
        
        self.questions = [q.strip() for q in questions if q.strip()]
        
        if correct_answers:
            self.correct_answers = correct_answers
        else:
            # Автогенерация correct_answers на основе вопросов
            self.correct_answers = {question: "" for question in self.questions}
        
        logger.info(f"Установлено {len(self.questions)} вопросов")
    
    def ask_question(self, question: str, context: Optional[str] = None) -> Dict:
        """
        Задать вопрос модели.
        
        Args:
            question: Вопрос для модели
            context: Контекст (если None, используется установленный контекст)
            
        Returns:
            Dict: Результат модели с ответом и score
            
        Raises:
            RuntimeError: Если модель не загружена
            ValueError: Если вопрос или контекст невалидны
        """
        if not self.qa_pipeline:
            raise RuntimeError("Модель не загружена. Вызовите load_model() сначала.")
        
        if not question or not isinstance(question, str):
            raise ValueError("Вопрос должен быть непустой строкой")
        
        current_context = context if context else self.context
        if not current_context:
            raise ValueError("Контекст не установлен")
        
        try:
            result = self.qa_pipeline(question=question, context=current_context)
            return result
        except Exception as e:
            logger.error(f"Ошибка при запросе к модели: {e}")
            raise
    
    def evaluate_answers(self, threshold: float = 0.5) -> Dict:
        """
        Оценка качества ответов модели.
        
        Args:
            threshold: Порог уверенности для high confidence
            
        Returns:
            Dict: Статистика оценки качества
        """
        if not self.questions:
            raise ValueError("Вопросы не установлены")
        
        results = []
        total_score = 0.0
        
        logger.info("Начало оценки вопросов...")
        
        for i, question in enumerate(self.questions, 1):
            try:
                result = self.ask_question(question)
                results.append(result)
                total_score += result['score']
                
                logger.info(f"Вопрос {i}/{len(self.questions)} обработан: score={result['score']:.4f}")
                
            except Exception as e:
                logger.error(f"Ошибка при обработке вопроса '{question}': {e}")
                # Добавляем пустой результат для сохранения порядка
                results.append({'answer': 'ERROR', 'score': 0.0, 'start': 0, 'end': 0})
        
        # Расчет метрик
        accuracy = self._calculate_accuracy(results)
        average_score = (total_score / len(self.questions)) * 100 if self.questions else 0
        high_confidence_count = sum(1 for r in results if r['score'] > threshold)
        high_confidence_rate = (high_confidence_count / len(self.questions)) * 100 if self.questions else 0
        
        evaluation_stats = {
            'total_questions': len(self.questions),
            'accuracy_percent': accuracy,
            'average_score_percent': average_score,
            'high_confidence_count': high_confidence_count,
            'high_confidence_rate_percent': high_confidence_rate,
            'results': results
        }
        
        return evaluation_stats
    
    def _calculate_accuracy(self, results: List[Dict]) -> float:
        """
        Расчет точности ответов.
        
        Args:
            results: Результаты модели
            
        Returns:
            float: Точность в процентах
        """
        if not self.correct_answers or all(not ans for ans in self.correct_answers.values()):
            logger.warning("Правильные ответы не установлены, точность не может быть рассчитана")
            return 0.0
        
        correct_count = 0
        
        for question, result in zip(self.questions, results):
            if result['answer'] == 'ERROR':
                continue
                
            predicted_answer = result['answer'].strip().lower()
            correct_answer = self.correct_answers.get(question, "").strip().lower()
            
            if not correct_answer:
                continue
                
            # Гибкая проверка совпадения
            is_correct = (
                correct_answer in predicted_answer or 
                predicted_answer in correct_answer or
                any(word in predicted_answer for word in correct_answer.split())
            )
            
            if is_correct:
                correct_count += 1
        
        valid_questions = sum(1 for q in self.questions if self.correct_answers.get(q, "").strip())
        accuracy = (correct_count / valid_questions) * 100 if valid_questions > 0 else 0.0
        
        return accuracy
    
    def print_evaluation_report(self, evaluation_stats: Dict) -> None:
        """
        Печать отчета об оценке.
        
        Args:
            evaluation_stats: Статистика оценки
        """
        print("\n" + "="*60)
        print("ОТЧЕТ ОБ ОЦЕНКЕ КАЧЕСТВА QUESTION-ANSWERING")
        print("="*60)
        
        print(f"\nМодель: {self.model_name}")
        print(f"Длина контекста: {len(self.context)} символов")
        print(f"Количество вопросов: {evaluation_stats['total_questions']}")
        
        print("\n" + "-"*40)
        print("РЕЗУЛЬТАТЫ ПО ВОПРОСАМ:")
        print("-"*40)
        
        for i, (question, result) in enumerate(zip(self.questions, evaluation_stats['results']), 1):
            print(f"\n{i}. Вопрос: {question}")
            
            if result['answer'] == 'ERROR':
                print("   ❌ ОШИБКА: не удалось получить ответ")
                continue
                
            print(f"   Ответ: '{result['answer'].strip()}'")
            print(f"   Score: {result['score']:.4f}")
            print(f"   Позиция: [{result['start']}:{result['end']}]")
            
            # Проверка правильности ответа
            if question in self.correct_answers and self.correct_answers[question]:
                correct_answer = self.correct_answers[question]
                predicted_answer = result['answer'].strip().lower()
                is_correct = (
                    correct_answer.lower() in predicted_answer or 
                    predicted_answer in correct_answer.lower()
                )
                
                status = "✅ ПРАВИЛЬНО" if is_correct else "❌ НЕПРАВИЛЬНО"
                print(f"   {status} (ожидалось: '{correct_answer}')")
        
        print("\n" + "-"*40)
        print("СТАТИСТИКА:")
        print("-"*40)
        
        print(f"Точность: {evaluation_stats['accuracy_percent']:.1f}%")
        print(f"Средний score: {evaluation_stats['average_score_percent']:.1f}%")
        print(f"Высокоуверенные ответы (>0.5): {evaluation_stats['high_confidence_count']}/{evaluation_stats['total_questions']}")
        print(f"Доля высокоуверенных ответов: {evaluation_stats['high_confidence_rate_percent']:.1f}%")
        


def create_sample_test() -> Tuple[str, List[str], Dict[str, str]]:
    """
    Создание тестового примера с вопросами о 'Войне и мире'.
    
    Returns:
        Tuple: (context, questions, correct_answers)
    """
    context = """
    "Война и мир" - роман-эпопея Льва Николаевича Толстого, описывающий русское общество 
    в эпоху войн против Наполеона в 1805-1812 годах. Роман был написан в 1863-1869 годах 
    и является одним из самых известных произведений русской литературы. Главные герои 
    романа - Пьер Безухов, Андрей Болконский и Наташа Ростова.
    """
    
    questions = [
        "Кто написал Войну и мир?",
        "В какие годы был написан роман?",
        "Кто главные герои произведения?",
        "Против кого велась война?",
        "Какой жанр произведения Война и мир?"
    ]
    
    correct_answers = {
        "Кто написал Войну и мир?": "Льва Николаевича Толстого",
        "В какие годы был написан роман?": "1863-1869", 
        "Кто главные герои произведения?": "Пьер Безухов, Андрей Болконский и Наташа Ростова",
        "Против кого велась война?": "против Наполеона",
        "Какой жанр произведения Война и мир?": "роман-эпопея"
    }
    
    return context, questions, correct_answers


def main():
    """Основная функция для демонстрации работы модуля."""
    try:
        # Инициализация оценщика
        evaluator = QAEvaluator()
        
        # Загрузка модели
        if not evaluator.load_model():
            sys.exit(1)
        
        # Создание тестовых данных
        context, questions, correct_answers = create_sample_test()
        
        # Установка контекста и вопросов
        evaluator.set_context(context)
        evaluator.set_questions(questions, correct_answers)
        
        # Оценка качества
        evaluation_stats = evaluator.evaluate_answers(threshold=0.5)
        
        # Печать отчета
        evaluator.print_evaluation_report(evaluation_stats)
        
    except Exception as e:
        logger.error(f"Критическая ошибка в main: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
