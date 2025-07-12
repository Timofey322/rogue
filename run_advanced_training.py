#!/usr/bin/env python3
"""
🚀 Advanced Training Runner - Продвинутая система обучения моделей
Включает: Stacking, Voting, Deep Learning, ASHA/BOHB, TA-Lib, GPU поддержка
"""

import os
import sys
import time
import psutil
from datetime import datetime

from model_trainer import ModelTrainer

def check_system_resources():
    """
    Проверяет системные ресурсы и дает рекомендации
    """
    print("🔍 Проверка системных ресурсов...")
    
    # CPU
    cpu_count = psutil.cpu_count()
    cpu_percent = psutil.cpu_percent(interval=1)
    print(f"   CPU: {cpu_count} ядер, загрузка: {cpu_percent:.1f}%")
    
    # Память
    memory = psutil.virtual_memory()
    memory_gb = memory.total / (1024**3)
    memory_available_gb = memory.available / (1024**3)
    print(f"   Память: {memory_gb:.1f}GB всего, {memory_available_gb:.1f}GB доступно")
    
    # Диск
    disk = psutil.disk_usage('/')
    disk_gb = disk.total / (1024**3)
    disk_free_gb = disk.free / (1024**3)
    print(f"   Диск: {disk_gb:.1f}GB всего, {disk_free_gb:.1f}GB свободно")
    
    # Рекомендации
    if memory_available_gb < 8:
        print("⚠️  ВНИМАНИЕ: Мало оперативной памяти! Рекомендуется минимум 8GB")
        print("   Система будет использовать ограниченную параллелизацию")
    
    if disk_free_gb < 10:
        print("⚠️  ВНИМАНИЕ: Мало места на диске! Рекомендуется минимум 10GB")
    
    return {
        'cpu_count': cpu_count,
        'memory_available_gb': memory_available_gb,
        'disk_free_gb': disk_free_gb
    }

def estimate_training_time(timeframe, resources):
    """
    Оценивает время обучения на основе ресурсов
    """
    print("⏱️  Оценка времени обучения...")
    
    # Базовое время для каждого таймфрейма (в минутах)
    base_times = {
        '5m': 120,  # 2 часа
        '15m': 90   # 1.5 часа
    }
    
    # Корректировка на основе ресурсов
    time_multiplier = 1.0
    
    if resources['memory_available_gb'] < 8:
        time_multiplier *= 1.5  # Медленнее из-за ограниченной памяти
    elif resources['memory_available_gb'] > 16:
        time_multiplier *= 0.7  # Быстрее с большим объемом памяти
    
    if resources['cpu_count'] < 4:
        time_multiplier *= 1.3  # Медленнее с меньшим количеством ядер
    elif resources['cpu_count'] > 8:
        time_multiplier *= 0.8  # Быстрее с большим количеством ядер
    
    estimated_time = base_times[timeframe] * time_multiplier
    
    print(f"   Базовое время для {timeframe}: {base_times[timeframe]} минут")
    print(f"   Корректировка: {time_multiplier:.2f}x")
    print(f"   Оценка времени: {estimated_time:.0f} минут ({estimated_time/60:.1f} часов)")
    
    return estimated_time

def train_timeframe_advanced(timeframe):
    """
    Обучает модели для конкретного таймфрейма с продвинутыми возможностями
    """
    print(f"\n{'='*70}")
    print(f"🔄 ПРОДВИНУТОЕ ОБУЧЕНИЕ ДЛЯ {timeframe.upper()} ТАЙМФРЕЙМА")
    print(f"{'='*70}")
    
    start_time = time.time()
    
    # Проверяем ресурсы
    resources = check_system_resources()
    estimated_time = estimate_training_time(timeframe, resources)
    
    # Создаем тренер
    trainer = ModelTrainer(timeframe=timeframe)
    
    # Загружаем данные (до 2025 года)
    print(f"📂 Загрузка данных...")
    df = trainer.load_data()
    
    # Создаем признаки
    print(f"🔧 Создание продвинутых признаков...")
    df = trainer.create_features(df)
    
    # Создаем целевую переменную
    print(f"🎯 Создание целевой переменной...")
    df = trainer.create_target(df)
    
    # Подготавливаем данные
    feature_columns = [col for col in df.columns if col not in ['timestamp', 'datetime', 'target', 'target_original']]
    X = df[feature_columns].fillna(0)
    y = df['target']
    
    print(f"📊 Данные подготовлены:")
    print(f"   Период: 2021-07-05 - 2024-12-31")
    print(f"   Признаков: {len(feature_columns)}")
    print(f"   Примеров: {len(X)}")
    print(f"   Класс 0 (шорт): {(y == 0).sum()} ({(y == 0).mean()*100:.1f}%)")
    print(f"   Класс 1 (нейтрально): {(y == 1).sum()} ({(y == 1).mean()*100:.1f}%)")
    print(f"   Класс 2 (лонг): {(y == 2).sum()} ({(y == 2).mean()*100:.1f}%)")
    
    # Продвинутая кросс-валидация с ASHA/BOHB
    print(f"\n⚡ Запуск продвинутой кросс-валидации...")
    print(f"   Алгоритмы: ASHA, BOHB, Hyperband, CMA-ES")
    print(f"   Модели: XGBoost, LightGBM, CatBoost, RandomForest, ExtraTrees, SVM, MLP, DeepNeuralNetwork")
    print(f"   Ансамбли: Voting (Hard/Soft), Stacking")
    
    results, importance = trainer.time_series_cross_validation(X, y, n_splits=5)
    
    # Выводим результаты
    trainer.print_results(results)
    
    # Создаем график
    trainer.create_comparison_plot(results)
    
    # Обучаем финальные модели
    print(f"\n🎯 Обучение финальных моделей...")
    trainer.train_final_models(X, y)
    
    # Сохраняем результаты
    trainer.save_results(results, importance)
    
    end_time = time.time()
    training_time = end_time - start_time
    
    print(f"\n✅ Продвинутое обучение завершено для {timeframe}!")
    print(f"⏱️  Время выполнения: {training_time/60:.1f} минут")
    print(f"📈 Оценка точности: {estimated_time/60:.1f} часов")
    
    return results, importance

def main():
    """
    Основная функция
    """
    print("🚀 ADVANCED TRAINING RUNNER - Продвинутая система обучения")
    print("=" * 70)
    print("✨ ВКЛЮЧАЕТ:")
    print("   🔄 Временную кросс-валидацию с ASHA/BOHB/Hyperband")
    print("   ⚡ Optuna с продвинутыми алгоритмами оптимизации")
    print("   🎯 Продвинутую балансировку классов (SMOTE, ADASYN, BorderlineSMOTE)")
    print("   📊 Автоматический отбор признаков с TA-Lib")
    print("   🧠 Глубокое обучение с PyTorch")
    print("   🎯 Ансамблевые модели (Stacking, Voting)")
    print("   🚀 GPU поддержку и умную параллелизацию")
    print("   💾 Кэширование и оптимизацию памяти")
    print("=" * 70)
    
    # Создаем папки
    os.makedirs('model_results', exist_ok=True)
    os.makedirs('model_results/models', exist_ok=True)
    os.makedirs('model_results/features', exist_ok=True)
    os.makedirs('model_results/validation', exist_ok=True)
    os.makedirs('model_results/optuna', exist_ok=True)
    os.makedirs('model_results/ensemble', exist_ok=True)
    os.makedirs('model_results/deep_learning', exist_ok=True)
    
    start_time = time.time()
    
    # Обучаем для 5M
    print(f"\n🎯 Начинаем обучение для 5M таймфрейма...")
    results_5m, importance_5m = train_timeframe_advanced('5m')
    
    # Обучаем для 15M
    print(f"\n🎯 Начинаем обучение для 15M таймфрейма...")
    results_15m, importance_15m = train_timeframe_advanced('15m')
    
    end_time = time.time()
    total_time = end_time - start_time
    
    print(f"\n{'='*70}")
    print(f"🎉 ПРОДВИНУТОЕ ОБУЧЕНИЕ ЗАВЕРШЕНО!")
    print(f"{'='*70}")
    print(f"⏱️  Общее время: {total_time/60:.1f} минут ({total_time/3600:.1f} часов)")
    print(f"📁 Результаты сохранены в model_results/")
    print(f"\n📊 Архитектура системы:")
    print(f"   🎯 Обучение: 2021-07-05 - 2024-12-31 (с продвинутой кросс-валидацией)")
    print(f"   🧪 Тестирование: 2025-01-01 - 2025-07-02")
    print(f"\n🚀 Следующий шаг: запустите run_backtest.py")
    print(f"\n💡 Новые возможности в этой версии:")
    print(f"   ✅ Продвинутые алгоритмы оптимизации (ASHA, BOHB, Hyperband)")
    print(f"   ✅ Глубокое обучение с PyTorch")
    print(f"   ✅ Ансамблевые модели (Stacking, Voting)")
    print(f"   ✅ TA-Lib технические индикаторы")
    print(f"   ✅ Продвинутая балансировка классов")
    print(f"   ✅ GPU поддержка и умная параллелизация")
    print(f"   ✅ Кэширование и оптимизация памяти")
    print(f"   ✅ Автоматическая оценка времени обучения")

if __name__ == "__main__":
    main() 