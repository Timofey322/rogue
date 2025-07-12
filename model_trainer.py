#!/usr/bin/env python3
"""
🚀 Model Trainer - Система обучения моделей с автоматическим отбором признаков
Использует оптимизированные параметры и справедливую валидацию с окнами
Добавлены: Optuna для максимальной производительности
НОВЫЕ УЛУЧШЕНИЯ: Stacking, Voting, Deep Learning, ASHA/BOHB, TA-Lib, GPU поддержка
"""

import os
import psutil

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# ML библиотеки
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.feature_selection import SelectKBest, f_classif, RFE, SelectFromModel
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

# Градиентный бустинг
import xgboost as xgb
import lightgbm as lgb
import catboost as cb

# Дополнительные модели
from ngboost import NGBClassifier
from sklearn.ensemble import HistGradientBoostingClassifier

# SMOTE для балансировки
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN, SMOTETomek

# Optuna для байесовской оптимизации
import optuna
from optuna.samplers import TPESampler, CmaEsSampler
from optuna.pruners import HyperbandPruner, MedianPruner, SuccessiveHalvingPruner

# Визуализация
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
import json
import pickle
import joblib

# Продвинутые библиотеки
try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False
    print("⚠️ TA-Lib не установлен. Установите: pip install TA-Lib")

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False
    print("⚠️ PyTorch не установлен. Установите: pip install torch")

# GPU поддержка
try:
    import cupy as cp
    GPU_AVAILABLE = True
    print("🚀 GPU поддержка доступна (CuPy)")
except ImportError:
    GPU_AVAILABLE = False
    print("⚠️ GPU поддержка недоступна. Установите: pip install cupy-cuda11x")

# Кэширование
from functools import lru_cache
import hashlib

class DeepNeuralNetwork(nn.Module):
    """
    Продвинутая нейронная сеть для торговли
    """
    def __init__(self, input_size, hidden_sizes=[256, 128, 64], dropout=0.3):
        super(DeepNeuralNetwork, self).__init__()
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, 3))  # 3 класса
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

class ModelTrainer:
    """
    Система обучения моделей с автоматическим отбором признаков
    """
    
    def __init__(self, timeframe='5m'):
        self.timeframe = timeframe
        self.models = {}
        self.feature_importance = {}
        self.results = {}
        self.scaler = RobustScaler()  # Более устойчивый к выбросам
        self.best_params = {}
        
        # Оптимизированные параметры из Optuna (НЕ ИЗМЕНЯЕМ!)
        self.optimal_params = {
            '5m': {'percent': 0.75, 'horizon': 10},
            '15m': {'percent': 0.80, 'horizon': 7}
        }
        
        # Создаем папки для результатов
        os.makedirs('model_results', exist_ok=True)
        os.makedirs('model_results/models', exist_ok=True)
        os.makedirs('model_results/features', exist_ok=True)
        os.makedirs('model_results/validation', exist_ok=True)
        os.makedirs('model_results/optuna', exist_ok=True)
        os.makedirs('model_results/ensemble', exist_ok=True)
        os.makedirs('model_results/deep_learning', exist_ok=True)
        
        # Кэш для ускорения
        self._feature_cache = {}
        self._data_cache = {}
        
        print(f"🚀 Model Trainer инициализирован для {timeframe}")
        print(f"   Оптимальные параметры: {self.optimal_params[timeframe]}")
        print(f"   TA-Lib: {'✅' if TALIB_AVAILABLE else '❌'}")
        print(f"   PyTorch: {'✅' if PYTORCH_AVAILABLE else '❌'}")
        print(f"   GPU: {'✅' if GPU_AVAILABLE else '❌'}")
    
    @lru_cache(maxsize=128)
    def _get_cached_data(self, filename):
        """
        Кэшированная загрузка данных
        """
        return pd.read_csv(filename)
    
    def load_data(self):
        """
        Загружает исторические данные для обучения (2021-2024)
        """
        if self.timeframe == '5m':
            filename = '../data/historical/BTCUSDT_5m_5years_20210705_20250702.csv'
        else:
            filename = '../data/historical/BTCUSDT_15m_5years_20210705_20250702.csv'
        
        print(f"📊 Загрузка данных: {filename}")
        
        # Используем кэш
        cache_key = hashlib.md5(filename.encode()).hexdigest()
        if cache_key in self._data_cache:
            df = self._data_cache[cache_key]
            print("✅ Данные загружены из кэша")
        else:
            df = self._get_cached_data(filename)
            self._data_cache[cache_key] = df
        
        # Обрабатываем timestamp
        if 'datetime' in df.columns:
            df['timestamp'] = pd.to_datetime(df['datetime'])
        else:
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # Используем данные до 2025 года для обучения
        train_end_date = '2025-01-01'
        df_train = df[df['timestamp'] < train_end_date].copy()
        
        print(f"✅ Данные загружены: {len(df)} записей")
        print(f"   Полный период: {df['timestamp'].min()} - {df['timestamp'].max()}")
        print(f"   Обучающие данные: {len(df_train)} записей (до {train_end_date})")
        print(f"   Кросс-валидация: используется TimeSeriesSplit во время обучения")
        
        return df_train
    
    def create_features(self, df):
        """
        Создает комплексный набор признаков с TA-Lib и статистическими признаками
        """
        print("🔧 Создание продвинутых признаков...")
        
        # Базовые признаки цены
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        df['high_low_ratio'] = df['high'] / df['low']
        df['close_open_ratio'] = df['close'] / df['open']
        
        # Волатильность
        df['volatility_5'] = df['returns'].rolling(5).std()
        df['volatility_10'] = df['returns'].rolling(10).std()
        df['volatility_20'] = df['returns'].rolling(20).std()
        
        # TA-Lib индикаторы (если доступны)
        if TALIB_AVAILABLE:
            print("   Добавление TA-Lib индикаторов...")
            
            # RSI
            df['rsi_14'] = talib.RSI(df['close'].values, timeperiod=14)
            df['rsi_7'] = talib.RSI(df['close'].values, timeperiod=7)
            
            # MACD
            macd, macdsignal, macdhist = talib.MACD(df['close'].values)
            df['macd_line'] = macd
            df['macd_signal'] = macdsignal
            df['macd_histogram'] = macdhist
            
            # Moving Averages
            for period in [5, 10, 12, 20, 26, 50]:
                df[f'sma_{period}'] = df['close'].rolling(period).mean()
                df[f'ema_{period}'] = df['close'].ewm(span=period).mean()
                df[f'price_sma_{period}_ratio'] = df['close'] / df[f'sma_{period}']
            
            # Bollinger Bands
            bb_upper, bb_middle, bb_lower = talib.BBANDS(df['close'].values, timeperiod=20)
            df['bb_upper'] = bb_upper
            df['bb_middle'] = bb_middle
            df['bb_lower'] = bb_lower
            df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
            df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
            
            # Stochastic
            for period in [14, 7]:
                slowk, slowd = talib.STOCH(df['high'].values, df['low'].values, df['close'].values, 
                                         fastk_period=period, slowk_period=3, slowd_period=3)
                df[f'stoch_k_{period}'] = slowk
                df[f'stoch_d_{period}'] = slowd
            
            # Williams %R
            for period in [14, 7]:
                df[f'williams_r_{period}'] = talib.WILLR(df['high'].values, df['low'].values, df['close'].values, timeperiod=period)
            
            # CCI
            for period in [20, 10]:
                df[f'cci_{period}'] = talib.CCI(df['high'].values, df['low'].values, df['close'].values, timeperiod=period)
            
            # ATR
            for period in [14, 7]:
                df[f'atr_{period}'] = talib.ATR(df['high'].values, df['low'].values, df['close'].values, timeperiod=period)
                df[f'atr_ratio_{period}'] = df[f'atr_{period}'] / df['close']
            
            # Дополнительные индикаторы
            df['adx'] = talib.ADX(df['high'].values, df['low'].values, df['close'].values, timeperiod=14)
            df['obv'] = talib.OBV(df['close'].values, df['volume'].values)
            df['mfi'] = talib.MFI(df['high'].values, df['low'].values, df['close'].values, df['volume'].values, timeperiod=14)
            
            # Momentum
            for period in [5, 10, 20]:
                df[f'momentum_{period}'] = talib.MOM(df['close'].values, timeperiod=period)
                df[f'roc_{period}'] = talib.ROC(df['close'].values, timeperiod=period)
            
            # Volume indicators
            df['volume_sma'] = talib.SMA(df['volume'].values, timeperiod=20)
            df['volume_ratio'] = df['volume'] / df['volume_sma']
            df['ad'] = talib.AD(df['high'].values, df['low'].values, df['close'].values, df['volume'].values)
            
        else:
            # Fallback без TA-Lib
            print("   Используем базовые индикаторы (TA-Lib недоступен)...")
            
            # RSI
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            df['rsi_14'] = 100 - (100 / (1 + rs))
            df['rsi_7'] = 100 - (100 / (1 + (gain.rolling(7).mean() / loss.rolling(7).mean())))
            
            # MACD
            ema_12 = df['close'].ewm(span=12).mean()
            ema_26 = df['close'].ewm(span=26).mean()
            df['macd_line'] = ema_12 - ema_26
            df['macd_signal'] = df['macd_line'].ewm(span=9).mean()
            df['macd_histogram'] = df['macd_line'] - df['macd_signal']
            
            # Moving Averages
            for period in [5, 10, 12, 20, 26, 50]:
                df[f'sma_{period}'] = df['close'].rolling(period).mean()
                df[f'ema_{period}'] = df['close'].ewm(span=period).mean()
                df[f'price_sma_{period}_ratio'] = df['close'] / df[f'sma_{period}']
            
            # Bollinger Bands
            bb_period = 20
            bb_std = 2
            df['bb_middle'] = df['close'].rolling(bb_period).mean()
            bb_std_dev = df['close'].rolling(bb_period).std()
            df['bb_upper'] = df['bb_middle'] + (bb_std_dev * bb_std)
            df['bb_lower'] = df['bb_middle'] - (bb_std_dev * bb_std)
            df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
            df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
            
            # Volume indicators
            df['volume_sma'] = df['volume'].rolling(20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_sma']
            df['volume_price_trend'] = (df['volume'] * df['returns']).rolling(10).sum()
            
            # Momentum indicators
            for period in [5, 10, 20]:
                df[f'momentum_{period}'] = df['close'].diff(period)
                df[f'roc_{period}'] = ((df['close'] - df['close'].shift(period)) / df['close'].shift(period) * 100)
            
            # Stochastic
            for period in [14, 7]:
                lowest_low = df['low'].rolling(period).min()
                highest_high = df['high'].rolling(period).max()
                df[f'stoch_k_{period}'] = 100 * ((df['close'] - lowest_low) / (highest_high - lowest_low))
                df[f'stoch_d_{period}'] = df[f'stoch_k_{period}'].rolling(3).mean()
            
            # Williams %R
            for period in [14, 7]:
                highest_high = df['high'].rolling(period).max()
                lowest_low = df['low'].rolling(period).min()
                df[f'williams_r_{period}'] = -100 * ((highest_high - df['close']) / (highest_high - lowest_low))
            
            # CCI
            for period in [20, 10]:
                typical_price = (df['high'] + df['low'] + df['close']) / 3
                sma_tp = typical_price.rolling(period).mean()
                mad = typical_price.rolling(period).apply(lambda x: np.mean(np.abs(x - x.mean())) if len(x) > 0 else 0)
                df[f'cci_{period}'] = (typical_price - sma_tp) / (0.015 * mad)
            
            # ATR
            for period in [14, 7]:
                high_low = df['high'] - df['low']
                high_close = np.abs(df['high'] - df['close'].shift())
                low_close = np.abs(df['low'] - df['close'].shift())
                true_range = np.maximum(high_low, np.maximum(high_close, low_close))
                df[f'atr_{period}'] = true_range.rolling(period).mean()
                df[f'atr_ratio_{period}'] = df[f'atr_{period}'] / df['close']
        
        # Статистические признаки
        print("   Добавление статистических признаков...")
        
        # Rolling statistics
        for period in [5, 10, 20]:
            df[f'returns_mean_{period}'] = df['returns'].rolling(period).mean()
            df[f'returns_std_{period}'] = df['returns'].rolling(period).std()
            df[f'returns_skew_{period}'] = df['returns'].rolling(period).skew()
            df[f'returns_kurt_{period}'] = df['returns'].rolling(period).kurt()
            df[f'returns_var_{period}'] = df['returns'].rolling(period).var()
            df[f'returns_median_{period}'] = df['returns'].rolling(period).median()
            
            # Volume statistics
            df[f'volume_mean_{period}'] = df['volume'].rolling(period).mean()
            df[f'volume_std_{period}'] = df['volume'].rolling(period).std()
            df[f'volume_skew_{period}'] = df['volume'].rolling(period).skew()
        
        # Lag features
        for lag in [1, 2, 3, 5, 10]:
            df[f'returns_lag_{lag}'] = df['returns'].shift(lag)
            df[f'volume_lag_{lag}'] = df['volume'].shift(lag)
            df[f'rsi_lag_{lag}'] = df['rsi_14'].shift(lag)
            df[f'close_lag_{lag}'] = df['close'].shift(lag)
        
        # Временные признаки
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['month'] = df['timestamp'].dt.month
        df['day_of_month'] = df['timestamp'].dt.day
        
        # Циклические признаки
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        # Продвинутые признаки
        print("   Добавление продвинутых признаков...")
        
        # Cross-over признаки
        df['sma_cross_5_20'] = (df['sma_5'] > df['sma_20']).astype(int)
        df['ema_cross_12_26'] = (df['ema_12'] > df['ema_26']).astype(int)
        df['price_above_sma_20'] = (df['close'] > df['sma_20']).astype(int)
        
        # Волатильность признаков
        df['volatility_ratio_5_20'] = df['volatility_5'] / df['volatility_20']
        df['volatility_change'] = df['volatility_10'].pct_change()
        
        # Momentum признаки
        df['momentum_acceleration'] = df['momentum_5'].diff()
        df['rsi_momentum'] = df['rsi_14'].diff()
        df['macd_momentum'] = df['macd_line'].diff()
        
        # Volume-price признаки
        df['volume_price_correlation'] = df['volume'].rolling(10).corr(df['close'])
        df['volume_returns_correlation'] = df['volume'].rolling(10).corr(df['returns'])
        
        # Support/Resistance признаки
        df['support_level'] = df['low'].rolling(20).min()
        df['resistance_level'] = df['high'].rolling(20).max()
        df['price_to_support'] = (df['close'] - df['support_level']) / df['close']
        df['price_to_resistance'] = (df['resistance_level'] - df['close']) / df['close']
        
        print(f"✅ Создано {len(df.columns)} признаков")
        return df
    
    def create_target(self, df):
        """
        Создает целевую переменную с 3 классами - ПРЯМАЯ СВЯЗЬ с торговыми параметрами
        Модель учится предсказывать именно те движения, которые нужны для торговли
        """
        print("🎯 Создание целевой переменной с прямой связью с торговлей...")
        
        # ПРЯМАЯ СВЯЗЬ: используем те же параметры что и для торговли
        if self.timeframe == '5m':
            horizon = 10
            tp_percent = 0.75  # Take Profit = целевая переменная
        else:
            horizon = 7
            tp_percent = 0.80  # Take Profit = целевая переменная
        
        print(f"✅ ПРЯМАЯ СВЯЗЬ с торговыми параметрами:")
        print(f"   Таймфрейм: {self.timeframe}")
        print(f"   Горизонт: {horizon} свечей")
        print(f"   Take Profit = Целевая переменная: {tp_percent}%")
        print(f"   Модель учится предсказывать именно {tp_percent}% движения!")
        
        # Рассчитываем максимальное движение цены в горизонте
        future_high = df['high'].rolling(horizon, min_periods=1).max().shift(-horizon)
        future_low = df['low'].rolling(horizon, min_periods=1).min().shift(-horizon)
        
        # Рассчитываем потенциальную прибыль и убыток
        potential_profit_long = (future_high - df['close']) / df['close'] * 100
        potential_profit_short = (df['close'] - future_low) / df['close'] * 100
        
        # ПРЯМАЯ СВЯЗЬ: используем tp_percent как единственный порог
        tp_threshold = tp_percent  # Take Profit = целевая переменная
        sl_threshold = tp_percent * 0.67  # Stop Loss (1/1.5 от TP)
        
        df['target'] = 0  # По умолчанию нейтрально
        
        # Сигнал на лонг: потенциальная прибыль >= TP И потенциальный убыток <= SL
        long_condition = (
            (potential_profit_long >= tp_threshold) &
            (potential_profit_short <= sl_threshold)
        )
        df.loc[long_condition, 'target'] = 1
        
        # Сигнал на шорт: потенциальная прибыль по шорту >= TP И потенциальный убыток <= SL
        short_condition = (
            (potential_profit_short >= tp_threshold) &
            (potential_profit_long <= sl_threshold)
        )
        df.loc[short_condition, 'target'] = -1
        
        # Убираем последние записи где нет будущих данных
        df = df.iloc[:-horizon].copy()
        
        # Преобразуем метки классов для совместимости с моделями: -1->0, 0->1, 1->2
        df['target_original'] = df['target'].copy()  # Сохраняем оригинальные метки
        df['target'] = df['target'].map({-1: 0, 0: 1, 1: 2})  # Преобразуем для обучения
        
        # Статистика классов
        long_samples = (df['target_original'] == 1).sum()
        short_samples = (df['target_original'] == -1).sum()
        neutral_samples = (df['target_original'] == 0).sum()
        total_samples = len(df)
        
        # Сохраняем статистику в атрибуты класса
        self.long_samples = long_samples
        self.short_samples = short_samples
        self.neutral_samples = neutral_samples
        self.total_samples = total_samples
        
        print(f"✅ Целевая переменная создана (ПРЯМАЯ СВЯЗЬ):")
        print(f"   Take Profit = Целевая переменная: {tp_threshold:.2f}% за {horizon} свечей")
        print(f"   Stop Loss: {sl_threshold:.2f}%")
        print(f"   Соотношение TP/SL: 1:1.5")
        print(f"")
        print(f"📊 Статистика сигналов:")
        print(f"   Лонг сигналы: {long_samples} ({long_samples/total_samples*100:.1f}%)")
        print(f"   Шорт сигналы: {short_samples} ({short_samples/total_samples*100:.1f}%)")
        print(f"   Нейтральные: {neutral_samples} ({neutral_samples/total_samples*100:.1f}%)")
        print(f"   Всего примеров: {total_samples}")
        print(f"")
        print(f"🎯 Модель учится предсказывать именно {tp_threshold}% движения!")
        print(f"📈 Метки для обучения: 0 (шорт), 1 (нейтрально), 2 (лонг)")
        
        # Сохраняем параметры
        self.target_params = {
            'horizon': horizon,
            'tp_threshold': tp_threshold,
            'sl_threshold': sl_threshold,
            'tp_sl_ratio': 1.5
        }
        
        return df
    
    def select_features(self, X, y, method='kbest', n_features=50):
        """
        Автоматический отбор признаков
        """
        print(f"🔍 Отбор признаков методом {method}...")
        
        if method == 'kbest':
            selector = SelectKBest(score_func=f_classif, k=n_features)
        elif method == 'rfe':
            estimator = RandomForestClassifier(n_estimators=100, random_state=42)
            selector = RFE(estimator=estimator, n_features_to_select=n_features)
        else:
            return X, list(X.columns)
        
        X_selected = selector.fit_transform(X, y)
        selected_features = X.columns[selector.get_support()].tolist()
        
        print(f"✅ Отобрано {len(selected_features)} признаков из {len(X.columns)}")
        
        return X_selected, selected_features
    
    def get_models_for_optuna(self):
        """
        Возвращает модели с параметрами для Optuna оптимизации (поддержка 3 классов)
        Включает: продвинутые модели, ансамбли, глубокое обучение
        """
        models = {
            # Градиентный бустинг с продвинутыми параметрами
            'XGBoost': {
                'model': xgb.XGBClassifier(random_state=42, eval_metric='mlogloss', use_label_encoder=False, objective='multi:softprob', num_class=3, tree_method='gpu_hist' if GPU_AVAILABLE else 'hist'),
                'params': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [4, 6, 8],
                    'learning_rate': [0.05, 0.1, 0.2],
                    'subsample': [0.8, 0.9, 1.0],
                    'colsample_bytree': [0.8, 0.9, 1.0],
                    'reg_alpha': [0.001, 0.01, 0.1, 1.0],
                    'reg_lambda': [0.001, 0.01, 0.1, 1.0],
                    'min_child_weight': [1, 3, 5, 7]
                }
            },
            'LightGBM': {
                'model': lgb.LGBMClassifier(random_state=42, verbose=-1, objective='multiclass', device='gpu' if GPU_AVAILABLE else 'cpu'),
                'params': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [4, 6, 8],
                    'learning_rate': [0.05, 0.1, 0.2],
                    'subsample': [0.8, 0.9, 1.0],
                    'colsample_bytree': [0.8, 0.9, 1.0],
                    'reg_alpha': [0.001, 0.01, 0.1, 1.0],
                    'reg_lambda': [0.001, 0.01, 0.1, 1.0],
                    'min_child_samples': [10, 20, 50, 100]
                }
            },
            'CatBoost': {
                'model': cb.CatBoostClassifier(random_state=42, verbose=False, loss_function='MultiClass', task_type='GPU' if GPU_AVAILABLE else 'CPU'),
                'params': {
                    'iterations': [100, 200, 300],
                    'depth': [4, 6, 8],
                    'learning_rate': [0.05, 0.1, 0.2],
                    'l2_leaf_reg': [1, 3, 5, 10],
                    'border_count': [32, 64, 128]
                }
            },
            
            # Ансамблевые методы с продвинутыми параметрами
            'RandomForest': {
                'model': RandomForestClassifier(random_state=42),
                'params': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [8, 10, 12, 15],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'max_features': ['sqrt', 'log2', None]
                }
            },
            'ExtraTrees': {
                'model': ExtraTreesClassifier(random_state=42),
                'params': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [8, 10, 12, 15],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'max_features': ['sqrt', 'log2', None]
                }
            },
            'HistGradientBoosting': {
                'model': HistGradientBoostingClassifier(random_state=42),
                'params': {
                    'max_iter': [100, 200, 300],
                    'max_depth': [4, 6, 8],
                    'learning_rate': [0.05, 0.1, 0.2],
                    'l2_regularization': [0.1, 0.5, 1.0],
                    'min_samples_leaf': [10, 20, 50]
                }
            },
            
            # Нейронная сеть с продвинутыми параметрами
            'MLP': {
                'model': MLPClassifier(max_iter=1000, random_state=42),
                'params': {
                    'hidden_layer_sizes': [(50,), (100,), (100, 50), (200, 100), (200, 100, 50)],
                    'alpha': [0.001, 0.01, 0.1],
                    'learning_rate_init': [0.001, 0.01, 0.1]
                }
            }
        }
        
        # Добавляем продвинутые модели если доступны
        if PYTORCH_AVAILABLE:
            models['DeepNeuralNetwork'] = {
                'model': 'custom',  # Специальная обработка
                'params': {
                    'hidden_sizes': [[128, 64], [256, 128, 64], [512, 256, 128, 64]],
                    'dropout': [0.2, 0.3, 0.4],
                    'learning_rate': [0.001, 0.01],
                    'batch_size': [32, 64, 128]
                }
            }
        
        return models
    
    def create_ensemble_models(self, base_models):
        """
        Создает ансамблевые модели (Stacking, Voting)
        """
        print("🎯 Создание ансамблевых моделей...")
        
        ensemble_models = {}
        
        # Voting Classifier (Hard Voting)
        voting_hard = VotingClassifier(
            estimators=[(name, model) for name, model in base_models.items()],
            voting='hard'
        )
        ensemble_models['VotingHard'] = {
            'model': voting_hard,
            'params': {}
        }
        
        # Voting Classifier (Soft Voting)
        voting_soft = VotingClassifier(
            estimators=[(name, model) for name, model in base_models.items()],
            voting='soft'
        )
        ensemble_models['VotingSoft'] = {
            'model': voting_soft,
            'params': {}
        }
        
        # Stacking Classifier
        estimators = [(name, model) for name, model in base_models.items()]
        stacking = StackingClassifier(
            estimators=estimators,
            final_estimator=LogisticRegression(random_state=42),
            cv=3
        )
        ensemble_models['Stacking'] = {
            'model': stacking,
            'params': {}
        }
        
        print(f"✅ Создано {len(ensemble_models)} ансамблевых моделей")
        return ensemble_models
    
    def create_model_with_params(self, model_name, params):
        """
        Создает модель с заданными параметрами
        """
        try:
            if model_name == 'XGBoost':
                return xgb.XGBClassifier(**params, random_state=42, eval_metric='mlogloss', use_label_encoder=False, objective='multi:softprob', num_class=3, tree_method='gpu_hist' if GPU_AVAILABLE else 'hist')
            elif model_name == 'LightGBM':
                return lgb.LGBMClassifier(**params, random_state=42, verbose=-1, objective='multiclass', device='gpu' if GPU_AVAILABLE else 'cpu')
            elif model_name == 'CatBoost':
                return cb.CatBoostClassifier(**params, random_state=42, verbose=False, loss_function='MultiClass', task_type='GPU' if GPU_AVAILABLE else 'CPU')
            elif model_name == 'RandomForest':
                # Если bootstrap=False, удаляем max_samples из params
                if 'bootstrap' in params and not params['bootstrap']:
                    params.pop('max_samples', None)
                return RandomForestClassifier(**params, random_state=42)
            elif model_name == 'ExtraTrees':
                # Если bootstrap=False, удаляем max_samples из params
                if 'bootstrap' in params and not params['bootstrap']:
                    params.pop('max_samples', None)
                return ExtraTreesClassifier(**params, random_state=42)
            elif model_name == 'HistGradientBoosting':
                return HistGradientBoostingClassifier(**params, random_state=42)
            elif model_name == 'MLP':
                # Убираем max_iter из params если он там есть, чтобы избежать дублирования
                if 'max_iter' in params:
                    del params['max_iter']
                return MLPClassifier(**params, max_iter=500, random_state=42)
            else:
                raise ValueError(f"Неизвестная модель: {model_name}")
        except Exception as e:
            print(f"⚠️ Ошибка создания модели {model_name}: {e}")
            print(f"   Параметры: {params}")
            import traceback
            traceback.print_exc()
            return None
    
    def optuna_objective(self, trial, X_train, X_val, y_train, y_val, model_name):
        """
        Целевая функция для Optuna с продвинутыми алгоритмами (поддержка всех моделей)
        Улучшено для более тщательного обучения с увеличенной целевой переменной
        """
        try:
            if model_name == 'XGBoost':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 200, 1000),  # Увеличено
                    'max_depth': trial.suggest_int('max_depth', 4, 12),  # Расширен диапазон
                    'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.3, log=True),  # Более мелкие значения
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                    'reg_alpha': trial.suggest_float('reg_alpha', 0.001, 20, log=True),  # Расширен диапазон
                    'reg_lambda': trial.suggest_float('reg_lambda', 0.001, 20, log=True),  # Расширен диапазон
                    'min_child_weight': trial.suggest_int('min_child_weight', 1, 15),  # Расширен диапазон
                    'gamma': trial.suggest_float('gamma', 0, 5),  # Добавлен gamma
                    'scale_pos_weight': trial.suggest_float('scale_pos_weight', 0.5, 3.0)  # Добавлен для несбалансированных классов
                }
            elif model_name == 'LightGBM':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 200, 1000),  # Увеличено
                    'max_depth': trial.suggest_int('max_depth', 4, 12),  # Расширен диапазон
                    'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.3, log=True),  # Более мелкие значения
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                    'reg_alpha': trial.suggest_float('reg_alpha', 0.001, 20, log=True),  # Расширен диапазон
                    'reg_lambda': trial.suggest_float('reg_lambda', 0.001, 20, log=True),  # Расширен диапазон
                    'min_child_samples': trial.suggest_int('min_child_samples', 10, 200),  # Расширен диапазон
                    'min_split_gain': trial.suggest_float('min_split_gain', 0, 1),  # Добавлен
                    'class_weight': trial.suggest_categorical('class_weight', ['balanced', None])  # Добавлен
                }
            elif model_name == 'CatBoost':
                params = {
                    'iterations': trial.suggest_int('iterations', 200, 1000),  # Увеличено
                    'depth': trial.suggest_int('depth', 4, 12),  # Расширен диапазон
                    'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.3, log=True),  # Более мелкие значения
                    'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 30, log=True),  # Расширен диапазон
                    'border_count': trial.suggest_categorical('border_count', [32, 64, 128, 254]),
                    'bagging_temperature': trial.suggest_float('bagging_temperature', 0, 2),  # Расширен диапазон
                    'random_strength': trial.suggest_float('random_strength', 0, 10),  # Добавлен
                    'class_weights': trial.suggest_categorical('class_weights', [None, 'balanced'])  # Добавлен
                }
            elif model_name == 'RandomForest':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 200, 1000),  # Увеличено
                    'max_depth': trial.suggest_int('max_depth', 10, 25),  # Расширен диапазон
                    'min_samples_split': trial.suggest_int('min_samples_split', 2, 30),  # Расширен диапазон
                    'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 15),  # Расширен диапазон
                    'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
                    'class_weight': trial.suggest_categorical('class_weight', ['balanced', 'balanced_subsample', None]),  # Добавлен
                    'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),  # Добавлен
                    'max_samples': trial.suggest_float('max_samples', 0.5, 1.0)  # Добавлен
                }
            elif model_name == 'ExtraTrees':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 200, 1000),  # Увеличено
                    'max_depth': trial.suggest_int('max_depth', 10, 25),  # Расширен диапазон
                    'min_samples_split': trial.suggest_int('min_samples_split', 2, 30),  # Расширен диапазон
                    'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 15),  # Расширен диапазон
                    'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
                    'class_weight': trial.suggest_categorical('class_weight', ['balanced', 'balanced_subsample', None]),  # Добавлен
                    'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),  # Добавлен
                    'max_samples': trial.suggest_float('max_samples', 0.5, 1.0)  # Добавлен
                }
            elif model_name == 'HistGradientBoosting':
                params = {
                    'max_iter': trial.suggest_int('max_iter', 200, 1000),  # Увеличено
                    'max_depth': trial.suggest_int('max_depth', 4, 12),  # Расширен диапазон
                    'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.3, log=True),  # Более мелкие значения
                    'l2_regularization': trial.suggest_float('l2_regularization', 0.1, 5.0, log=True),  # Расширен диапазон
                    'min_samples_leaf': trial.suggest_int('min_samples_leaf', 10, 200),  # Расширен диапазон
                    'class_weight': trial.suggest_categorical('class_weight', ['balanced', None])  # Добавлен
                }
            elif model_name == 'MLP':
                params = {
                    'hidden_layer_sizes': trial.suggest_categorical('hidden_layer_sizes', 
                        [(100,), (200,), (100, 50), (200, 100), (200, 100, 50), (300, 150, 75), (500, 250, 125)]),  # Добавлены большие сети
                    'alpha': trial.suggest_float('alpha', 0.0001, 0.1, log=True),  # Расширен диапазон
                    'learning_rate_init': trial.suggest_float('learning_rate_init', 0.0001, 0.1, log=True),  # Расширен диапазон
                    'max_iter': trial.suggest_int('max_iter', 500, 2000),  # Добавлен
                    'early_stopping': trial.suggest_categorical('early_stopping', [True, False]),  # Добавлен
                    'validation_fraction': trial.suggest_float('validation_fraction', 0.1, 0.2)  # Добавлен
                }
            elif model_name == 'DeepNeuralNetwork':
                params = {
                    'hidden_sizes': trial.suggest_categorical('hidden_sizes', 
                        [[128, 64], [256, 128, 64], [512, 256, 128, 64], [1024, 512, 256, 128], [2048, 1024, 512, 256]]),  # Добавлены большие сети
                    'dropout': trial.suggest_float('dropout', 0.1, 0.6),  # Расширен диапазон
                    'learning_rate': trial.suggest_float('learning_rate', 0.0001, 0.01, log=True),  # Расширен диапазон
                    'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64, 128, 256]),  # Добавлены варианты
                    'epochs': trial.suggest_int('epochs', 50, 200)  # Добавлен параметр epochs
                }
            else:
                raise ValueError(f"Неизвестная модель: {model_name}")
            
            # Создаем модель
            if model_name == 'DeepNeuralNetwork':
                model = self.create_deep_learning_model(X_train.shape[1], params)
            else:
                model = self.create_model_with_params(model_name, params)
            
            # Проверяем, что модель создана
            if model is None:
                print(f"⚠️ Ошибка: модель {model_name} не создана")
                return 0.0
            
            # Используем естественную балансировку (без SMOTE)
            X_train_balanced, y_train_balanced = X_train, y_train
            
            # Обучение с early stopping для некоторых моделей
            if model_name in ['XGBoost', 'LightGBM', 'CatBoost']:
                # Early stopping для градиентного бустинга
                eval_set = [(X_val, y_val)]
                if model_name == 'XGBoost':
                    model.fit(X_train_balanced, y_train_balanced, 
                             eval_set=eval_set, early_stopping_rounds=50, verbose=False)  # Увеличено с 20 до 50
                elif model_name == 'LightGBM':
                    model.fit(X_train_balanced, y_train_balanced,
                             eval_set=eval_set, callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])  # Увеличено с 20 до 50
                elif model_name == 'CatBoost':
                    model.fit(X_train_balanced, y_train_balanced,
                             eval_set=(X_val, y_val), early_stopping_rounds=50, verbose=False)  # Увеличено с 20 до 50
            elif model_name == 'DeepNeuralNetwork':
                # Специальная обработка для PyTorch
                score = self.train_deep_learning_model(model, X_train_balanced, X_val, y_train_balanced, y_val, params)
                return score
            else:
                # Обычное обучение
                model.fit(X_train_balanced, y_train_balanced)
            
            # Предсказания
            if model_name == 'DeepNeuralNetwork':
                y_pred = self.predict_deep_learning(model, X_val)
            else:
                y_pred = model.predict(X_val)
            
            # Проверяем, что предсказания не пустые
            if len(y_pred) == 0:
                print(f"⚠️ Ошибка: пустые предсказания для {model_name}")
                return 0.0
            
            # Оценка с F1-score (лучше для несбалансированных классов)
            score = f1_score(y_val, y_pred, average='weighted')
            
            # Проверяем, что score не nan
            if np.isnan(score):
                print(f"⚠️ Ошибка: nan score для {model_name}")
                return 0.0
            
            return score
            
        except Exception as e:
            print(f"⚠️ Ошибка в optuna_objective для {model_name}: {e}")
            import traceback
            traceback.print_exc()
            return 0.0
    
    def create_deep_learning_model(self, input_size, params):
        """
        Создает модель глубокого обучения
        """
        if not PYTORCH_AVAILABLE:
            raise ValueError("PyTorch не установлен")
        
        model = DeepNeuralNetwork(
            input_size=input_size,
            hidden_sizes=params['hidden_sizes'],
            dropout=params['dropout']
        )
        
        # GPU если доступен
        if torch.cuda.is_available():
            model = model.cuda()
        
        return model
    
    def train_deep_learning_model(self, model, X_train, X_val, y_train, y_val, params):
        """
        Обучает модель глубокого обучения
        """
        if not PYTORCH_AVAILABLE:
            return 0.0
        
        try:
            # Преобразуем данные в numpy массивы
            if hasattr(X_train, 'values'):
                X_train = X_train.values
            elif hasattr(X_train, 'numpy'):
                X_train = X_train.numpy()
            else:
                X_train = np.array(X_train)
                
            if hasattr(X_val, 'values'):
                X_val = X_val.values
            elif hasattr(X_val, 'numpy'):
                X_val = X_val.numpy()
            else:
                X_val = np.array(X_val)
                
            if hasattr(y_train, 'values'):
                y_train = y_train.values
            elif hasattr(y_train, 'numpy'):
                y_train = y_train.numpy()
            else:
                y_train = np.array(y_train)
                
            if hasattr(y_val, 'values'):
                y_val = y_val.values
            elif hasattr(y_val, 'numpy'):
                y_val = y_val.numpy()
            else:
                y_val = np.array(y_val)
            
            # Проверяем, что данные не пустые
            if len(X_train) == 0 or len(X_val) == 0:
                print("⚠️ Ошибка: пустые данные для DeepNeuralNetwork")
                return 0.0
            
            # Преобразуем данные в тензоры
            X_train_tensor = torch.FloatTensor(X_train)
            y_train_tensor = torch.LongTensor(y_train)
            X_val_tensor = torch.FloatTensor(X_val)
            y_val_tensor = torch.LongTensor(y_val)
            
            # GPU если доступен
            if torch.cuda.is_available():
                X_train_tensor = X_train_tensor.cuda()
                y_train_tensor = y_train_tensor.cuda()
                X_val_tensor = X_val_tensor.cuda()
                y_val_tensor = y_val_tensor.cuda()
            
            # DataLoader
            train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
            train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True)
            
            # Оптимизатор и функция потерь
            optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'])
            criterion = nn.CrossEntropyLoss()
            
            # Обучение
            model.train()
            best_score = 0.0
            patience = 10
            no_improve = 0
            
            for epoch in range(min(50, params.get('epochs', 50))):  # Максимум 50 эпох или из params
                epoch_loss = 0
                for batch_X, batch_y in train_loader:
                    optimizer.zero_grad()
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()
                
                # Early stopping на валидации
                if epoch % 5 == 0:
                    model.eval()
                    with torch.no_grad():
                        val_outputs = model(X_val_tensor)
                        val_pred = torch.argmax(val_outputs, dim=1)
                        score = f1_score(y_val_tensor.cpu().numpy(), val_pred.cpu().numpy(), average='weighted', zero_division=0)
                        
                        if score > best_score:
                            best_score = score
                            no_improve = 0
                        else:
                            no_improve += 1
                            
                        if no_improve >= patience:
                            break
                    model.train()
            
            return best_score
            
        except Exception as e:
            print(f"⚠️ Ошибка в train_deep_learning_model: {e}")
            import traceback
            traceback.print_exc()
            return 0.0
    
    def predict_deep_learning(self, model, X):
        """
        Предсказания для модели глубокого обучения
        """
        if not PYTORCH_AVAILABLE:
            return np.zeros(len(X))
        
        model.eval()
        X_tensor = torch.FloatTensor(X)
        
        if torch.cuda.is_available():
            X_tensor = X_tensor.cuda()
        
        with torch.no_grad():
            outputs = model(X_tensor)
            predictions = torch.argmax(outputs, dim=1)
        
        return predictions.cpu().numpy()
    
    def evaluate_model(self, model, X_train, X_val, y_train, y_val, model_name):
        """
        Оценивает модель с различными метриками (поддержка 3 классов)
        """
        try:
            # Убираем sample_weight из признаков если есть
            X_train_clean = X_train.drop('sample_weight', axis=1, errors='ignore')
            X_val_clean = X_val.drop('sample_weight', axis=1, errors='ignore')
            
            # Используем естественную балансировку (без SMOTE)
            X_train_balanced, y_train_balanced = X_train_clean, y_train
            
            # Обучаем модель (без весов)
            model.fit(X_train_balanced, y_train_balanced)
            
            # Предсказания
            y_pred = model.predict(X_val_clean)
            y_pred_proba = model.predict_proba(X_val_clean)
            
            # Проверяем, что предсказания не пустые
            if len(y_pred) == 0:
                print(f"⚠️ Ошибка: пустые предсказания для {model_name}")
                return {
                    'accuracy': 0.0,
                    'precision': 0.0,
                    'recall': 0.0,
                    'f1': 0.0,
                    'roc_auc': 0.0
                }, None
            
            # Метрики для многоклассовой классификации
            try:
                roc_auc = roc_auc_score(y_val, y_pred_proba, multi_class='ovr', average='weighted')
            except Exception as e:
                print(f"⚠️ Ошибка ROC AUC для {model_name}: {e}")
                roc_auc = 0.0  # Fallback если roc_auc не работает
            
            metrics = {
                'accuracy': accuracy_score(y_val, y_pred),
                'precision': precision_score(y_val, y_pred, average='weighted', zero_division=0),
                'recall': recall_score(y_val, y_pred, average='weighted', zero_division=0),
                'f1': f1_score(y_val, y_pred, average='weighted', zero_division=0),
                'roc_auc': roc_auc
            }
            
            # Проверяем, что все метрики не nan
            for key, value in metrics.items():
                if np.isnan(value):
                    print(f"⚠️ Ошибка: nan значение для {key} в {model_name}")
                    metrics[key] = 0.0
            
            # Важность признаков
            feature_importance = None
            if hasattr(model, 'feature_importances_'):
                feature_importance = model.feature_importances_
            elif hasattr(model, 'coef_'):
                feature_importance = np.abs(model.coef_[0])
            
            return metrics, feature_importance
            
        except Exception as e:
            print(f"⚠️ Ошибка в evaluate_model для {model_name}: {e}")
            import traceback
            traceback.print_exc()
            return {
                'accuracy': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'f1': 0.0,
                'roc_auc': 0.0
            }, None
    
    def time_series_cross_validation(self, X, y, n_splits=8):  # Увеличено с 5 до 8
        """
        Продвинутая кросс-валидация с временными окнами
        Улучшено для более тщательного обучения с увеличенной целевой переменной
        """
        print(f"🔄 Запуск кросс-валидации с {n_splits} фолдами...")
        
        tscv = TimeSeriesSplit(n_splits=n_splits)
        models_config = self.get_models_for_optuna()
        
        results = {}
        feature_importance = {}
        
        for model_name, config in models_config.items():
            print(f"   Тестирование {model_name}...")
            
            cv_scores = {
                'accuracy': [], 'precision': [], 'recall': [], 
                'f1': [], 'roc_auc': []
            }
            
            all_feature_importance = []
            best_model = None
            best_score = 0
            
            for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                
                # Отбор признаков для каждого фолда
                X_train_selected, selected_features = self.select_features(
                    X_train, y_train, method='kbest', n_features=50
                )
                X_val_selected = X_val[selected_features]
                
                # Нормализация
                X_train_scaled = self.scaler.fit_transform(X_train_selected)
                X_val_scaled = self.scaler.transform(X_val_selected)
                
                # Выбираем продвинутый метод оптимизации
                if config['params']:
                    print(f"      Фолд {fold+1}: продвинутая оптимизация ({model_name})...")
                    
                    # Создаем уникальное имя для исследования
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # Микросекунды для уникальности
                    study_name = f"{model_name}_{self.timeframe}_fold{fold+1}_{timestamp}"
                    
                    # Выбираем лучший алгоритм в зависимости от модели
                    if model_name in ['XGBoost', 'LightGBM', 'CatBoost']:
                        # Hyperband для градиентного бустинга (ASHA не поддерживает max_resource)
                        study = optuna.create_study(
                            study_name=study_name,
                            direction='maximize',
                            sampler=optuna.samplers.TPESampler(n_startup_trials=20),
                            pruner=optuna.pruners.HyperbandPruner(
                                min_resource=20,
                                max_resource=200,
                                reduction_factor=3
                            )
                        )
                        n_trials = 50  # Увеличено с 10 до 50
                        
                    elif model_name in ['RandomForest', 'ExtraTrees']:
                        # BOHB (Bayesian Optimization and HyperBand) для ансамблевых методов
                        study = optuna.create_study(
                            study_name=study_name,
                            direction='maximize',
                            sampler=optuna.samplers.TPESampler(n_startup_trials=25),
                            pruner=optuna.pruners.HyperbandPruner(
                                min_resource=10,
                                max_resource=100,
                                reduction_factor=2
                            )
                        )
                        n_trials = 80  # Увеличено с 40 до 80
                        
                    elif model_name == 'DeepNeuralNetwork':
                        # Hyperband для глубокого обучения
                        study = optuna.create_study(
                            study_name=study_name,
                            direction='maximize',
                            sampler=optuna.samplers.TPESampler(n_startup_trials=10),
                            pruner=optuna.pruners.HyperbandPruner(
                                min_resource=5,
                                max_resource=50,
                                reduction_factor=2
                            )
                        )
                        n_trials = 60  # Увеличено с 30 до 60
                        
                    elif model_name == 'SVM':
                        # CMA-ES для SVM
                        study = optuna.create_study(
                            study_name=study_name,
                            direction='maximize',
                            sampler=optuna.samplers.CmaEsSampler(),
                            pruner=optuna.pruners.MedianPruner(
                                n_startup_trials=10,
                                n_warmup_steps=20
                            )
                        )
                        n_trials = 50  # Увеличено с 25 до 50
                        
                    else:
                        # TPE с MedianPruner для остальных моделей
                        study = optuna.create_study(
                            study_name=study_name,
                            direction='maximize',
                            sampler=optuna.samplers.TPESampler(n_startup_trials=20),
                            pruner=optuna.pruners.MedianPruner(
                                n_startup_trials=10,
                                n_warmup_steps=20
                            )
                        )
                        n_trials = 60  # Увеличено с 30 до 60
                    
                    # Оптимизация
                    try:
                        print(f"        Запуск оптимизации без таймаута...")
                        
                        study.optimize(
                            lambda trial: self.optuna_objective(
                                trial, X_train_scaled, X_val_scaled, y_train, y_val, model_name
                            ),
                            n_trials=n_trials,
                            show_progress_bar=False
                        )
                    except Exception as e:
                        print(f"      ⚠️ Ошибка оптимизации: {e}")
                        print(f"      Пропускаем оптимизацию для этого фолда...")
                        continue
                    
                    # Получаем лучшие параметры и создаем модель
                    best_params = study.best_params
                    if model_name == 'DeepNeuralNetwork':
                        model = self.create_deep_learning_model(X_train_scaled.shape[1], best_params)
                    else:
                        model = self.create_model_with_params(model_name, best_params)
                    
                    # Сохраняем результаты Optuna
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    study_file = f'model_results/optuna/{model_name}_{self.timeframe}_fold{fold+1}_{timestamp}.pkl'
                    with open(study_file, 'wb') as f:
                        pickle.dump(study, f)
                    
                else:
                    # Простое обучение без оптимизации
                    if model_name == 'DeepNeuralNetwork':
                        model = self.create_deep_learning_model(X_train_scaled.shape[1], {})
                    else:
                        model = config['model']
                
                # Оценка модели с продвинутой балансировкой
                try:
                    metrics, importance = self.evaluate_model_advanced(
                        model, X_train_scaled, X_val_scaled, y_train, y_val, model_name
                    )
                    
                    # Сохраняем результаты
                    for metric, value in metrics.items():
                        cv_scores[metric].append(value)
                    
                    if importance is not None:
                        all_feature_importance.append(importance)
                    
                    # Сохраняем лучшую модель (используем F1-score)
                    if metrics['f1'] > best_score:
                        best_score = metrics['f1']
                        best_model = model
                        
                except Exception as e:
                    print(f"      ❌ Ошибка оценки модели в фолде {fold+1}: {e}")
                    print(f"      Пропускаем фолд {fold+1} и продолжаем...")
                    continue
            
            # Средние результаты
            avg_results = {}
            for metric, values in cv_scores.items():
                avg_results[metric] = np.mean(values)
                avg_results[f'{metric}_std'] = np.std(values)
            
            results[model_name] = avg_results
            
            # Сохраняем лучшую модель
            self.models[model_name] = best_model
            
            # Сохраняем важность признаков
            if all_feature_importance:
                feature_importance[model_name] = np.mean(all_feature_importance, axis=0)
            
            print(f"   ✅ {model_name} завершен")
            print(f"      F1: {avg_results['f1']:.4f} ± {avg_results['f1_std']:.4f}")
            print(f"      ROC-AUC: {avg_results['roc_auc']:.4f} ± {avg_results['roc_auc_std']:.4f}")
        
        return results, feature_importance
    
    def evaluate_model_advanced(self, model, X_train, X_val, y_train, y_val, model_name):
        """
        Оценивает модель без балансировки классов (естественное распределение)
        """
        try:
            print(f"      Обучение {model_name} без балансировки (естественное распределение)...")
            print(f"        Размер обучающих данных: {X_train.shape}")
            print(f"        Распределение классов: {np.bincount(y_train)}")
            
            # Обучение без балансировки
            if model_name == 'DeepNeuralNetwork':
                self.train_deep_learning_model(model, X_train, X_val, y_train, y_val, {})
                y_pred = self.predict_deep_learning(model, X_val)
                y_pred_proba = None  # PyTorch не предоставляет вероятности по умолчанию
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_val)
                y_pred_proba = model.predict_proba(X_val)
            
            # Проверяем, что предсказания не пустые
            if len(y_pred) == 0:
                print(f"⚠️ Ошибка: пустые предсказания для {model_name}")
                return {
                    'accuracy': 0.0,
                    'precision': 0.0,
                    'recall': 0.0,
                    'f1': 0.0,
                    'roc_auc': 0.0
                }, None
            
            # Метрики для многоклассовой классификации
            try:
                roc_auc = roc_auc_score(y_val, y_pred_proba, multi_class='ovr', average='weighted') if y_pred_proba is not None else 0.0
            except Exception as e:
                print(f"⚠️ Ошибка ROC AUC для {model_name}: {e}")
                roc_auc = 0.0
                
            metrics = {
                'accuracy': accuracy_score(y_val, y_pred),
                'precision': precision_score(y_val, y_pred, average='weighted', zero_division=0),
                'recall': recall_score(y_val, y_pred, average='weighted', zero_division=0),
                'f1': f1_score(y_val, y_pred, average='weighted', zero_division=0),
                'roc_auc': roc_auc
            }
            
            # Проверяем, что все метрики не nan
            for key, value in metrics.items():
                if np.isnan(value):
                    print(f"⚠️ Ошибка: nan значение для {key} в {model_name}")
                    metrics[key] = 0.0
            
            # Важность признаков
            feature_importance = None
            if hasattr(model, 'feature_importances_'):
                feature_importance = model.feature_importances_
            elif hasattr(model, 'coef_'):
                feature_importance = np.abs(model.coef_[0])
            
            return metrics, feature_importance
            
        except Exception as e:
            print(f"⚠️ Ошибка в evaluate_model_advanced для {model_name}: {e}")
            import traceback
            traceback.print_exc()
            return {
                'accuracy': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'f1': 0.0,
                'roc_auc': 0.0
            }, None
    
    def train_final_models(self, X, y):
        """
        Обучает финальные модели на всех данных с лучшими параметрами (без балансировки)
        """
        print("🎯 Обучение финальных моделей...")
        
        # Отбор признаков
        X_selected, selected_features = self.select_features(X, y, method='kbest', n_features=50)
        
        # Нормализация
        X_scaled = self.scaler.fit_transform(X_selected)
        
        # Обучаем модели с лучшими параметрами (без балансировки)
        for model_name, model in self.models.items():
            print(f"   Финальное обучение {model_name}...")
            model.fit(X_scaled, y)
        
        # Сохраняем информацию о признаках
        self.selected_features = selected_features
        
        print(f"✅ Обучено {len(self.models)} финальных моделей")
    
    def save_results(self, results, feature_importance):
        """
        Сохраняет результаты обучения
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Сохраняем результаты
        results_file = f'model_results/validation/results_{self.timeframe}_{timestamp}.json'
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Сохраняем важность признаков
        if feature_importance:
            importance_file = f'model_results/features/importance_{self.timeframe}_{timestamp}.json'
            importance_data = {}
            for model_name, importance in feature_importance.items():
                importance_data[model_name] = importance.tolist()
            
            with open(importance_file, 'w') as f:
                json.dump(importance_data, f, indent=2)
        
        # Сохраняем модели
        for model_name, model in self.models.items():
            model_file = f'model_results/models/{model_name}_{self.timeframe}_{timestamp}.pkl'
            with open(model_file, 'wb') as f:
                pickle.dump(model, f)
        
        # Сохраняем скалер и признаки
        scaler_file = f'model_results/models/scaler_{self.timeframe}_{timestamp}.pkl'
        with open(scaler_file, 'wb') as f:
            pickle.dump(self.scaler, f)
        
        features_file = f'model_results/features/selected_features_{self.timeframe}_{timestamp}.json'
        with open(features_file, 'w') as f:
            json.dump(self.selected_features, f, indent=2)
        
        print(f"💾 Результаты сохранены:")
        print(f"   Результаты: {results_file}")
        print(f"   Модели: model_results/models/")
        print(f"   Признаки: model_results/features/")
    
    def create_comparison_plot(self, results):
        """
        Создает график сравнения моделей
        """
        print("📊 Создание графика сравнения...")
        
        # Подготавливаем данные
        metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
        model_names = list(results.keys())
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for i, metric in enumerate(metrics):
            values = [results[model][metric] for model in model_names]
            errors = [results[model][f'{metric}_std'] for model in model_names]
            
            bars = axes[i].bar(model_names, values, yerr=errors, capsize=5)
            axes[i].set_title(f'{metric.upper()}')
            axes[i].set_ylabel('Score')
            axes[i].tick_params(axis='x', rotation=45)
            
            # Добавляем значения на столбцы
            for bar, value in zip(bars, values):
                height = bar.get_height()
                axes[i].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{value:.3f}', ha='center', va='bottom')
        
        # Убираем лишний subplot
        axes[-1].remove()
        
        plt.tight_layout()
        
        # Сохраняем график
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_file = f'model_results/validation/comparison_{self.timeframe}_{timestamp}.png'
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"�� График сохранен: {plot_file}")
    
    def print_results(self, results):
        """
        Выводит результаты в красивом формате (поддержка 3 классов)
        """
        print(f"\n🏆 РЕЗУЛЬТАТЫ СРАВНЕНИЯ МОДЕЛЕЙ ({self.timeframe.upper()})")
        print("=" * 80)
        print("Включает: Optuna с BOHB/Hyperband/MedianPruner (без балансировки)")
        print("Целевая переменная: 0 (шорт), 1 (нейтрально), 2 (лонг)")
        print("=" * 80)
        
        # Сортируем по ROC AUC
        sorted_results = sorted(results.items(), key=lambda x: x[1]['roc_auc'], reverse=True)
        
        print(f"{'Модель':<20} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1':<10} {'ROC AUC':<10}")
        print("-" * 80)
        
        for model_name, metrics in sorted_results:
            print(f"{model_name:<20} "
                  f"{metrics['accuracy']:<10.3f} "
                  f"{metrics['precision']:<10.3f} "
                  f"{metrics['recall']:<10.3f} "
                  f"{metrics['f1']:<10.3f} "
                  f"{metrics['roc_auc']:<10.3f}")
        
        print("-" * 80)
        
        # Лучшая модель
        best_model = sorted_results[0]
        print(f"🥇 Лучшая модель: {best_model[0]} (ROC AUC: {best_model[1]['roc_auc']:.3f})")
        print(f"🥈 Вторая модель: {sorted_results[1][0]} (ROC AUC: {sorted_results[1][1]['roc_auc']:.3f})")
        print(f"🥉 Третья модель: {sorted_results[2][0]} (ROC AUC: {sorted_results[2][1]['roc_auc']:.3f})")
        
        print(f"\n📊 Статистика классов:")
        print(f"   -1 (шорт): {self.short_samples} ({self.short_samples/self.total_samples*100:.1f}%)")
        print(f"    0 (нейтрально): {self.neutral_samples} ({self.neutral_samples/self.total_samples*100:.1f}%)")
        print(f"   +1 (лонг): {self.long_samples} ({self.long_samples/self.total_samples*100:.1f}%)")

def main():
    """
    Основная функция
    """
    print("🚀 Model Trainer - Система обучения моделей")
    print("=" * 60)
    print("✨ Включает: Optuna с продвинутыми алгоритмами (без балансировки)")
    print("🎯 Целевые переменные и горизонты НЕ ИЗМЕНЯЮТСЯ!")
    print("=" * 60)
    
    # Создаем тренер для 5M
    trainer_5m = ModelTrainer(timeframe='5m')
    
    # Загружаем данные
    df_5m = trainer_5m.load_data()
    
    # Создаем признаки
    df_5m = trainer_5m.create_features(df_5m)
    
    # Создаем целевую переменную (НЕ ИЗМЕНЯЕМ!)
    df_5m = trainer_5m.create_target(df_5m)
    
    # Подготавливаем данные
    feature_columns = [col for col in df_5m.columns if col not in ['timestamp', 'datetime', 'target']]
    X_5m = df_5m[feature_columns].fillna(0)
    y_5m = df_5m['target']
    
    print(f"📊 Данные подготовлены:")
    print(f"   Признаков: {len(feature_columns)}")
    print(f"   Примеров: {len(X_5m)}")
    print(f"   Положительных: {y_5m.sum()} ({y_5m.mean()*100:.1f}%)")
    
    # Временная кросс-валидация с оптимизацией
    results_5m, importance_5m = trainer_5m.time_series_cross_validation(X_5m, y_5m)
    
    # Выводим результаты
    trainer_5m.print_results(results_5m)
    
    # Создаем график
    trainer_5m.create_comparison_plot(results_5m)
    
    # Обучаем финальные модели
    trainer_5m.train_final_models(X_5m, y_5m)
    
    # Сохраняем результаты
    trainer_5m.save_results(results_5m, importance_5m)
    
    print(f"\n🎉 Обучение завершено для 5M таймфрейма!")
    
    # Аналогично для 15M
    print(f"\n" + "="*60)
    print("🔄 Обучение для 15M таймфрейма...")
    
    trainer_15m = ModelTrainer(timeframe='15m')
    df_15m = trainer_15m.load_data()
    df_15m = trainer_15m.create_features(df_15m)
    df_15m = trainer_15m.create_target(df_15m)
    
    X_15m = df_15m[feature_columns].fillna(0)
    y_15m = df_15m['target']
    
    results_15m, importance_15m = trainer_15m.time_series_cross_validation(X_15m, y_15m)
    trainer_15m.print_results(results_15m)
    trainer_15m.create_comparison_plot(results_15m)
    trainer_15m.train_final_models(X_15m, y_15m)
    trainer_15m.save_results(results_15m, importance_15m)
    
    print(f"\n🎉 Обучение завершено для 15M таймфрейма!")
    print(f"\n📁 Все результаты сохранены в папке model_results/")

if __name__ == "__main__":
    main()