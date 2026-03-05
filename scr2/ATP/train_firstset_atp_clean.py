#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Обучение классификационной модели first_set_winner для ATP на чистом датасете.
Целевая переменная: 1 – победил первый игрок в первом сете, 0 – второй.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss, confusion_matrix
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
import json
import os
import joblib
import warnings
warnings.filterwarnings('ignore')

# ========== КОНФИГУРАЦИЯ ==========
DATA_PATH = 'data/clean_multimarket_features.csv'
MODEL_DIR = 'ATP/FIRSTSET/model/first_set_atp'
FEATURE_LIST_PATH = 'data/feature_analysis/selected_features.txt'
os.makedirs(MODEL_DIR, exist_ok=True)

RANDOM_SEED = 42

# ========== ЗАГРУЗКА ДАННЫХ ==========
print("Загрузка датасета...")
df = pd.read_csv(DATA_PATH, parse_dates=['date'])
print(f"Всего записей: {len(df)}")

# Оставляем только ATP
df = df[df['gender'] == 'ATP'].copy()
print(f"Записей ATP: {len(df)}")

if len(df) == 0:
    raise ValueError("Нет данных для ATP! Проверьте колонку gender.")

# Удаляем матчи с неизвестным победителем первого сета
df = df.dropna(subset=['first_set_winner']).copy()
print(f"После удаления пропусков в first_set_winner: {len(df)}")

# Создаём бинарную целевую переменную
df['first_set_bin'] = (df['first_set_winner'] == df['player1_id']).astype(int)
target = 'first_set_bin'

# Проверим распределение
print(f"Распределение целевой переменной:\n{df[target].value_counts()}")

# Загружаем список отобранных признаков
with open(FEATURE_LIST_PATH, 'r') as f:
    selected_features = [line.strip() for line in f if line.strip()]

feature_cols = [col for col in selected_features if col in df.columns]
print(f"Используется признаков: {len(feature_cols)}")

X = df[feature_cols].copy()
y = df[target].values

# Заполняем пропуски (если есть)
X = X.fillna(X.median(numeric_only=True)).fillna(0)

# ========== ВРЕМЕННОЕ РАЗДЕЛЕНИЕ ==========
split_date = df['date'].quantile(0.8)
train_mask = df['date'] <= split_date
test_mask = df['date'] > split_date

X_train, X_test = X[train_mask], X[test_mask]
y_train, y_test = y[train_mask], y[test_mask]
dates_test = df.loc[test_mask, 'date']

print(f"\nТренировочных матчей ATP: {len(X_train)}")
print(f"Тестовых матчей ATP: {len(X_test)}")
print(f"Дата разделения: {split_date.date()}")

# ========== ОБУЧЕНИЕ ==========
model = CatBoostClassifier(
    iterations=1000,
    learning_rate=0.05,
    depth=6,
    loss_function='Logloss',
    eval_metric='AUC',
    random_seed=RANDOM_SEED,
    verbose=200,
    early_stopping_rounds=50
)

model.fit(X_train, y_train, eval_set=(X_test, y_test))

y_pred_proba = model.predict_proba(X_test)[:, 1]
y_pred = model.predict(X_test)

# ========== МЕТРИКИ КЛАССИФИКАЦИИ ==========
acc = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred_proba)
logloss = log_loss(y_test, y_pred_proba)

print("\n=== ОСНОВНЫЕ МЕТРИКИ (ATP first_set_winner) ===")
print(f"Accuracy = {acc:.4f}")
print(f"AUC      = {auc:.4f}")
print(f"LogLoss  = {logloss:.4f}")

# Матрица ошибок
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)

# ========== КАЛИБРОВКА ВЕРОЯТНОСТЕЙ ==========
iso_reg = IsotonicRegression(out_of_bounds='clip')
iso_reg.fit(y_pred_proba, y_test)
y_pred_calib = iso_reg.predict(y_pred_proba)

# Калибровочные кривые
prob_true_raw, prob_pred_raw_binned = calibration_curve(y_test, y_pred_proba, n_bins=10)
prob_true_calib, prob_pred_calib_binned = calibration_curve(y_test, y_pred_calib, n_bins=10)

# ========== СОХРАНЕНИЕ МОДЕЛИ ==========
model.save_model(os.path.join(MODEL_DIR, 'model.cbm'))

# Сохраняем калибратор
joblib.dump(iso_reg, os.path.join(MODEL_DIR, 'isotonic_calibrator.pkl'))

# Список признаков
with open(os.path.join(MODEL_DIR, 'feature_names.txt'), 'w') as f:
    for feat in X.columns:
        f.write(f"{feat}\n")

# Важность признаков
fi = pd.DataFrame({'feature': X.columns, 'importance': model.feature_importances_})
fi = fi.sort_values('importance', ascending=False)
fi.to_csv(os.path.join(MODEL_DIR, 'feature_importance.csv'), index=False)

# ========== ДИАГНОСТИЧЕСКИЕ ГРАФИКИ ==========
plt.figure(figsize=(18, 10))

plt.subplot(2, 3, 1)
plt.hist(y_pred_proba, bins=50, alpha=0.7, edgecolor='black')
plt.xlabel('Предсказанная вероятность P(победа 1-го игрока в 1-м сете)')
plt.ylabel('Частота')
plt.title('Распределение предсказанных вероятностей')

plt.subplot(2, 3, 2)
plt.plot(prob_pred_raw_binned, prob_true_raw, marker='o', label='Сырая', linestyle='--')
plt.plot(prob_pred_calib_binned, prob_true_calib, marker='s', label='После калибровки')
plt.plot([0, 1], [0, 1], 'k--', label='Идеальная')
plt.xlabel('Средняя предсказанная вероятность')
plt.ylabel('Доля положительных исходов')
plt.title('Калибровочная кривая')
plt.legend()

plt.subplot(2, 3, 3)
top10 = fi.head(10)
sns.barplot(data=top10, y='feature', x='importance', palette='viridis')
plt.xlabel('Важность')
plt.title('Топ-10 важных признаков')

plt.subplot(2, 3, 4)
from sklearn.metrics import roc_curve
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
plt.plot(fpr, tpr, label=f'AUC = {auc:.3f}')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC-кривая')
plt.legend()

plt.subplot(2, 3, 5)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Предсказанный класс')
plt.ylabel('Истинный класс')
plt.title('Матрица ошибок')

plt.subplot(2, 3, 6)
test_df = pd.DataFrame({'date': dates_test, 'y_true': y_test, 'y_pred_proba': y_pred_proba})
test_df.set_index('date', inplace=True)
monthly_true = test_df['y_true'].resample('ME').mean()
monthly_pred = test_df['y_pred_proba'].resample('ME').mean()
plt.plot(monthly_true.index, monthly_true, marker='o', label='Истинная доля')
plt.plot(monthly_pred.index, monthly_pred, marker='s', label='Средняя предсказанная вероятность')
plt.xlabel('Дата')
plt.ylabel('Доля побед 1-го игрока')
plt.title('Стабильность во времени')
plt.legend()
plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig(os.path.join(MODEL_DIR, 'diagnostics.png'), dpi=150)
plt.show()
print(f"✅ Диагностический график сохранён: {MODEL_DIR}/diagnostics.png")

# ========== СОХРАНЕНИЕ ОТЧЁТА ==========
report = {
    'model': 'first_set_atp_clean',
    'train_size': len(X_train),
    'test_size': len(X_test),
    'split_date': str(split_date.date()),
    'metrics': {
        'accuracy': float(acc),
        'auc': float(auc),
        'logloss': float(logloss)
    },
    'calibration': {
        'raw': {
            'prob_true': [float(x) for x in prob_true_raw],
            'prob_pred': [float(x) for x in prob_pred_raw_binned]
        },
        'calibrated': {
            'prob_true': [float(x) for x in prob_true_calib],
            'prob_pred': [float(x) for x in prob_pred_calib_binned]
        }
    }
}
with open(os.path.join(MODEL_DIR, 'evaluation_report.json'), 'w') as f:
    json.dump(report, f, indent=2)

print(f"\n✅ Модель, калибратор и отчёт сохранены в {MODEL_DIR}")
print("Готово!")