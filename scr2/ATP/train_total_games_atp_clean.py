#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Обучение регрессионной модели total_games для ATP на чистом датасете
с использованием отфильтрованных признаков (selected_features.txt).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from scipy.stats import pearsonr, probplot, skew, kurtosis, norm
import json
import os
import joblib
import warnings
warnings.filterwarnings('ignore')

# ========== КОНФИГУРАЦИЯ ==========
DATA_PATH = 'data/clean_multimarket_features.csv'
MODEL_DIR = 'ATP/TOTAL/model/total_games_atp'
FEATURE_LIST_PATH = 'data/feature_analysis/selected_features.txt'
os.makedirs(MODEL_DIR, exist_ok=True)

RANDOM_SEED = 42
TOTAL_THRESHOLD = 22.5          # порог для калибровки вероятности

# ========== ЗАГРУЗКА ДАННЫХ ==========
print("Загрузка датасета...")
df = pd.read_csv(DATA_PATH, parse_dates=['date'])
print(f"Всего записей: {len(df)}")

# Оставляем только ATP
df = df[df['gender'] == 'ATP'].copy()
print(f"Записей ATP: {len(df)}")

if len(df) == 0:
    raise ValueError("Нет данных для ATP! Проверьте колонку gender.")

target = 'total_games'

# Загружаем список отобранных признаков
with open(FEATURE_LIST_PATH, 'r') as f:
    selected_features = [line.strip() for line in f if line.strip()]

# Убедимся, что все признаки присутствуют в датасете
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
model = CatBoostRegressor(
    iterations=1000,
    learning_rate=0.05,
    depth=6,
    loss_function='MAE',
    eval_metric='MAE',
    random_seed=RANDOM_SEED,
    verbose=200,
    early_stopping_rounds=50
)

model.fit(X_train, y_train, eval_set=(X_test, y_test))

y_pred = model.predict(X_test)

# ========== МЕТРИКИ РЕГРЕССИИ ==========
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
mae_std = np.std(np.abs(y_test - y_pred))

print("\n=== ОСНОВНЫЕ МЕТРИКИ (ATP total_games) ===")
print(f"MAE  = {mae:.3f} ± {mae_std:.3f}")
print(f"RMSE = {rmse:.3f}")
print(f"R²   = {r2:.4f}")

# MAE по диапазонам
bins = [0, 15, 20, 25, 30, 35, 200]
labels = ['<15', '15-20', '20-25', '25-30', '30-35', '>35']
y_test_binned = pd.cut(y_test, bins=bins, labels=labels)
mae_by_bin = pd.Series(index=labels, dtype=float)
for label in labels:
    mask = y_test_binned == label
    if mask.sum() > 0:
        mae_by_bin[label] = mean_absolute_error(y_test[mask], y_pred[mask])
    else:
        mae_by_bin[label] = np.nan

print("\nMAE по диапазонам:")
print(mae_by_bin.round(2))

# ========== СОХРАНЕНИЕ МОДЕЛИ ==========
model.save_model(os.path.join(MODEL_DIR, 'model.cbm'))

# Сохраняем список признаков
with open(os.path.join(MODEL_DIR, 'feature_names.txt'), 'w') as f:
    for feat in X.columns:
        f.write(f"{feat}\n")

# Важность признаков
fi = pd.DataFrame({'feature': X.columns, 'importance': model.feature_importances_})
fi = fi.sort_values('importance', ascending=False)
fi.to_csv(os.path.join(MODEL_DIR, 'feature_importance.csv'), index=False)

# ========== ОСТАТКИ ==========
residuals = y_test - y_pred
sigma = np.std(residuals)

# ========== КАЛИБРОВКА ВЕРОЯТНОСТИ ТОТАЛА БОЛЬШЕ ПОРОГА ==========
prob_pred_raw = 1 - norm.cdf(TOTAL_THRESHOLD, loc=y_pred, scale=sigma)
y_binary = (y_test > TOTAL_THRESHOLD).astype(int)

iso_reg = IsotonicRegression(out_of_bounds='clip')
iso_reg.fit(prob_pred_raw, y_binary)
prob_pred_calib = iso_reg.predict(prob_pred_raw)

joblib.dump(iso_reg, os.path.join(MODEL_DIR, 'isotonic_calibrator.pkl'))

prob_true_raw, prob_pred_raw_binned = calibration_curve(y_binary, prob_pred_raw, n_bins=10)
prob_true_calib, prob_pred_calib_binned = calibration_curve(y_binary, prob_pred_calib, n_bins=10)

# ========== ДИАГНОСТИЧЕСКИЕ ГРАФИКИ ==========
plt.figure(figsize=(20, 12))

plt.subplot(2, 4, 1)
plt.scatter(y_test, y_pred, alpha=0.3, s=1)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Фактические')
plt.ylabel('Предсказанные')
plt.title(f'Фактические vs Предсказанные\nMAE={mae:.2f}, RMSE={rmse:.2f}')

plt.subplot(2, 4, 2)
plt.hist(residuals, bins=50, edgecolor='black', alpha=0.7)
plt.xlabel('Остатки')
plt.ylabel('Частота')
plt.title(f'Распределение остатков\nμ={np.mean(residuals):.2f}, σ={sigma:.2f}')

plt.subplot(2, 4, 3)
probplot(residuals, dist="norm", plot=plt)
plt.title('Q-Q plot остатков')

plt.subplot(2, 4, 4)
plt.scatter(y_pred, residuals, alpha=0.3, s=1)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Предсказанные')
plt.ylabel('Остатки')
plt.title('Остатки vs Предсказанные')

plt.subplot(2, 4, 5)
test_df = pd.DataFrame({'date': dates_test, 'residual_abs': np.abs(residuals)})
test_df.set_index('date', inplace=True)
monthly_mae = test_df['residual_abs'].resample('ME').mean()
monthly_mae.plot(marker='o', ax=plt.gca())
plt.xlabel('Дата')
plt.ylabel('MAE')
plt.title('Средняя абсолютная ошибка по месяцам')
plt.xticks(rotation=45)

plt.subplot(2, 4, 6)
top10 = fi.head(10)
sns.barplot(data=top10, y='feature', x='importance', palette='viridis')
plt.xlabel('Важность')
plt.title('Топ-10 важных признаков')

plt.subplot(2, 4, 7)
plt.plot(prob_pred_raw_binned, prob_true_raw, marker='o', label='Сырая', linestyle='--')
plt.plot(prob_pred_calib_binned, prob_true_calib, marker='s', label='После калибровки')
plt.plot([0, 1], [0, 1], 'k--', label='Идеальная')
plt.xlabel('Средняя предсказанная вероятность')
plt.ylabel('Доля положительных исходов')
plt.title(f'Калибровка (тотал > {TOTAL_THRESHOLD})')
plt.legend()

plt.subplot(2, 4, 8)
plt.hist(prob_pred_raw, bins=30, alpha=0.5, label='Сырая', edgecolor='black')
plt.hist(prob_pred_calib, bins=30, alpha=0.5, label='Откалиброванная', edgecolor='black')
plt.xlabel('Предсказанная вероятность P(тотал > 22.5)')
plt.ylabel('Частота')
plt.title('Распределение вероятностей')
plt.legend()

plt.tight_layout()
plt.savefig(os.path.join(MODEL_DIR, 'diagnostics.png'), dpi=150)
plt.show()
print(f"✅ Диагностический график сохранён: {MODEL_DIR}/diagnostics.png")

# ========== ПРОВЕРКА УТЕЧЕК ==========
corr, p_value = pearsonr(dates_test.values.astype(int), np.abs(residuals))
print(f"\n=== ПРОВЕРКА УТЕЧЕК ===")
print(f"Корреляция абсолютной ошибки с датой: r={corr:.3f}, p={p_value:.3e}")
if abs(corr) < 0.05:
    print("➡️ Ошибка не зависит от времени, утечек скорее всего нет.")
else:
    print("⚠️ Ошибка изменяется со временем — возможно, дрейф концепции или утечка.")

# ========== СОХРАНЕНИЕ ОТЧЁТА ==========
report = {
    'model': 'total_games_atp_clean',
    'train_size': len(X_train),
    'test_size': len(X_test),
    'split_date': str(split_date.date()),
    'metrics': {
        'mae': float(mae),
        'mae_std': float(mae_std),
        'rmse': float(rmse),
        'r2': float(r2),
        'mae_by_bin': mae_by_bin.dropna().to_dict()
    },
    'residual_stats': {
        'mean': float(np.mean(residuals)),
        'std': float(sigma),
        'skew': float(skew(residuals)),
        'kurtosis': float(kurtosis(residuals))
    },
    'calibration': {
        'threshold': TOTAL_THRESHOLD,
        'raw': {
            'prob_true': [float(x) for x in prob_true_raw],
            'prob_pred': [float(x) for x in prob_pred_raw_binned]
        },
        'calibrated': {
            'prob_true': [float(x) for x in prob_true_calib],
            'prob_pred': [float(x) for x in prob_pred_calib_binned]
        }
    },
    'leakage_check': {
        'corr_error_time': float(corr),
        'p_value': float(p_value)
    }
}
with open(os.path.join(MODEL_DIR, 'evaluation_report.json'), 'w') as f:
    json.dump(report, f, indent=2)

print(f"\n✅ Модель, калибратор и отчёт сохранены в {MODEL_DIR}")
print("Готово!")