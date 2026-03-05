import pandas as pd
import numpy as np
import os
import json
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import (log_loss, roc_auc_score, roc_curve,
                             mean_absolute_error, mean_squared_error,
                             brier_score_loss)
from sklearn.isotonic import IsotonicRegression
from catboost import CatBoostClassifier, CatBoostRegressor
import warnings
warnings.filterwarnings('ignore')

# ========== Конфигурация ==========
DATA_PATH = "data/final_multimarket_features.csv"
FEATURE_DIR = "data/selected_features"
OUTPUT_BASE = "."  # корневая папка проекта

TARGET_MAP = {
    'winner': 'winner_1',
    'total_games': 'total_games',
    'games_diff': 'games_diff',
    'first_set_winner': 'first_set_bin'
}

TARGET_TYPE = {
    'winner': 'classification',
    'total_games': 'regression',
    'games_diff': 'regression',
    'first_set_winner': 'classification'
}

GENDERS = ['ATP', 'WTA']

EXCLUDE = ['match_id', 'date', 'player1_id', 'player2_id', 'player1_name', 'player2_name',
           'tourney_id', 'tourney_name', 'winner_id', 'first_set_winner',
           'first_set_bin', 'winner_1', 'total_games', 'games_diff', 'gender']

# Параметры обучения
ITERATIONS = 2000
LEARNING_RATE = 0.05
DEPTH = 6
RANDOM_STATE = 42
VAL_SIZE = 0.2  # доля валидационной выборки (для early stopping и калибровки)

# ========== Загрузка данных ==========
print("Загрузка данных...")
df = pd.read_csv(DATA_PATH, low_memory=False)
print(f"Размер датасета: {df.shape}")

# ========== Функции для визуализации ==========
def plot_feature_importance(model, feature_names, output_path):
    """Сохраняет график важности признаков (топ-20)"""
    importance = model.feature_importances_
    indices = np.argsort(importance)[::-1][:20]
    plt.figure(figsize=(10, 6))
    plt.title('Feature Importance (top 20)')
    plt.barh(range(len(indices)), importance[indices], align='center')
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.xlabel('Importance')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(output_path, dpi=100)
    plt.close()

def plot_learning_curve(evals_result, output_path):
    """Сохраняет график кривой обучения"""
    plt.figure(figsize=(10, 6))
    for dataset in evals_result:
        for metric in evals_result[dataset]:
            plt.plot(evals_result[dataset][metric], label=f'{dataset}_{metric}')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Learning Curve')
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=100)
    plt.close()

def plot_roc_curve(y_true, y_pred, output_path):
    """ROC-кривая для классификации"""
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    auc = roc_auc_score(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(output_path, dpi=100)
    plt.close()

def plot_residuals(y_true, y_pred, output_path):
    """График остатков для регрессии"""
    residuals = y_true - y_pred
    plt.figure(figsize=(10, 6))
    plt.scatter(y_pred, residuals, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted')
    plt.ylabel('Residuals')
    plt.title('Residuals Plot')
    plt.tight_layout()
    plt.savefig(output_path, dpi=100)
    plt.close()

# ========== Функция обучения одной модели ==========
def train_final_model(gender, target_key):
    target_col = TARGET_MAP[target_key]
    imp_file = os.path.join(FEATURE_DIR, f"feature_importance_{target_key}_{gender}.csv")
    if not os.path.exists(imp_file):
        print(f"Файл {imp_file} не найден, пропускаем.")
        return

    # Берём топ-11 признаков
    imp_df = pd.read_csv(imp_file)
    top_features = imp_df.head(11)['feature'].tolist()
    print(f"\n{'='*60}")
    print(f"Обучение модели: {target_key.upper()} | {gender}")
    print(f"Признаки (11): {top_features}")

    # Фильтр по полу
    df_gender = df[df['gender'] == gender].copy()
    if len(df_gender) == 0:
        print(f"Нет данных для {gender}")
        return

    # Сортируем по дате для временного разделения
    df_gender = df_gender.sort_values('date')
    X = df_gender[top_features].copy()
    y = df_gender[target_col].copy()

    # Категориальные признаки
    cat_features = [feat for feat in top_features if X[feat].dtype == 'object']
    
    # Разделение на train/val с сохранением временного порядка
    split_idx = int(len(X) * (1 - VAL_SIZE))
    X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]
    
    print(f"Train size: {len(X_train)}, Val size: {len(X_val)}")

    # Определяем путь для сохранения
    target_path_map = {
        'winner': 'WINNER',
        'total_games': 'TOTAL',
        'games_diff': 'HANDICAP',
        'first_set_winner': 'FIRSTSET'
    }
    folder = target_path_map[target_key]
    model_dir = os.path.join(OUTPUT_BASE, gender, folder, 'model', f'{target_key}_{gender.lower()}')
    os.makedirs(model_dir, exist_ok=True)
    print(f"Модель будет сохранена в: {model_dir}")

    # Создание модели
    if TARGET_TYPE[target_key] == 'classification':
        model = CatBoostClassifier(
            iterations=ITERATIONS,
            learning_rate=LEARNING_RATE,
            depth=DEPTH,
            loss_function='Logloss',
            eval_metric='Logloss',
            random_seed=RANDOM_STATE,
            cat_features=cat_features,
            early_stopping_rounds=100,
            verbose=100
        )
    else:
        model = CatBoostRegressor(
            iterations=ITERATIONS,
            learning_rate=LEARNING_RATE,
            depth=DEPTH,
            loss_function='RMSE',
            eval_metric='RMSE',
            random_seed=RANDOM_STATE,
            cat_features=cat_features,
            early_stopping_rounds=100,
            verbose=100
        )

    # Обучение
    model.fit(
        X_train, y_train,
        eval_set=(X_val, y_val),
        plot=False,
        verbose_eval=100
    )

    # История обучения
    evals_result = model.get_evals_result()

    # Предсказания на валидации
    if TARGET_TYPE[target_key] == 'classification':
        pred_proba = model.predict_proba(X_val)[:, 1]
        pred_class = (pred_proba > 0.5).astype(int)
        
        # Метрики
        logloss = log_loss(y_val, pred_proba)
        auc = roc_auc_score(y_val, pred_proba)
        brier = brier_score_loss(y_val, pred_proba)
        
        metrics = {
            'log_loss': logloss,
            'roc_auc': auc,
            'brier_score': brier,
            'accuracy': (pred_class == y_val).mean()
        }
        print(f"Метрики на валидации: LogLoss={logloss:.4f}, AUC={auc:.4f}, Brier={brier:.4f}")
        
        # Калибровка вероятностей
        calibrator = IsotonicRegression(out_of_bounds='clip')
        calibrator.fit(pred_proba, y_val)
        joblib.dump(calibrator, os.path.join(model_dir, 'isotonic_calibrator.pkl'))
        print("Калибратор сохранён.")
        
        # ROC-кривая
        plot_roc_curve(y_val, pred_proba, os.path.join(model_dir, 'roc_curve.png'))
    else:
        pred = model.predict(X_val)
        mae = mean_absolute_error(y_val, pred)
        mse = mean_squared_error(y_val, pred)
        rmse = np.sqrt(mse)
        metrics = {
            'mae': mae,
            'rmse': rmse,
            'mse': mse
        }
        print(f"Метрики на валидации: MAE={mae:.4f}, RMSE={rmse:.4f}")
        plot_residuals(y_val, pred, os.path.join(model_dir, 'residuals.png'))

    # Сохраняем метрики в JSON
    with open(os.path.join(model_dir, 'evaluation_report.json'), 'w') as f:
        json.dump(metrics, f, indent=4)

    # График важности признаков
    plot_feature_importance(model, top_features, os.path.join(model_dir, 'feature_importance.png'))
    
    # Кривая обучения
    plot_learning_curve(evals_result, os.path.join(model_dir, 'learning_curve.png'))

    # Сохраняем модель и список признаков
    model.save_model(os.path.join(model_dir, 'model.cbm'))
    with open(os.path.join(model_dir, 'feature_names.txt'), 'w') as f:
        f.write("\n".join(top_features))

    print(f"Модель и все артефакты сохранены в {model_dir}")
    print('='*60)

# ========== Запуск ==========
for gender in GENDERS:
    for target_key in TARGET_MAP.keys():
        train_final_model(gender, target_key)

print("\nВсе модели успешно переобучены!")