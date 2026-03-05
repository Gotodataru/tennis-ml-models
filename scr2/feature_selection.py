import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier, CatBoostRegressor, Pool
import warnings
warnings.filterwarnings('ignore')

# ========== Конфигурация ==========
DATA_PATH = "data/final_multimarket_features.csv"
OUTPUT_DIR = "data/selected_features"
os.makedirs(OUTPUT_DIR, exist_ok=True)

TARGET_MAP = {
    'winner': 'winner_1',
    'total_games': 'total_games',
    'games_diff': 'games_diff',
    'first_set_winner': 'first_set_bin'
}

TARGET_TYPES = {
    'winner': 'classification',
    'total_games': 'regression',
    'games_diff': 'regression',
    'first_set_winner': 'classification'
}

TOP_K = 30
MISSING_THRESHOLD = 0.10
CORR_THRESHOLD = 0.95
TEST_SIZE = 0.2
RANDOM_STATE = 42

EXCLUDE_COLUMNS = [
    'match_id', 'date', 'player1_id', 'player2_id', 'player1_name', 'player2_name',
    'tourney_id', 'tourney_name', 'winner_id', 'first_set_winner',
    'first_set_bin', 'winner_1', 'total_games', 'games_diff', 'gender'
]

# ========== Загрузка данных ==========
print("Загрузка данных...")
df = pd.read_csv(DATA_PATH, low_memory=False)
print(f"Размер датасета: {df.shape}")
print(f"Первые 5 колонок: {df.columns[:5].tolist()} ...")

available_targets = {key: col for key, col in TARGET_MAP.items() if col in df.columns}
print(f"Найденные целевые колонки: {available_targets}")

if not available_targets:
    raise ValueError("В датасете нет ни одной из указанных целевых колонок.")

# ========== Функции отбора ==========
def filter_by_missing(X, threshold=MISSING_THRESHOLD):
    missing_ratio = X.isnull().mean()
    keep_cols = missing_ratio[missing_ratio <= threshold].index.tolist()
    dropped = set(X.columns) - set(keep_cols)
    if dropped:
        print(f"  Удалено признаков из-за пропусков >{threshold:.0%}: {len(dropped)}")
    return X[keep_cols], dropped

def remove_constant_features(X):
    constant_cols = [col for col in X.columns if X[col].nunique(dropna=False) <= 1]
    if constant_cols:
        print(f"  Удалено константных признаков: {len(constant_cols)}")
    return X.drop(columns=constant_cols), set(constant_cols)

def get_feature_importance(X_train, y_train, target_type, cat_features=None):
    if target_type == 'classification':
        model = CatBoostClassifier(
            iterations=1000, learning_rate=0.05, depth=6,
            loss_function='Logloss', eval_metric='Logloss',
            early_stopping_rounds=50, verbose=False,
            random_seed=RANDOM_STATE, cat_features=cat_features if cat_features else []
        )
    else:
        model = CatBoostRegressor(
            iterations=1000, learning_rate=0.05, depth=6,
            loss_function='RMSE', eval_metric='RMSE',
            early_stopping_rounds=50, verbose=False,
            random_seed=RANDOM_STATE, cat_features=cat_features if cat_features else []
        )
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=RANDOM_STATE, shuffle=False
    )
    train_pool = Pool(X_tr, y_tr, cat_features=cat_features)
    val_pool = Pool(X_val, y_val, cat_features=cat_features)
    model.fit(train_pool, eval_set=val_pool, verbose=False)
    importance = model.feature_importances_
    importance = importance / importance.sum() * 100
    return pd.Series(importance, index=X_train.columns)

def remove_highly_correlated(features, importance, threshold=CORR_THRESHOLD):
    # Берём только числовые колонки для корреляции
    numeric_cols = features.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) < 2:
        return features.columns.tolist()
    
    corr_matrix = features[numeric_cols].fillna(features[numeric_cols].median()).corr().abs()
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = set()
    for col in upper_tri.columns:
        if col in to_drop:
            continue
        high_corr = upper_tri.index[upper_tri[col] > threshold].tolist()
        if high_corr:
            group = [col] + high_corr
            best = max(group, key=lambda x: importance.get(x, 0))
            to_drop.update(set(group) - {best})
    print(f"  Удалено из-за высокой корреляции: {len(to_drop)}")
    return [f for f in features.columns if f not in to_drop]

def select_features_for_target_gender(df, target_key, gender, gender_col='gender', gender_value=None):
    target_col = TARGET_MAP.get(target_key)
    if target_col not in df.columns:
        print(f"  ⚠️ Целевая колонка '{target_col}' для {target_key} не найдена. Пропускаем.")
        return None

    print(f"\n========== {target_key.upper()} | {gender} ==========")

    if gender_value is not None and gender_col in df.columns:
        df_gender = df[df[gender_col] == gender_value].copy()
        print(f"Размер выборки после фильтра по полу '{gender_value}': {df_gender.shape}")
    else:
        df_gender = df.copy()
        print("Фильтр по полу не применяется.")

    X = df_gender.drop(columns=[col for col in EXCLUDE_COLUMNS if col in df_gender.columns] + [target_col])
    y = df_gender[target_col]

    X, _ = filter_by_missing(X, MISSING_THRESHOLD)
    X, _ = remove_constant_features(X)

    cat_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
    if cat_features:
        print(f"  Категориальные признаки: {cat_features}")

    if 'date' in df_gender.columns:
        df_gender_sorted = df_gender.sort_values('date')
        X_sorted = X.loc[df_gender_sorted.index]
        y_sorted = y.loc[df_gender_sorted.index]
        split_idx = int(len(X_sorted) * (1 - TEST_SIZE))
        X_train, X_test = X_sorted.iloc[:split_idx], X_sorted.iloc[split_idx:]
        y_train, y_test = y_sorted.iloc[:split_idx], y_sorted.iloc[split_idx:]
        print(f"Разделение по времени: train {len(X_train)} (первые {1-TEST_SIZE:.0%}), test {len(X_test)} (последние {TEST_SIZE:.0%})")
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, shuffle=True
        )
        print("⚠️ Нет колонки 'date', используем случайное разделение (возможны утечки).")

    print("Оценка важности признаков...")
    importance = get_feature_importance(X_train, y_train, TARGET_TYPES[target_key], cat_features)

    print("Анализ мультиколлинеарности...")
    selected = remove_highly_correlated(X_train, importance, CORR_THRESHOLD)

    # Финальный отбор топ-K и сортировка
    importance_filtered = importance[selected].sort_values(ascending=False)
    final_features = importance_filtered.head(TOP_K).index.tolist()
    final_importance = importance_filtered.head(TOP_K).values

    print(f"Итоговое количество признаков: {len(final_features)}")
    print("Топ-10 признаков с важностью:")
    for i in range(min(10, len(final_features))):
        print(f"  {final_features[i]}: {final_importance[i]:.2f}%")

    # Сохраняем только имена
    names_file = os.path.join(OUTPUT_DIR, f"selected_features_{target_key}_{gender}.txt")
    with open(names_file, 'w') as f:
        f.write("\n".join(final_features))
    print(f"Сохранено (имена): {names_file}")

    # Сохраняем с важностью
    importance_df = pd.DataFrame({
        'feature': final_features,
        'importance': final_importance
    })
    importance_file = os.path.join(OUTPUT_DIR, f"feature_importance_{target_key}_{gender}.csv")
    importance_df.to_csv(importance_file, index=False)
    print(f"Сохранено (с важностью): {importance_file}")

    return final_features

# ========== Основной цикл ==========
if __name__ == "__main__":
    gender_map = {'ATP': 'ATP', 'WTA': 'WTA'}
    for gender, gender_val in gender_map.items():
        for target_key in TARGET_MAP.keys():
            select_features_for_target_gender(df, target_key, gender,
                                              gender_col='gender', gender_value=gender_val)
    print("\nОтбор признаков завершён. Проверьте папку", OUTPUT_DIR)