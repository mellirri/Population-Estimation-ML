import geopandas as gpd
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import pickle
from pathlib import Path

# Пути
TRAIN_PATH = Path("data/processed/training_buildings.geojson")
MODEL_PATH = Path("models/population_model.pkl")

# 1. Загрузка
print("Загрузка датасета...")
gdf = gpd.read_file(TRAIN_PATH)

# Признаки и Цель
features = ["area", "levels", "total_floor_area"]
X = gdf[features].copy()
y = gdf["population"].copy()

print(f"Обучение на признаках: {features}")

# 2. Split (Разделение на обучение и тест)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"Размер выборки: Train={len(X_train)}, Test={len(X_test)}")

# 3. Обучение
print("Запуск RandomForest...")
model = RandomForestRegressor(
    n_estimators=300,
    max_depth=15,       # Ограничение глубины против переобучения
    min_samples_leaf=3, # Не учить уникальные выбросы
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)

# 4. Расчет метрик (И для Train, и для Test)
print("\nРасчет метрик...")

# Предсказание на обучающей выборке (как модель запомнила материал)
y_train_pred = model.predict(X_train)
mae_train = mean_absolute_error(y_train, y_train_pred)
r2_train = r2_score(y_train, y_train_pred)

# Предсказание на тестовой выборке (как модель работает в реальности)
y_test_pred = model.predict(X_test)
mae_test = mean_absolute_error(y_test, y_test_pred)
r2_test = r2_score(y_test, y_test_pred)

# 5. Красивый вывод
print("-" * 40)
print(f"{'МЕТРИКА':<10} | {'TRAIN':<10} | {'TEST':<10}")
print("-" * 40)
print(f"{'R2 Score':<10} | {r2_train:<10.3f} | {r2_test:<10.3f}")
print(f"{'MAE':<10} | {mae_train:<10.2f} | {mae_test:<10.2f}")
print("-" * 40)

# Интерпретация для пользователя
if r2_train - r2_test > 0.15:
    print("Внимание: Есть признаки переобучения (Train сильно лучше Test).")
    print("Можно попробовать уменьшить max_depth.")
else:
    print("Баланс Train/Test в норме.")

# 6. Сохранение
Path("models").mkdir(parents=True, exist_ok=True)
with open(MODEL_PATH, "wb") as f:
    pickle.dump(model, f)
print("\nМодель сохранена:", MODEL_PATH)