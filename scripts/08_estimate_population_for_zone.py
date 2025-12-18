import geopandas as gpd
import pandas as pd
import pickle
from pathlib import Path
import osmnx as ox
import numpy as np
from sklearn.metrics import mean_absolute_error

# =========================
# Пути
# =========================
BUILDINGS_OSM_PATH = Path("data/osm/perm_buildings_clean.geojson")
BUILDINGS_TRUE_PATH = Path("data/processed/perm_buildings_with_population.geojson") 
POINTS_PATH = Path("data/processed/perm_population_points.geojson")
MODEL_PATH = Path("models/population_model.pkl")

OUT_PREDICTION = Path("data/processed/perm_buildings_population_pred.geojson")
OUT_DISTRICTS = Path("data/processed/perm_districts_comparison.geojson")

# =========================
# 1. Загрузка и Подготовка
# =========================
print("Загрузка зданий OSM...")
gdf = gpd.read_file(BUILDINGS_OSM_PATH)

# Преобразование в метры
gdf_metric = gdf.to_crs(epsg=32640)
gdf["area"] = gdf_metric.area

# Фильтр выбросов (на случай, если что-то просочилось)
gdf = gdf[gdf["area"] < 20000]

# Подготовка признаков
gdf["levels"] = pd.to_numeric(gdf.get("levels"), errors="coerce").fillna(1).clip(lower=1)
gdf["total_floor_area"] = gdf["area"] * gdf["levels"]

X = gdf[["area", "levels", "total_floor_area"]]

# Загрузка целевой суммы
gdf_points = gpd.read_file(POINTS_PATH)
REAL_TOTAL_POP = gdf_points["population"].sum()
print(f"Целевое население (Пермь): {REAL_TOTAL_POP:,.0f}")

# =========================
# 2. Предсказание (ML)
# =========================
print("Загрузка модели и прогноз...")
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

# Сырое предсказание
gdf["pop_pred_raw"] = model.predict(X)

# Нормализация
raw_sum = gdf["pop_pred_raw"].sum()
normalization_coef = REAL_TOTAL_POP / raw_sum
print(f"Сырая сумма ML: {raw_sum:,.0f} -> Коэфф: {normalization_coef:.4f}")

gdf["population_pred"] = gdf["pop_pred_raw"] * normalization_coef
print(f"Итоговая сумма ML: {gdf['population_pred'].sum():,.0f}")

# Сохранение результата (дома)
gdf.to_crs(4326).to_file(OUT_PREDICTION, driver="GeoJSON")

# =========================
# 3. Валидация по районам
# =========================
print("\n--- ВАЛИДАЦИЯ ПО РАЙОНАМ ---")

    # ИСПОЛЬЗУЕМ РУССКИЕ НАЗВАНИЯ
district_names = [
        "Ленинский район, Пермь",
        "Свердловский район, Пермь",
        "Дзержинский район, Пермь",
        "Индустриальный район, Пермь",
        "Кировский район, Пермь",
        "Мотовилихинский район, Пермь",
        "Орджоникидзевский район, Пермь"
        # "Новые Ляды" часто идут как поселок, а не район, их можно исключить, чтобы не ломать скрипт
]

try:
    print(f"Скачивание границ для {len(district_names)} районов...")
        
    # Скачиваем границы
    districts = ox.geocode_to_gdf(district_names).to_crs(gdf.crs)
        
    # Оставляем короткое имя для таблицы
    districts["name_short"] = districts["display_name"].apply(lambda x: x.split(" ")[0])

    # Загружаем "Истину" (Ground Truth из скрипта 05)
    gdf_true = gpd.read_file(BUILDINGS_TRUE_PATH).to_crs(gdf.crs)
    
    # Агрегация ИСТИНЫ
    # Считаем, сколько людей в каждом районе по факту
    true_join = gpd.sjoin(gdf_true, districts, how="inner", predicate="within")
    true_stats = true_join.groupby("name_short")["population"].sum().rename("pop_true")
    
    # Агрегация ПРЕДСКАЗАНИЯ
    # Считаем, сколько людей модель "поселила" в каждый район
    pred_join = gpd.sjoin(gdf, districts, how="inner", predicate="within")
    pred_stats = pred_join.groupby("name_short")["population_pred"].sum().rename("pop_pred")
    
    # Сведение таблицы
    comparison = pd.concat([true_stats, pred_stats], axis=1).fillna(0)
    
    comparison["diff"] = comparison["pop_pred"] - comparison["pop_true"]
    comparison["error_pct"] = (comparison["diff"] / comparison["pop_true"] * 100).round(1)
    
    print("\nСравнение Модель vs Факт:")
    pd.options.display.float_format = '{:,.0f}'.format
    print(comparison[["pop_true", "pop_pred", "diff", "error_pct"]])
    
    # Считаем MAE по районам
    mae_dist = mean_absolute_error(comparison["pop_true"], comparison["pop_pred"])
    print(f"\nСредняя ошибка (MAE) по району: {mae_dist:,.0f} чел.")
    
    # Сохраняем границы районов с данными (для раскраски карты)
    districts = districts.merge(comparison, on="name_short", how="left")
    districts = districts[["name_short", "geometry", "pop_true", "pop_pred", "error_pct"]]
    districts.to_file(OUT_DISTRICTS, driver="GeoJSON")
    print(f"✅ Файл районов сохранен: {OUT_DISTRICTS}")

except Exception as e:
    print(f"Ошибка при валидации районов: {e}")
    pass
