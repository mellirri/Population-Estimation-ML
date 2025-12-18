import geopandas as gpd
from pathlib import Path
import pandas as pd

# =========================
# Пути
# =========================
SVERD_PATH = Path("data/processed/sverdlovsk_buildings.geojson")
OUT_PATH = Path("data/processed/training_buildings.geojson")

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

print("Загрузка данных Свердловской области...")
gdf = gpd.read_file(SVERD_PATH)

# =========================
# Подготовка признаков
# =========================
# Целевая переменная
gdf["population"] = gdf["INHAB"]

# Базовые признаки
if "area" not in gdf.columns:
    gdf["area"] = gdf["area_total"]

if "levels" not in gdf.columns:
    gdf["levels"] = 1

# НОВЫЙ ПРИЗНАК: Общая площадь 
gdf["total_floor_area"] = gdf["area"] * gdf["levels"]

# Очистка от нулей и ошибок
gdf = gdf[
    (gdf["population"] > 0) &
    (gdf["area"] >= 10) &
    (gdf["levels"] >= 1) &
    (gdf["total_floor_area"] > 0)
]

# Выбор колонок для обучения
gdf_train = gdf[[
    "geometry",
    "area",
    "levels",
    "total_floor_area", # <-- Добавили
    "population"
]]

# =========================
# Сохранение
# =========================
gdf_train.to_file(OUT_PATH, driver="GeoJSON")
print("Training dataset готов:", OUT_PATH)
print(f"Примеров для обучения: {len(gdf_train)}")