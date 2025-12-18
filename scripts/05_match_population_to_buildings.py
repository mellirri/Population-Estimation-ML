# scripts/05_match_population_to_buildings_dasymetric.py
"""
Корректное распределение населения от точек к зданиям (dasymetric mapping)

Логика:
1. Для каждой точки населения ищем ВСЕ жилые здания в радиусе R
2. Считаем вес здания = area * levels
3. Распределяем population точки пропорционально весам
4. Агрегируем вклад от всех точек к каждому зданию
"""

import geopandas as gpd
import pandas as pd
from pathlib import Path
import numpy as np

# -----------------------------
# Пути
# -----------------------------
BUILDINGS_PATH = Path("data/osm/perm_buildings_clean.geojson")
POINTS_PATH = Path("data/processed/perm_population_points.geojson")
OUT_PATH = Path("data/processed/perm_buildings_with_population.geojson")

RADIUS_METERS = 300  # радиус поиска зданий вокруг точки

# -----------------------------
# Загрузка данных
# -----------------------------
print("Загрузка зданий...")
buildings = gpd.read_file(BUILDINGS_PATH).to_crs(3857)
print(f"Всего зданий: {len(buildings)}")

print("Загрузка точек населения...")
points = gpd.read_file(POINTS_PATH).to_crs(3857)
print(f"Всего точек населения: {len(points)}")

# -----------------------------
# Подготовка признаков зданий
# -----------------------------
buildings["levels"] = pd.to_numeric(buildings.get("levels"), errors="coerce").fillna(1)
buildings["area"] = buildings.geometry.area
buildings["weight"] = buildings["area"] * buildings["levels"]
buildings["population"] = 0.0

# spatial index
b_sindex = buildings.sindex

# -----------------------------
# Распределение населения
# -----------------------------
print("Распределение населения по зданиям...")

for idx, row in points.iterrows():
    pop = row["population"]
    if pop <= 0:
        continue

    buffer_geom = row.geometry.buffer(RADIUS_METERS)
    possible_idx = list(b_sindex.intersection(buffer_geom.bounds))
    nearby = buildings.iloc[possible_idx]
    nearby = nearby[nearby.geometry.intersects(buffer_geom)]

    if len(nearby) == 0:
        continue

    total_weight = nearby["weight"].sum()
    if total_weight <= 0:
        continue

    share = pop * (nearby["weight"] / total_weight)
    buildings.loc[nearby.index, "population"] += share.values

# -----------------------------
# Финальная очистка
# -----------------------------
buildings = buildings[buildings["population"] > 0]
buildings = buildings.to_crs(4326)

# -----------------------------
# Сохранение
# -----------------------------
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
buildings.to_file(OUT_PATH, driver="GeoJSON")

print("Готово!")
print(f"Жилых зданий с населением: {len(buildings)}")
print(f"Суммарное население: {buildings['population'].sum():,.0f}")
