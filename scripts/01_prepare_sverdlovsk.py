import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from pathlib import Path
import osmnx as ox
import numpy as np

RAW_PATH = Path("data/raw/sverdlovsk_buildings.xlsx")
OUT_PATH = Path("data/processed/sverdlovsk_buildings.geojson")

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

print("Загрузка XLSX...")
df = pd.read_excel(RAW_PATH)
print(f"Всего строк: {len(df)}")

# -----------------------------
# 1. Базовая фильтрация координат и населения
# -----------------------------
df = df[
    (df["LAT"].notna()) &
    (df["LON"].notna()) &
    (df["LAT"] != 0) &
    (df["LON"] != 0) &
    (df["INHAB"].notna()) &
    (df["INHAB"] > 0)
]

print(f"После фильтра координат и INHAB: {len(df)}")

# -----------------------------
# 2. Формируем единую площадь
# -----------------------------
df["area_total"] = df["AREA"]

mask_no_area = df["area_total"].isna()
df.loc[mask_no_area, "area_total"] = df.loc[mask_no_area, "AREA_LIVE"]

# если нет вообще никакой площади — выбрасываем
df = df[df["area_total"].notna()]

print(f"После формирования area_total: {len(df)}")

# -----------------------------
# 3. Удаляем явный мусор
# -----------------------------
df = df[
    (df["area_total"] >= 5)  # минимально возможная площадь
]

print(f"После фильтрации малых площадей: {len(df)}")

# -----------------------------
# 4. Геометрия
# -----------------------------
geometry = [
    Point(lon, lat) for lon, lat in zip(df["LON"], df["LAT"])
]

gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")

# -----------------------------
# 5. Пространственная валидация по региону
# -----------------------------
print("Загрузка границы Свердловской области...")
region = ox.geocode_to_gdf("Sverdlovsk Oblast, Russia").to_crs("EPSG:4326")

gdf = gdf[gdf.geometry.within(region.geometry.iloc[0])]

print(f"После фильтрации по региону: {len(gdf)}")

# -----------------------------
# 6. Финальные признаки
# -----------------------------
# -----------------------------
# Этажность (LEVELS)
# -----------------------------
gdf["levels_missing"] = gdf["LEVELS"].isna().astype(int)

gdf["levels"] = pd.to_numeric(
    gdf["LEVELS"],
    errors="coerce"
)

# импутация
mask_missing = gdf["levels"].isna()

gdf.loc[
    mask_missing & gdf["HOUSE_TYPE"].str.contains("Многоквартир", na=False),
    "levels"
] = 5

gdf.loc[
    mask_missing & ~gdf["HOUSE_TYPE"].str.contains("Многоквартир", na=False),
    "levels"
] = 1

# защита
gdf["levels"] = gdf["levels"].clip(lower=1)


print("Сохранение GeoJSON...")
gdf.to_file(OUT_PATH, driver="GeoJSON")
print("Сумма населения:", df["INHAB"].sum())
print("Диапазон площадей (area_total):", df["area_total"].min(), "-", df["area_total"].max())
print("Диапазон этажей (levels):", df["LEVELS"].min(), "-", df["LEVELS"].max())
print("Готово:", OUT_PATH)
