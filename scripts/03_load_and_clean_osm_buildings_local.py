import geopandas as gpd
import pandas as pd
from shapely.geometry import box
from pathlib import Path

# -----------------------------
# Пути к файлам
# -----------------------------
IN_PATH = Path("data/osm/perm_geojson_800mb.geojson") 
OUT_PATH = Path("data/osm/perm_buildings_clean.geojson")
POINTS_XLSX = Path("data/raw/perm_population_points.xlsx")
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

# -----------------------------
# 1. Загружаем точки населения для bbox
# -----------------------------
print("Загрузка точек населения...")
df_points = pd.read_excel(POINTS_XLSX)
gdf_points = gpd.GeoDataFrame(
    df_points,
    geometry=gpd.points_from_xy(df_points.Longitude, df_points.Latitude),
    crs="EPSG:4326"
)

# Создаём bounding box + буфер
buffer_deg = 0.05
minx, miny, maxx, maxy = gdf_points.total_bounds
bbox_geom = box(minx-buffer_deg, miny-buffer_deg, maxx+buffer_deg, maxy+buffer_deg)

# -----------------------------
# 2. Загружаем OSM
# -----------------------------
print("Загрузка OSM (фильтрация по bbox при чтении)...")
# Примечание: bbox поддерживается в последних версиях geopandas/fiona
try:
    gdf = gpd.read_file(
        IN_PATH,
        bbox=bbox_geom,
        columns=["geometry", "building", "building:levels"]
    )
except TypeError:
    # Если версия старая и bbox не поддерживается
    gdf = gpd.read_file(
        IN_PATH,
        columns=["geometry", "building", "building:levels"]
    )
    gdf = gdf[gdf.geometry.within(bbox_geom)]

print(f"Объектов в зоне: {len(gdf)}")

# -----------------------------
# 3. Фильтрация геометрии и площади
# -----------------------------
gdf = gdf[gdf.geometry.type.isin(["Polygon", "MultiPolygon"])]

# Считаем площадь в метрической системе
gdf_metric = gdf.to_crs(epsg=32640) # UTM зона 40N для Перми
gdf["area_total"] = gdf_metric.area 

# Удаляем здания < 10 м² (сараи) и > 15 000 м² (заводы, ошибочные полигоны районов)
gdf = gdf[
    (gdf["area_total"] >= 10) & 
    (gdf["area_total"] < 15000)
]
print(f"После фильтра площади (10 - 15000 кв.м): {len(gdf)}")

# -----------------------------
# 4. Обработка атрибутов
# -----------------------------
gdf["levels"] = pd.to_numeric(gdf.get("building:levels"), errors="coerce")
# Заменяем <= 0 или NaN на None, чтобы потом заполнить медианой или 1
gdf.loc[gdf["levels"] <= 0, "levels"] = None

# Фильтр жилых типов
residential_tags = ["residential", "apartments", "house", "detached", 
                    "semidetached_house", "terrace"]
# Оставляем жилые + те, где тег building пустой (часто бывает в OSM), но геометрически похож на дом
gdf = gdf[gdf["building"].isin(residential_tags) | gdf["building"].isna()]

print(f"После фильтра жилых: {len(gdf)}")

# -----------------------------
# 5. Сохранение
# -----------------------------
gdf = gdf.to_crs("EPSG:4326")
gdf.to_file(OUT_PATH, driver="GeoJSON")
print("Готово:", OUT_PATH)