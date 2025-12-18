import pandas as pd
import geopandas as gpd
from pathlib import Path

RAW_PATH = Path("data/raw/perm_population_points.xlsx")
OUT_PATH = Path("data/processed/perm_population_points.geojson")

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

print("Загрузка XLSX...")
df = pd.read_excel(RAW_PATH)

gdf = gpd.GeoDataFrame(
    df,
    geometry=gpd.points_from_xy(
        df["Longitude"],
        df["Latitude"]
    ),
    crs="EPSG:4326"
)

gdf = gdf.rename(columns={"ЧН_Расчет": "population"})

print("Сохранение GeoJSON...")
gdf.to_file(OUT_PATH, driver="GeoJSON")

print("Готово:", OUT_PATH)
