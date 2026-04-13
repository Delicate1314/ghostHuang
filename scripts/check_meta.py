import pandas as pd

df = pd.read_parquet("data/processed/movies_meta.parquet")
print("=== 元数据概览 ===")
print(f"电影数: {len(df)}")

has_overview = sum(df["overview"].str.len() > 0)
print(f"有概述: {has_overview}")

has_dir = sum(df["directors"].apply(lambda x: len(x) > 0 if isinstance(x, list) else False))
print(f"有导演: {has_dir}")

has_cast = sum(df["cast"].apply(lambda x: len(x) > 0 if isinstance(x, list) else False))
print(f"有演员: {has_cast}")

print("\n=== 样例: Toy Story ===")
row = df[df["title"].str.contains("Toy Story", na=False)].iloc[0]
for col in ["title", "year", "genres", "overview", "directors", "cast", "poster_description"]:
    print(f"  {col}: {row[col]}")
