"""通过 DeepSeek API 生成海报视觉风格描述"""
import json
import os
import time

import pandas as pd
from tqdm import tqdm

from ..llm.client import get_client
from ..llm.prompts import CAPTION_PROMPT


def generate_captions(
    meta_path: str = "data/processed/movies_meta.parquet",
    output_path: str = "data/processed/poster_captions.json",
    cache_path: str = "data/processed/caption_cache.json",
):
    client = get_client()
    df = pd.read_parquet(meta_path)

    cache = {}
    if os.path.exists(cache_path):
        with open(cache_path, "r") as f:
            cache = json.load(f)

    results = {}
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Caption 生成"):
        mid = str(row["tmdb_id"])
        if mid in cache:
            results[mid] = cache[mid]
            continue

        try:
            genres = row.get("genres", [])
            if isinstance(genres, list):
                genres = ", ".join(genres)

            directors = row.get("directors", [])
            if isinstance(directors, list):
                directors = ", ".join(directors)

            prompt = CAPTION_PROMPT.format(
                title=row.get("title", ""),
                year=str(row.get("release_date", ""))[:4],
                genres=genres,
                overview=str(row.get("overview", ""))[:500],
                director=directors,
            )

            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=150,
            )

            caption = response.choices[0].message.content.strip()
            results[mid] = caption
            cache[mid] = caption
        except Exception as e:
            print(f"\n跳过 {mid}: {e}")
            time.sleep(1)

        if len(results) % 200 == 0 and len(results) > 0:
            with open(cache_path, "w") as f:
                json.dump(cache, f, ensure_ascii=False)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nCaption 生成完成：{len(results)} 部电影")


if __name__ == "__main__":
    generate_captions()
