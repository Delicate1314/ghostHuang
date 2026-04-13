"""用 DeepSeek API 批量提取电影语义标签"""
import json
import os
import time

import pandas as pd
from tqdm import tqdm

from .client import get_client
from .prompts import SEMANTIC_EXTRACTION_PROMPT


def extract_one(client, movie: dict) -> dict | None:
    prompt = SEMANTIC_EXTRACTION_PROMPT.format(
        title=movie.get("title", ""),
        overview=movie.get("overview", ""),
        genres=", ".join(movie.get("genres", [])) if isinstance(movie.get("genres"), list) else str(movie.get("genres", "")),
        director=", ".join(movie.get("directors", [])) if isinstance(movie.get("directors"), list) else str(movie.get("directors", "")),
        cast=", ".join(movie.get("cast", [])[:5]) if isinstance(movie.get("cast"), list) else str(movie.get("cast", "")),
        year=str(movie.get("release_date", ""))[:4],
    )

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=512,
        response_format={"type": "json_object"},
    )

    text = response.choices[0].message.content.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def batch_extract(
    meta_path: str = "data/processed/movies_meta.parquet",
    output_path: str = "data/processed/llm_semantic_tags.json",
    cache_path: str = "data/processed/llm_cache.json",
):
    client = get_client()
    df = pd.read_parquet(meta_path)

    cache = {}
    if os.path.exists(cache_path):
        with open(cache_path, "r") as f:
            cache = json.load(f)

    results = {}
    failed = 0
    for _, row in tqdm(df.iterrows(), total=len(df), desc="LLM 语义提取"):
        mid = str(row["tmdb_id"])
        if mid in cache:
            results[mid] = cache[mid]
            continue

        try:
            tags = extract_one(client, row.to_dict())
            if tags:
                results[mid] = tags
                cache[mid] = tags
            else:
                failed += 1
        except Exception as e:
            failed += 1
            print(f"\n电影 {mid} 失败: {e}")
            time.sleep(1)

        if len(results) % 200 == 0 and len(results) > 0:
            with open(cache_path, "w") as f:
                json.dump(cache, f, ensure_ascii=False)

    # 最终保存
    with open(cache_path, "w") as f:
        json.dump(cache, f, ensure_ascii=False)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n语义标签提取完成：成功 {len(results)} 部，失败 {failed} 部")


if __name__ == "__main__":
    batch_extract()
