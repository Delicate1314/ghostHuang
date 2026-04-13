"""直接从 ML-1M 的 movies.dat 构建元数据，用 LLM 世界知识补全"""
import json
import os
import time

import pandas as pd
from tqdm import tqdm


def parse_movies_dat(movies_path: str = "data/raw/ml-1m/ml-1m/movies.dat") -> pd.DataFrame:
    """解析 ML-1M 的 movies.dat"""
    movies = pd.read_csv(
        movies_path,
        sep="::",
        header=None,
        names=["movieId", "title", "genres"],
        engine="python",
        encoding="latin-1",
    )
    # 提取年份
    movies["year"] = movies["title"].str.extract(r"\((\d{4})\)")
    movies["clean_title"] = movies["title"].str.replace(r"\s*\(\d{4}\)\s*$", "", regex=True)
    movies["genre_list"] = movies["genres"].str.split("|")
    return movies


def enrich_with_llm(
    movies_path: str = "data/raw/ml-1m/ml-1m/movies.dat",
    output_path: str = "data/processed/movies_meta.parquet",
    cache_path: str = "data/processed/enrich_cache.json",
):
    """用 DeepSeek 的世界知识补全电影信息（单线程，稳定可靠）"""
    from src.llm.client import get_client

    movies = parse_movies_dat(movies_path)
    client = get_client()

    cache = {}
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                cache = json.load(f)
        except (json.JSONDecodeError, Exception):
            print("缓存文件损坏，从头开始")
            cache = {}

    ENRICH_PROMPT = """Given this movie title and genres, provide factual information in JSON format.

Movie: {title} ({year})
Genres: {genres}

Output JSON with these fields:
{{
  "overview": "<1-2 sentence plot summary>",
  "directors": ["<director name(s)>"],
  "cast": ["<top 5 actor names>"],
  "poster_description": "<1 sentence describing the likely poster visual style>"
}}

If you don't know the movie, make reasonable inferences from the title and genre.
Output ONLY the JSON."""

    # 筛出需要请求的
    todo_rows = []
    for _, row in movies.iterrows():
        mid = str(row["movieId"])
        if mid not in cache:
            todo_rows.append(row)

    print(f"已缓存 {len(cache)} 部，待请求 {len(todo_rows)} 部")

    for i, row in enumerate(tqdm(todo_rows, desc="LLM 补全电影信息")):
        mid = str(row["movieId"])
        prompt = ENRICH_PROMPT.format(
            title=row["clean_title"],
            year=row.get("year", "unknown"),
            genres=row["genres"],
        )
        for attempt in range(3):
            try:
                response = client.chat.completions.create(
                    model="deepseek-chat",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                    max_tokens=300,
                    response_format={"type": "json_object"},
                    timeout=30,
                )
                enriched = json.loads(response.choices[0].message.content.strip())
                cache[mid] = enriched
                break
            except Exception as e:
                if attempt == 2:
                    cache[mid] = {
                        "overview": "", "directors": [], "cast": [],
                        "poster_description": "",
                    }
                    tqdm.write(f"电影 {mid} 失败: {e}")
                else:
                    time.sleep(2)

        # 每 50 部保存一次缓存
        if (i + 1) % 50 == 0:
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(cache, f, ensure_ascii=False)
            tqdm.write(f"  缓存已保存: {len(cache)} 部")

    # 最终保存缓存
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False)

    # 组装结果
    results = []
    for _, row in movies.iterrows():
        mid = str(row["movieId"])
        enriched = cache.get(mid, {})
        results.append({
            "movieId": int(row["movieId"]),
            "tmdb_id": int(row["movieId"]),
            "title": row["clean_title"],
            "year": row.get("year", ""),
            "genres": row["genre_list"],
            "overview": enriched.get("overview", ""),
            "directors": enriched.get("directors", []),
            "cast": enriched.get("cast", []),
            "poster_description": enriched.get("poster_description", ""),
        })

    # 最终保存缓存
    with open(cache_path, "w") as f:
        json.dump(cache, f, ensure_ascii=False)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df = pd.DataFrame(results)
    df.to_parquet(output_path)
    print(f"\n元数据构建完成：{len(df)} 部电影 → {output_path}")
    print(f"有概述的电影：{sum(df['overview'].str.len() > 0)}/{len(df)}")
    return df


if __name__ == "__main__":
    enrich_with_llm()
