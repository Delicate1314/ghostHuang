"""用 DeepSeek API 异步并发补全电影元数据（快速版）"""
import asyncio
import json
import os
import time

import httpx
import pandas as pd

DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
DEEPSEEK_URL = "https://api.deepseek.com/chat/completions"

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


def parse_movies_dat(path="data/raw/ml-1m/ml-1m/movies.dat"):
    movies = pd.read_csv(
        path, sep="::", header=None,
        names=["movieId", "title", "genres"],
        engine="python", encoding="latin-1",
    )
    movies["year"] = movies["title"].str.extract(r"\((\d{4})\)")
    movies["clean_title"] = movies["title"].str.replace(r"\s*\(\d{4}\)\s*$", "", regex=True)
    movies["genre_list"] = movies["genres"].str.split("|")
    return movies


async def fetch_one(client: httpx.AsyncClient, mid: str, prompt: str, semaphore: asyncio.Semaphore):
    """单个电影请求，带信号量限制并发"""
    async with semaphore:
        payload = {
            "model": "deepseek-chat",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1,
            "max_tokens": 300,
            "response_format": {"type": "json_object"},
        }
        for attempt in range(3):
            try:
                resp = await client.post(
                    DEEPSEEK_URL,
                    json=payload,
                    headers={"Authorization": f"Bearer {DEEPSEEK_API_KEY}"},
                    timeout=30.0,
                )
                resp.raise_for_status()
                data = resp.json()
                text = data["choices"][0]["message"]["content"].strip()
                return mid, json.loads(text)
            except Exception as e:
                if attempt == 2:
                    return mid, None
                await asyncio.sleep(1 * (attempt + 1))


async def main():
    cache_path = "data/processed/enrich_cache.json"
    output_path = "data/processed/movies_meta.parquet"

    movies = parse_movies_dat()

    # 加载缓存
    cache = {}
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                cache = json.load(f)
        except Exception:
            cache = {}

    # 筛出待请求
    todo = []
    for _, row in movies.iterrows():
        mid = str(row["movieId"])
        if mid not in cache:
            prompt = ENRICH_PROMPT.format(
                title=row["clean_title"],
                year=row.get("year", "unknown"),
                genres=row["genres"],
            )
            todo.append((mid, prompt))

    print(f"已缓存 {len(cache)} 部，待请求 {len(todo)} 部")

    if todo:
        semaphore = asyncio.Semaphore(15)  # 最多 15 并发
        done = 0
        failed = 0
        start_time = time.time()

        async with httpx.AsyncClient() as client:
            # 分批处理，每批 100 个
            batch_size = 100
            for batch_start in range(0, len(todo), batch_size):
                batch = todo[batch_start:batch_start + batch_size]
                tasks = [fetch_one(client, mid, prompt, semaphore) for mid, prompt in batch]
                results = await asyncio.gather(*tasks)

                for mid, enriched in results:
                    if enriched:
                        cache[mid] = enriched
                    else:
                        cache[mid] = {"overview": "", "directors": [], "cast": [], "poster_description": ""}
                        failed += 1
                    done += 1

                # 每批保存缓存
                with open(cache_path, "w", encoding="utf-8") as f:
                    json.dump(cache, f, ensure_ascii=False)

                elapsed = time.time() - start_time
                speed = done / elapsed if elapsed > 0 else 0
                eta = (len(todo) - done) / speed if speed > 0 else 0
                print(f"进度: {done}/{len(todo)} | 速度: {speed:.1f} 部/秒 | ETA: {eta:.0f}秒 | 失败: {failed}")

    # 组装最终结果
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

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df = pd.DataFrame(results)
    df.to_parquet(output_path)
    print(f"\n完成！{len(df)} 部电影 → {output_path}")
    print(f"有概述: {sum(df['overview'].str.len() > 0)}/{len(df)}")


if __name__ == "__main__":
    asyncio.run(main())
