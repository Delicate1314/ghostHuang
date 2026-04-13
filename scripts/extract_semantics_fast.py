"""用 DeepSeek API 异步并发提取电影语义标签"""
import asyncio
import json
import os
import time

import httpx
import pandas as pd

DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
DEEPSEEK_URL = "https://api.deepseek.com/chat/completions"

SEMANTIC_PROMPT = """You are a movie analyst. Given the following movie information, extract structured semantic tags.

**Movie Title:** {title}
**Overview:** {overview}
**Genres:** {genres}
**Director:** {director}
**Main Cast:** {cast}
**Release Year:** {year}

Output a JSON object with EXACTLY these fields:
{{
  "genre_fine": ["<up to 3 fine-grained genre tags, e.g. psychological-thriller, romantic-comedy>"],
  "mood": "<one of: dark, tense, uplifting, melancholic, whimsical, intense, serene, humorous, dramatic, mysterious>",
  "theme": ["<up to 3 core themes, e.g. redemption, coming-of-age, survival>"],
  "pace": "<one of: fast, moderate, slow, varied>",
  "audience": "<one of: general, family, mature, cinephile, teen>",
  "narrative_style": "<one of: linear, nonlinear, ensemble, character-study, documentary-style, anthology>",
  "visual_style": "<one of: realistic, stylized, gritty, colorful, minimalist, epic>",
  "era_setting": "<time period of the story, e.g. modern, 1940s, futuristic, medieval>",
  "emotion_arc": "<brief 5-word description of emotional arc>"
}}

Output ONLY the JSON, no explanation."""


async def fetch_one(client: httpx.AsyncClient, mid: str, prompt: str, semaphore: asyncio.Semaphore):
    async with semaphore:
        payload = {
            "model": "deepseek-chat",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1,
            "max_tokens": 512,
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
            except Exception:
                if attempt == 2:
                    return mid, None
                await asyncio.sleep(1 * (attempt + 1))


async def main():
    cache_path = "data/processed/llm_semantic_cache.json"
    output_path = "data/processed/llm_semantic_tags.json"
    meta_path = "data/processed/movies_meta.parquet"

    df = pd.read_parquet(meta_path)
    print(f"电影总数: {len(df)}")

    # 加载缓存
    cache = {}
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                cache = json.load(f)
        except Exception:
            cache = {}

    # 构建待请求列表
    todo = []
    for _, row in df.iterrows():
        mid = str(row["tmdb_id"])
        if mid not in cache:
            genres = row.get("genres", [])
            if isinstance(genres, list):
                genres = ", ".join(genres)
            directors = row.get("directors", [])
            if isinstance(directors, list):
                directors = ", ".join(directors)
            cast = row.get("cast", [])
            if isinstance(cast, list):
                cast = ", ".join(cast[:5])

            prompt = SEMANTIC_PROMPT.format(
                title=row.get("title", ""),
                overview=str(row.get("overview", ""))[:800],
                genres=genres,
                director=directors,
                cast=cast,
                year=str(row.get("year", "")),
            )
            todo.append((mid, prompt))

    print(f"已缓存 {len(cache)} 部，待请求 {len(todo)} 部")

    if todo:
        semaphore = asyncio.Semaphore(15)
        done = 0
        failed = 0
        start_time = time.time()

        async with httpx.AsyncClient() as client:
            batch_size = 100
            for batch_start in range(0, len(todo), batch_size):
                batch = todo[batch_start:batch_start + batch_size]
                tasks = [fetch_one(client, mid, prompt, semaphore) for mid, prompt in batch]
                results = await asyncio.gather(*tasks)

                for mid, enriched in results:
                    if enriched:
                        cache[mid] = enriched
                    else:
                        failed += 1
                    done += 1

                with open(cache_path, "w", encoding="utf-8") as f:
                    json.dump(cache, f, ensure_ascii=False)

                elapsed = time.time() - start_time
                speed = done / elapsed if elapsed > 0 else 0
                eta = (len(todo) - done) / speed if speed > 0 else 0
                print(f"进度: {done}/{len(todo)} | 速度: {speed:.1f} 部/秒 | ETA: {eta:.0f}秒 | 失败: {failed}")

    # 保存最终结果
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)
    print(f"\n语义标签提取完成: {len(cache)} 部电影 → {output_path}")

    # 打印样例
    sample_id = list(cache.keys())[0]
    print(f"\n样例 (movieId={sample_id}):")
    print(json.dumps(cache[sample_id], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    asyncio.run(main())
