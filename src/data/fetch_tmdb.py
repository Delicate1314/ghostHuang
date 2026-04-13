"""从 TMDb 获取电影元数据"""
import requests
import json
import time
import os

import pandas as pd

TMDB_API_KEY = os.environ.get("TMDB_API_KEY", "")
BASE_URL = "https://api.themoviedb.org/3"


def fetch_movie_details(tmdb_id: int, retries: int = 3) -> dict | None:
    url = f"{BASE_URL}/movie/{tmdb_id}"
    params = {
        "api_key": TMDB_API_KEY,
        "language": "en-US",
        "append_to_response": "credits",
    }
    for attempt in range(retries):
        try:
            resp = requests.get(url, params=params, timeout=20)
            if resp.status_code != 200:
                return None
            data = resp.json()
            return {
                "tmdb_id": tmdb_id,
                "title": data.get("title"),
                "overview": data.get("overview", ""),
                "genres": [g["name"] for g in data.get("genres", [])],
                "release_date": data.get("release_date", ""),
                "vote_average": data.get("vote_average", 0),
                "poster_path": data.get("poster_path"),
                "directors": [
                    c["name"]
                    for c in data.get("credits", {}).get("crew", [])
                    if c["job"] == "Director"
                ],
                "cast": [
                    c["name"] for c in data.get("credits", {}).get("cast", [])[:10]
                ],
            }
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2 * (attempt + 1))
            else:
                print(f"获取 tmdb_id={tmdb_id} 失败: {e}")
                return None


def batch_fetch(
    links_csv: str,
    output_path: str,
    cache_path: str = "data/raw/tmdb_cache.json",
    filter_movies: str = None,
):
    if not TMDB_API_KEY:
        raise ValueError(
            "请设置环境变量 TMDB_API_KEY，去 https://www.themoviedb.org/settings/api 免费申请"
        )

    links = pd.read_csv(links_csv)

    # 如果指定了过滤文件，只获取其中的电影
    if filter_movies:
        if filter_movies.endswith(".dat"):
            movies_df = pd.read_csv(
                filter_movies, sep="::", header=None,
                names=["movieId", "title", "genres"],
                engine="python", encoding="latin-1",
            )
        else:
            movies_df = pd.read_csv(filter_movies)
        valid_ids = set(movies_df["movieId"].unique())
        links = links[links["movieId"].isin(valid_ids)]
        print(f"过滤后：{len(links)} 部电影（原 links 共 {len(pd.read_csv(links_csv))} 部）")

    cache = {}
    if os.path.exists(cache_path):
        with open(cache_path, "r") as f:
            cache = json.load(f)

    results = []
    for idx, row in links.iterrows():
        tmdb_id = str(int(row["tmdbId"])) if pd.notna(row.get("tmdbId")) else None
        movie_id = int(row["movieId"])
        if tmdb_id is None:
            continue
        if tmdb_id in cache:
            item = cache[tmdb_id].copy()
            item["movieId"] = movie_id
            results.append(item)
            continue

        detail = fetch_movie_details(int(tmdb_id))
        if detail:
            cache[tmdb_id] = detail
            item = detail.copy()
            item["movieId"] = movie_id
            results.append(item)

        time.sleep(0.05)

        if len(results) % 500 == 0 and len(results) > 0:
            with open(cache_path, "w") as f:
                json.dump(cache, f)
            print(f"已获取 {len(results)} 部电影")

    with open(cache_path, "w") as f:
        json.dump(cache, f)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    pd.DataFrame(results).to_parquet(output_path)
    print(f"完成！共 {len(results)} 部电影 → {output_path}")


if __name__ == "__main__":
    batch_fetch(
        links_csv="data/raw/ml-latest-small/links.csv",
        output_path="data/processed/movies_meta.parquet",
        filter_movies="data/raw/ml-1m/ml-1m/movies.dat",
    )
