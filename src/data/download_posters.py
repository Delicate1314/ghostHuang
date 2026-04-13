"""下载电影海报"""
import os
from concurrent.futures import ThreadPoolExecutor

import requests
import pandas as pd

POSTER_BASE = "https://image.tmdb.org/t/p/w500"


def download_one(args):
    tmdb_id, poster_path, save_dir = args
    if poster_path is None or (isinstance(poster_path, float)):
        return False
    url = f"{POSTER_BASE}{poster_path}"
    save_path = os.path.join(save_dir, f"{tmdb_id}.jpg")
    if os.path.exists(save_path):
        return True
    try:
        resp = requests.get(url, timeout=15)
        if resp.status_code == 200:
            with open(save_path, "wb") as f:
                f.write(resp.content)
            return True
    except Exception:
        pass
    return False


def batch_download(
    meta_path: str, save_dir: str = "data/raw/posters", workers: int = 8
):
    os.makedirs(save_dir, exist_ok=True)
    df = pd.read_parquet(meta_path)
    tasks = [
        (row["tmdb_id"], row.get("poster_path"), save_dir)
        for _, row in df.iterrows()
    ]

    success = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for ok in pool.map(download_one, tasks):
            if ok:
                success += 1

    print(f"海报下载完成：{success}/{len(tasks)}")


if __name__ == "__main__":
    batch_download("data/processed/movies_meta.parquet")
