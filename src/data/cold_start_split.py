"""冷启动划分：随机选 15% 电影作为新电影"""
import json
import os

import numpy as np
import pandas as pd


def make_cold_start_split(
    ratings_path: str,
    output_dir: str = "data/splits",
    cold_ratio: float = 0.15,
    seed: int = 42,
):
    if ratings_path.endswith(".dat"):
        df = pd.read_csv(
            ratings_path,
            sep="::",
            header=None,
            names=["userId", "movieId", "rating", "timestamp"],
            engine="python",
            encoding="latin-1",
        )
    else:
        df = pd.read_csv(ratings_path)

    np.random.seed(seed)
    all_movies = df["movieId"].unique()
    n_cold = int(len(all_movies) * cold_ratio)
    cold_movies = set(np.random.choice(all_movies, n_cold, replace=False).tolist())

    os.makedirs(output_dir, exist_ok=True)
    cold_path = os.path.join(output_dir, "cold_movies.json")
    with open(cold_path, "w") as f:
        json.dump(list(cold_movies), f)

    print(f"冷启动电影数：{len(cold_movies)}/{len(all_movies)} ({cold_ratio*100:.0f}%)")
    print(f"已保存 → {cold_path}")
    return cold_movies


if __name__ == "__main__":
    make_cold_start_split("data/raw/ml-1m/ml-1m/ratings.dat")
