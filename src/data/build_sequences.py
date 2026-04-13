"""构建用户观影序列 + leave-last-one-out 划分"""
import json
import os

import pandas as pd
import numpy as np


def build_sequences(
    ratings_path: str,
    output_dir: str = "data/splits",
    cold_movies_path: str = None,
    min_len: int = 5,
    max_len: int = 50,
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

    # 加载冷启动电影集合
    cold_movies = set()
    if cold_movies_path and os.path.exists(cold_movies_path):
        with open(cold_movies_path, "r") as f:
            cold_movies = set(json.load(f))
        print(f"冷启动电影数：{len(cold_movies)}，将从训练序列中移除")

    # 只保留正反馈
    df = df[df["rating"] >= 4].copy()
    df = df.sort_values(["userId", "timestamp"])

    sequences = df.groupby("userId")["movieId"].apply(list).reset_index()
    sequences.columns = ["userId", "movie_seq"]

    # 过滤过短
    sequences = sequences[sequences["movie_seq"].apply(len) >= min_len]
    # 截断过长
    sequences["movie_seq"] = sequences["movie_seq"].apply(lambda x: x[-max_len:])

    # Leave-last-one-out
    train, val, test = [], [], []
    for _, row in sequences.iterrows():
        uid = int(row["userId"])
        seq = [int(m) for m in row["movie_seq"]]
        if len(seq) < 3:
            continue
        # 训练序列中移除冷启动电影
        train_seq = [m for m in seq[:-2] if m not in cold_movies]
        if len(train_seq) < 2:
            continue
        train.append({"userId": uid, "movie_seq": train_seq})
        val.append({"userId": uid, "movie_seq": train_seq, "target": seq[-2]})
        test.append({"userId": uid, "movie_seq": seq[:-1], "target": seq[-1]})

    os.makedirs(output_dir, exist_ok=True)
    for name, data in [("train", train), ("val", val), ("test", test)]:
        path = os.path.join(output_dir, f"{name}.json")
        with open(path, "w") as f:
            json.dump(data, f)
        print(f"{name}: {len(data)} 条 → {path}")

    print(f"\n用户总数: {len(train)}")
    seq_lens = [len(d["movie_seq"]) for d in train]
    print(f"序列长度: min={min(seq_lens)}, max={max(seq_lens)}, avg={np.mean(seq_lens):.1f}")


if __name__ == "__main__":
    build_sequences(
        "data/raw/ml-1m/ml-1m/ratings.dat",
        cold_movies_path="data/splits/cold_movies.json",
    )
