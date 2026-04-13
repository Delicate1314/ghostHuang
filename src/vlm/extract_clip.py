"""视觉代理特征提取（用 BGE-M3 编码 poster_description 作为视觉风格表示）
与 text_embeddings 使用相同模型但输入不同：
- text_embeddings: 结构化语义标签（genre, mood, theme, pace...）
- clip_embeddings: 自然语言视觉描述（海报构图、色调、视觉风格）
"""
import json
import os

import numpy as np
import pandas as pd


def extract_clip_features(
    meta_path: str = "data/processed/movies_meta.parquet",
    output_path: str = "data/processed/clip_embeddings.npy",
    batch_size: int = 64,
):
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("BAAI/bge-m3")

    df = pd.read_parquet(meta_path)
    texts = []
    movie_ids = []
    for _, row in df.iterrows():
        desc = str(row.get("poster_description", ""))
        if not desc or desc == "nan":
            desc = f"A movie poster for {row.get('title', 'unknown film')}"
        texts.append(desc)
        movie_ids.append(str(row["movieId"]))

    print(f"编码 {len(texts)} 部电影的海报视觉描述 ...")
    embeddings = model.encode(
        texts, batch_size=batch_size, show_progress_bar=True, normalize_embeddings=True
    )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.save(output_path, embeddings)
    with open(output_path.replace(".npy", "_ids.json"), "w") as f:
        json.dump(movie_ids, f)

    print(f"视觉代理特征完成：{embeddings.shape}")


if __name__ == "__main__":
    extract_clip_features()
