"""文本 Embedding 生成（BGE-M3 本地，~600MB 显存）"""
import json
import os

import numpy as np


def build_text_embeddings(
    tags_path: str = "data/processed/llm_semantic_tags.json",
    output_path: str = "data/processed/text_embeddings.npy",
):
    from sentence_transformers import SentenceTransformer

    embed_model = SentenceTransformer("BAAI/bge-m3")

    with open(tags_path, "r") as f:
        tags = json.load(f)

    texts = []
    movie_ids = []
    for mid, t in tags.items():
        genre_fine = t.get("genre_fine", [])
        if isinstance(genre_fine, list):
            genre_fine = ", ".join(genre_fine)
        themes = t.get("theme", [])
        if isinstance(themes, list):
            themes = ", ".join(themes)

        desc = (
            f"Genre: {genre_fine}. "
            f"Mood: {t.get('mood', '')}. "
            f"Theme: {themes}. "
            f"Pace: {t.get('pace', '')}. "
            f"Audience: {t.get('audience', '')}. "
            f"Style: {t.get('narrative_style', '')}. "
            f"Visual: {t.get('visual_style', '')}. "
            f"Era: {t.get('era_setting', '')}. "
            f"Arc: {t.get('emotion_arc', '')}."
        )
        texts.append(desc)
        movie_ids.append(mid)

    print(f"编码 {len(texts)} 部电影的语义标签 ...")
    embeddings = embed_model.encode(
        texts, batch_size=64, show_progress_bar=True, normalize_embeddings=True
    )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.save(output_path, embeddings)
    with open(output_path.replace(".npy", "_ids.json"), "w") as f:
        json.dump(movie_ids, f)

    print(f"文本 embedding 完成：{embeddings.shape}")


if __name__ == "__main__":
    build_text_embeddings()
