"""语义 ID 构建：显式标签 token + 隐式层次化量化 token"""
import json
import os

import numpy as np
from sklearn.cluster import KMeans


def build_explicit_tokens(tags_path: str) -> dict:
    """从 LLM 语义标签中提取显式 token（固定维度字段）"""
    with open(tags_path, "r", encoding="utf-8") as f:
        tags = json.load(f)

    # 收集所有可能的标签值，构建词表
    vocab = {"<PAD>": 0, "<UNK>": 1}

    # 需要编码的字段及类型
    fields = [
        ("mood", "str"),
        ("pace", "str"),
        ("audience", "str"),
        ("narrative_style", "str"),
        ("visual_style", "str"),
        ("era_setting", "str"),
    ]
    list_fields = [
        ("genre_fine", "list"),
    ]

    # 第一遍：构建词表
    for mid, t in tags.items():
        for field, ftype in fields:
            val = t.get(field, "")
            if val and val not in vocab:
                vocab[val] = len(vocab)
        for field, ftype in list_fields:
            vals = t.get(field, [])
            if isinstance(vals, str):
                vals = [vals]
            for v in vals:
                if v and v not in vocab:
                    vocab[v] = len(vocab)

    print(f"显式 token 词表大小：{len(vocab)}")

    # 第二遍：构建每部电影的显式 token 序列
    # 格式: [genre1, genre2, genre3, mood, pace, audience, narrative, visual, era]
    max_genres = 3
    explicit_tokens = {}

    for mid, t in tags.items():
        tokens = []

        # genre_fine（取前3个）
        genres = t.get("genre_fine", [])
        if isinstance(genres, str):
            genres = [genres]
        for i in range(max_genres):
            if i < len(genres):
                tokens.append(vocab.get(genres[i], 1))
            else:
                tokens.append(0)  # PAD

        # 固定字段
        for field, _ in fields:
            val = t.get(field, "")
            tokens.append(vocab.get(val, 1))

        explicit_tokens[mid] = tokens

    token_len = max_genres + len(fields)
    print(f"显式 token 长度：{token_len}（3 genre + {len(fields)} 属性）")
    return explicit_tokens, vocab, token_len


def build_implicit_tokens(
    fused_path: str,
    ids_path: str,
    n_levels: int = 3,
    n_clusters: int = 32,
) -> dict:
    """层次化 K-Means 量化，生成多级隐式 token"""
    fused = np.load(fused_path)
    with open(ids_path, "r") as f:
        ids = json.load(f)

    implicit_tokens = {mid: [] for mid in ids}

    residual = fused.copy()
    offset = 0  # token ID 偏移，避免不同层级冲突

    for level in range(n_levels):
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(residual)
        centroids = kmeans.cluster_centers_

        for i, mid in enumerate(ids):
            implicit_tokens[mid].append(int(labels[i]) + offset)

        # 残差：原向量减去聚类中心
        residual = residual - centroids[labels]
        offset += n_clusters

        inertia = kmeans.inertia_ / len(fused)
        print(f"Level {level+1}: {n_clusters} clusters, avg inertia: {inertia:.4f}")

    total_implicit_vocab = offset
    print(f"隐式 token 词表大小：{total_implicit_vocab}（{n_levels} 级 × {n_clusters} 聚类）")
    return implicit_tokens, total_implicit_vocab, n_levels


def build_semantic_ids(
    tags_path: str = "data/processed/llm_semantic_tags.json",
    fused_path: str = "data/processed/fused_embeddings.npy",
    fused_ids_path: str = "data/processed/fused_embeddings_ids.json",
    output_path: str = "data/processed/semantic_ids.json",
    vocab_path: str = "data/processed/sid_vocab.json",
    n_levels: int = 3,
    n_clusters: int = 32,
):
    print("=" * 50)
    print("构建语义 ID")
    print("=" * 50)

    # 1. 显式 token
    explicit_tokens, explicit_vocab, explicit_len = build_explicit_tokens(tags_path)

    # 2. 隐式 token
    implicit_tokens, implicit_vocab_size, n_levels_out = build_implicit_tokens(
        fused_path, fused_ids_path, n_levels, n_clusters
    )

    # 3. 合并：semantic_id = [explicit_tokens..., implicit_tokens...]
    # 隐式 token 的 ID 需要偏移，避免和显式 token 冲突
    explicit_vocab_size = len(explicit_vocab)

    semantic_ids = {}
    for mid in explicit_tokens:
        exp = explicit_tokens[mid]  # list of int
        imp = implicit_tokens.get(mid, [0] * n_levels_out)
        # 隐式 token 偏移
        imp_shifted = [t + explicit_vocab_size for t in imp]
        semantic_ids[mid] = exp + imp_shifted

    total_vocab_size = explicit_vocab_size + implicit_vocab_size
    sid_length = explicit_len + n_levels_out

    print(f"\n总 vocab 大小：{total_vocab_size}")
    print(f"语义 ID 长度：{sid_length}（{explicit_len} 显式 + {n_levels_out} 隐式）")
    print(f"覆盖电影数：{len(semantic_ids)}")

    # 保存
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(semantic_ids, f, ensure_ascii=False)

    vocab_info = {
        "explicit_vocab": explicit_vocab,
        "explicit_vocab_size": explicit_vocab_size,
        "implicit_vocab_size": implicit_vocab_size,
        "total_vocab_size": total_vocab_size,
        "explicit_len": explicit_len,
        "implicit_len": n_levels_out,
        "sid_length": sid_length,
        "n_clusters": n_clusters,
    }
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(vocab_info, f, ensure_ascii=False, indent=2)

    # 打印样例
    sample_ids = list(semantic_ids.keys())[:3]
    for mid in sample_ids:
        print(f"  Movie {mid}: {semantic_ids[mid]}")

    return semantic_ids, vocab_info


if __name__ == "__main__":
    build_semantic_ids()
