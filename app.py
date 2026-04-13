"""电影推荐系统 Web 服务"""
import json
import csv
import os
import re

import numpy as np
import torch
from flask import Flask, jsonify, request, send_from_directory

from src.model.evaluate import load_model, build_sid_matrix, generate_sid, match_sid_to_movies

app = Flask(__name__, static_folder="web", static_url_path="")

# ── 全局数据（启动时加载一次） ──────────────────────────────────
MODEL = None
CHECKPOINT = None
SEMANTIC_IDS = None
MOVIE_IDS = None
SID_MATRIX = None
MOVIES_META = {}  # movie_id -> {title, year, genres, overview, directors, cast, tags, tmdb_id}
CN_TITLES = {}   # movie_id -> 中文标题


def init():
    """加载模型和所有元数据"""
    global MODEL, CHECKPOINT, SEMANTIC_IDS, MOVIE_IDS, SID_MATRIX, MOVIES_META, CN_TITLES

    device = "cpu"
    MODEL, CHECKPOINT = load_model("checkpoints/sid_model.pt", device)

    with open("data/processed/semantic_ids.json", "r", encoding="utf-8") as f:
        SEMANTIC_IDS = json.load(f)

    MOVIE_IDS, SID_MATRIX = build_sid_matrix(SEMANTIC_IDS)

    # 加载 movies.dat (movieId::Title (Year)::Genres)
    movies_dat = {}
    with open("data/raw/ml-1m/ml-1m/movies.dat", "r", encoding="latin-1") as f:
        for line in f:
            parts = line.strip().split("::")
            if len(parts) >= 3:
                mid = parts[0]
                title_raw = parts[1]
                genres = parts[2]
                # 提取年份
                m = re.search(r"\((\d{4})\)", title_raw)
                year = int(m.group(1)) if m else None
                title = re.sub(r"\s*\(\d{4}\)\s*$", "", title_raw).strip()
                movies_dat[mid] = {"title": title_raw, "title_clean": title, "year": year, "genres": genres}

    # 加载中文标题
    cn_path = "data/processed/cn_titles.json"
    if os.path.exists(cn_path):
        with open(cn_path, "r", encoding="utf-8") as f:
            CN_TITLES = json.load(f)

    # 加载 enrich_cache (overview, directors, cast)
    with open("data/processed/enrich_cache.json", "r", encoding="utf-8") as f:
        enrich = json.load(f)

    # 加载 llm_semantic_tags
    with open("data/processed/llm_semantic_tags.json", "r", encoding="utf-8") as f:
        tags = json.load(f)

    # 加载 links.csv -> tmdb_id
    tmdb_map = {}
    links_path = "data/raw/ml-latest-small/links.csv"
    if os.path.exists(links_path):
        with open(links_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                tmdb_map[row["movieId"]] = row.get("tmdbId", "")

    # 合并所有信息
    for mid in SEMANTIC_IDS:
        meta = {"id": mid}
        if mid in movies_dat:
            meta.update(movies_dat[mid])
        if mid in enrich:
            meta["overview"] = enrich[mid].get("overview", "")
            meta["directors"] = enrich[mid].get("directors", [])
            meta["cast"] = enrich[mid].get("cast", [])
        if mid in tags:
            meta["tags"] = tags[mid]
        if mid in tmdb_map and tmdb_map[mid]:
            meta["tmdb_id"] = tmdb_map[mid]
        if mid in CN_TITLES:
            meta["cn_title"] = CN_TITLES[mid]
        MOVIES_META[mid] = meta

    print(f"✓ 模型加载完成 | 电影数: {len(MOVIES_META)} | SID 词表: {CHECKPOINT['vocab_size']}")


# ── API 路由 ────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory("web", "index.html")


@app.route("/api/movies")
def api_movies():
    """返回所有电影列表（用于搜索选择）"""
    q = request.args.get("q", "").lower().strip()
    results = []
    for mid, meta in MOVIES_META.items():
        title = meta.get("title", "")
        cn_title = meta.get("cn_title", "")
        if q and q not in title.lower() and q not in cn_title:
            continue
        results.append({
            "id": mid,
            "title": title,
            "cn_title": cn_title,
            "genres": meta.get("genres", ""),
            "year": meta.get("year"),
        })
        if len(results) >= 50:
            break
    return jsonify(results)


@app.route("/api/movie/<movie_id>")
def api_movie_detail(movie_id):
    """返回单部电影的详细信息"""
    meta = MOVIES_META.get(movie_id)
    if not meta:
        return jsonify({"error": "未找到该电影"}), 404
    return jsonify(meta)


@app.route("/api/recommend", methods=["POST"])
def api_recommend():
    """根据用户选择的电影序列生成推荐"""
    data = request.get_json()
    if not data or "movie_ids" not in data:
        return jsonify({"error": "请提供 movie_ids 列表"}), 400

    movie_ids = [str(mid) for mid in data["movie_ids"]]
    top_k = min(int(data.get("top_k", 10)), 50)

    # 过滤无 SID 的电影
    valid_ids = [mid for mid in movie_ids if mid in SEMANTIC_IDS]
    if len(valid_ids) < 1:
        return jsonify({"error": "所选电影均无语义ID，请至少选一部有效电影"}), 400

    # 只取最近 9 部
    valid_ids = valid_ids[-9:]

    sid_length = CHECKPOINT["sid_length"]
    bos_token = MODEL.bos_token
    sep_token = MODEL.sep_token

    # 构建 token 序列
    tokens = [bos_token]
    movie_pos = [0]
    sid_pos = [0]

    for m_idx, mid in enumerate(valid_ids):
        sid = SEMANTIC_IDS[mid]
        for s_idx, token in enumerate(sid):
            tokens.append(token)
            movie_pos.append(m_idx + 1)
            sid_pos.append(s_idx)
        tokens.append(sep_token)
        movie_pos.append(m_idx + 1)
        sid_pos.append(sid_length)

    input_ids = torch.tensor([tokens])
    movie_pos_t = torch.tensor([movie_pos])
    sid_pos_t = torch.tensor([sid_pos])

    # 生成 SID
    gen_sid = generate_sid(MODEL, input_ids, movie_pos_t, sid_pos_t, sid_length, "cpu")

    # 匹配
    top_movies, scores = match_sid_to_movies(gen_sid, SID_MATRIX, MOVIE_IDS, top_k=top_k)

    # 排除用户已看的电影
    selected_set = set(valid_ids)
    recommendations = []
    for mid, score in zip(top_movies, scores):
        if mid in selected_set:
            continue
        meta = MOVIES_META.get(mid, {})
        recommendations.append({
            "id": mid,
            "title": meta.get("title", f"Movie {mid}"),
            "cn_title": meta.get("cn_title", ""),
            "genres": meta.get("genres", ""),
            "year": meta.get("year"),
            "overview": meta.get("overview", ""),
            "directors": meta.get("directors", []),
            "cast": meta.get("cast", []),
            "tags": meta.get("tags", {}),
            "tmdb_id": meta.get("tmdb_id", ""),
            "score": float(score),
        })
    if len(recommendations) < top_k:
        # 补充更多推荐
        extra_movies, extra_scores = match_sid_to_movies(gen_sid, SID_MATRIX, MOVIE_IDS, top_k=top_k + len(selected_set))
        for mid, score in zip(extra_movies, extra_scores):
            if mid in selected_set or any(r["id"] == mid for r in recommendations):
                continue
            meta = MOVIES_META.get(mid, {})
            recommendations.append({
                "id": mid,
                "title": meta.get("title", f"Movie {mid}"),
                "cn_title": meta.get("cn_title", ""),
                "genres": meta.get("genres", ""),
                "year": meta.get("year"),
                "overview": meta.get("overview", ""),
                "directors": meta.get("directors", []),
                "cast": meta.get("cast", []),
                "tags": meta.get("tags", {}),
                "tmdb_id": meta.get("tmdb_id", ""),
                "score": float(score),
            })
            if len(recommendations) >= top_k:
                break

    # 解码生成的 SID token 为可读标签
    with open("data/processed/sid_vocab.json", "r", encoding="utf-8") as f:
        vocab_info = json.load(f)
    inv_vocab = {v: k for k, v in vocab_info["explicit_vocab"].items()}
    decoded_sid = []
    labels = ["genre1", "genre2", "genre3", "mood", "pace", "audience", "narrative", "visual", "era", "cluster1", "cluster2", "cluster3"]
    for i, tok in enumerate(gen_sid):
        label = labels[i] if i < len(labels) else f"tok{i}"
        name = inv_vocab.get(tok, f"implicit_{tok}")
        decoded_sid.append({"position": label, "token": tok, "name": name})

    return jsonify({
        "recommendations": recommendations,
        "generated_sid": decoded_sid,
        "input_movies": len(valid_ids),
    })


if __name__ == "__main__":
    init()
    app.run(host="127.0.0.1", port=5000, debug=False)
