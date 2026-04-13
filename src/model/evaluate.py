"""评估脚本：推荐模型 + 冷启动实验"""
import json
import os
from collections import defaultdict
from functools import partial

import numpy as np
import torch
import torch.nn.functional as F

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.model.sid_model import SIDGenerativeModel, SIDDataset, collate_fn


def load_model(model_path: str, device: str = "cpu"):
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    model = SIDGenerativeModel(
        vocab_size=checkpoint["vocab_size"],
        sid_length=checkpoint["sid_length"],
        d_model=checkpoint["d_model"],
        nhead=checkpoint["nhead"],
        num_layers=checkpoint["num_layers"],
        max_seq_len=12,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


def build_sid_matrix(semantic_ids: dict):
    """构建电影 SID 矩阵用于匹配"""
    movie_ids = list(semantic_ids.keys())
    sid_matrix = np.array([semantic_ids[mid] for mid in movie_ids])
    return movie_ids, sid_matrix


def generate_sid(model, input_ids, movie_pos, sid_pos, sid_length, device):
    """自回归生成一个完整 SID（12 个 token）"""
    model.eval()
    with torch.no_grad():
        # 用整个历史序列的 logits，取最后一个位置开始生成
        generated_tokens = []
        current_input = input_ids.clone()
        current_movie_pos = movie_pos.clone()
        current_sid_pos = sid_pos.clone()

        # 最后一个电影的位置 + 1 作为预测电影的位置
        next_movie_pos = current_movie_pos[0, -1].item() + 1

        for sid_idx in range(sid_length + 1):  # +1 for SEP
            logits = model(
                current_input.to(device),
                current_movie_pos.to(device),
                current_sid_pos.to(device),
            )
            # 取最后一个位置的 logits
            next_logits = logits[0, -1, :]
            next_token = next_logits.argmax().item()
            generated_tokens.append(next_token)

            if sid_idx < sid_length:
                # 拼接到输入继续生成
                new_input = torch.tensor([[next_token]])
                new_mp = torch.tensor([[next_movie_pos]])
                new_sp = torch.tensor([[sid_idx]])

                current_input = torch.cat([current_input, new_input], dim=1)
                current_movie_pos = torch.cat([current_movie_pos, new_mp], dim=1)
                current_sid_pos = torch.cat([current_sid_pos, new_sp], dim=1)

    # 返回前 sid_length 个 token（不含 SEP）
    return generated_tokens[:sid_length]


def match_sid_to_movies(generated_sid, sid_matrix, movie_ids, top_k=20):
    """将生成的 SID 与所有电影的 SID 做匹配，返回 Top-K"""
    gen = np.array(generated_sid)

    # 逐 token 匹配计分（精确匹配得 1 分）
    scores = np.sum(sid_matrix == gen, axis=1).astype(float)

    # 前面的 token 权重更高（genre > mood > ... > implicit）
    weights = np.array([3, 3, 3, 2, 2, 2, 2, 2, 2, 1, 1, 1], dtype=float)
    weights = weights[:sid_matrix.shape[1]]
    weighted_scores = np.sum((sid_matrix == gen) * weights, axis=1)

    top_indices = np.argsort(-weighted_scores)[:top_k]
    return [movie_ids[i] for i in top_indices], weighted_scores[top_indices]


def evaluate(
    model_path: str = "checkpoints/sid_model.pt",
    test_path: str = "data/splits/test.json",
    sid_path: str = "data/processed/semantic_ids.json",
    vocab_path: str = "data/processed/sid_vocab.json",
    cold_path: str = "data/splits/cold_movies.json",
    top_k: int = 10,
):
    device = "cpu"

    # 加载模型
    model, checkpoint = load_model(model_path, device)
    sid_length = checkpoint["sid_length"]
    bos_token = model.bos_token
    sep_token = model.sep_token

    # 加载语义 ID
    with open(sid_path, "r") as f:
        semantic_ids = json.load(f)
    with open(vocab_path, "r") as f:
        vocab_info = json.load(f)

    # 加载测试数据
    with open(test_path, "r") as f:
        test_data = json.load(f)

    # 加载冷启动电影
    with open(cold_path, "r") as f:
        cold_movies = set(str(m) for m in json.load(f))

    # 构建 SID 匹配矩阵
    movie_ids, sid_matrix = build_sid_matrix(semantic_ids)
    movie_id_set = set(movie_ids)

    print(f"模型加载完成，Loss: {checkpoint['best_loss']:.4f}")
    print(f"测试用户数: {len(test_data)}")
    print(f"候选电影数: {len(movie_ids)}")
    print(f"冷启动电影数: {len(cold_movies)}")
    print(f"评估 Top-{top_k}")
    print("=" * 60)

    # 评估指标
    metrics = {"hits": 0, "ndcg": 0, "mrr": 0, "total": 0}
    cold_metrics = {"hits": 0, "ndcg": 0, "mrr": 0, "total": 0}

    evaluated = 0
    for i, item in enumerate(test_data):
        target = str(item["target"])
        seq = item["movie_seq"]

        # 跳过无 SID 的目标
        if target not in semantic_ids:
            continue

        # 构造输入序列
        valid_seq = [str(m) for m in seq if str(m) in semantic_ids]
        if len(valid_seq) < 2:
            continue

        # 只取最近 9 部（max_movies=10 中留 1 个给预测）
        valid_seq = valid_seq[-9:]

        # 构建 token 序列
        tokens = [bos_token]
        movie_pos = [0]
        sid_pos = [0]

        for m_idx, mid in enumerate(valid_seq):
            sid = semantic_ids[mid]
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
        gen_sid = generate_sid(model, input_ids, movie_pos_t, sid_pos_t, sid_length, device)

        # 匹配
        top_movies, scores = match_sid_to_movies(gen_sid, sid_matrix, movie_ids, top_k=top_k)

        # 计算指标
        is_cold = target in cold_movies
        target_metrics = cold_metrics if is_cold else metrics

        target_metrics["total"] += 1

        if target in top_movies:
            rank = top_movies.index(target) + 1
            target_metrics["hits"] += 1
            target_metrics["ndcg"] += 1.0 / np.log2(rank + 1)
            target_metrics["mrr"] += 1.0 / rank

        evaluated += 1
        if evaluated % 500 == 0:
            print(f"  已评估 {evaluated} 个用户...")

    # 计算最终指标
    print("\n" + "=" * 60)
    print(f"总体评估结果 (Top-{top_k})")
    print("=" * 60)

    results = {}
    for name, m in [("Overall", metrics), ("Cold-Start", cold_metrics)]:
        total = m["total"]
        if total == 0:
            print(f"\n{name}: 无有效样本")
            continue

        recall = m["hits"] / total
        ndcg = m["ndcg"] / total
        mrr = m["mrr"] / total
        hr = m["hits"] / total

        results[name] = {
            "Recall@K": recall,
            "NDCG@K": ndcg,
            "MRR": mrr,
            "HitRate@K": hr,
            "total_samples": total,
        }

        print(f"\n{name} ({total} samples):")
        print(f"  Recall@{top_k}:  {recall:.4f}")
        print(f"  NDCG@{top_k}:    {ndcg:.4f}")
        print(f"  MRR:           {mrr:.4f}")
        print(f"  HitRate@{top_k}: {hr:.4f}")

    # 保存结果
    os.makedirs("results", exist_ok=True)
    with open("results/evaluation_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n结果已保存至 results/evaluation_results.json")

    return results


if __name__ == "__main__":
    evaluate()
