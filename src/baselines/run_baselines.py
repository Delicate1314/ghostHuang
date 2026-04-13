"""基线方法评估：Popularity, BPR-MF, SASRec
与 SID 模型使用完全相同的数据划分和评估协议
"""
import json
import math
import os
from collections import Counter, defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ────────────────────────────────────────────────────────────
# 共用工具
# ────────────────────────────────────────────────────────────

def load_data(
    train_path="data/splits/train.json",
    test_path="data/splits/test.json",
    cold_path="data/splits/cold_movies.json",
):
    with open(train_path) as f:
        train_data = json.load(f)
    with open(test_path) as f:
        test_data = json.load(f)
    with open(cold_path) as f:
        cold_movies = set(str(m) for m in json.load(f))

    # 收集所有电影
    all_movies = set()
    for item in train_data:
        all_movies.update(str(m) for m in item["movie_seq"])
    for item in test_data:
        all_movies.update(str(m) for m in item["movie_seq"])
        all_movies.add(str(item["target"]))

    # 构建 movie -> index 映射
    movie_list = sorted(all_movies)
    movie2idx = {m: i for i, m in enumerate(movie_list)}
    n_movies = len(movie_list)

    print(f"训练用户: {len(train_data)}, 测试用户: {len(test_data)}")
    print(f"电影总数: {n_movies}, 冷启动电影: {len(cold_movies)}")

    return train_data, test_data, cold_movies, movie_list, movie2idx, n_movies


def compute_metrics(results_list, cold_movies, top_k=10):
    """统一指标计算"""
    warm = {"hits": 0, "ndcg": 0, "mrr": 0, "total": 0}
    cold = {"hits": 0, "ndcg": 0, "mrr": 0, "total": 0}

    for target, ranked_list in results_list:
        is_cold = target in cold_movies
        m = cold if is_cold else warm
        m["total"] += 1

        top = ranked_list[:top_k]
        if target in top:
            rank = top.index(target) + 1
            m["hits"] += 1
            m["ndcg"] += 1.0 / math.log2(rank + 1)
            m["mrr"] += 1.0 / rank

    out = {}
    for name, m in [("Warm", warm), ("Cold-Start", cold)]:
        t = m["total"]
        if t == 0:
            out[name] = {"Recall@K": 0, "NDCG@K": 0, "MRR": 0, "total_samples": 0}
            continue
        out[name] = {
            "Recall@K": m["hits"] / t,
            "NDCG@K": m["ndcg"] / t,
            "MRR": m["mrr"] / t,
            "total_samples": t,
        }
    # Overall
    tw = warm["total"] + cold["total"]
    if tw > 0:
        out["Overall"] = {
            "Recall@K": (warm["hits"] + cold["hits"]) / tw,
            "NDCG@K": (warm["ndcg"] + cold["ndcg"]) / tw,
            "MRR": (warm["mrr"] + cold["mrr"]) / tw,
            "total_samples": tw,
        }
    return out


def print_results(name, results):
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    for group in ["Overall", "Warm", "Cold-Start"]:
        if group not in results:
            continue
        r = results[group]
        print(f"  {group} ({r['total_samples']} samples):")
        print(f"    Recall@10: {r['Recall@K']:.4f}")
        print(f"    NDCG@10:   {r['NDCG@K']:.4f}")
        print(f"    MRR:       {r['MRR']:.4f}")


# ────────────────────────────────────────────────────────────
# Baseline 1: Popularity
# ────────────────────────────────────────────────────────────

def run_popularity(train_data, test_data, cold_movies, movie_list, top_k=10):
    """热门推荐：按训练集中出现频次排序"""
    print("\n>>> 运行 Popularity 基线...")

    # 统计训练集中每部电影的出现次数
    counter = Counter()
    for item in train_data:
        for m in item["movie_seq"]:
            counter[str(m)] += 1

    # 按频次排序得到全局 Top-K
    pop_ranking = [m for m, _ in counter.most_common()]
    # 冷启动电影在训练集中不存在，所以永远不在 pop_ranking 中

    results_list = []
    for item in test_data:
        target = str(item["target"])
        # 排除用户已看过的
        seen = set(str(m) for m in item["movie_seq"])
        ranked = [m for m in pop_ranking if m not in seen][:top_k]
        results_list.append((target, ranked))

    return compute_metrics(results_list, cold_movies, top_k)


# ────────────────────────────────────────────────────────────
# Baseline 2: BPR-MF (Matrix Factorization)
# ────────────────────────────────────────────────────────────

class BPRModel(nn.Module):
    def __init__(self, n_users, n_items, dim=64):
        super().__init__()
        self.user_emb = nn.Embedding(n_users, dim)
        self.item_emb = nn.Embedding(n_items, dim)
        nn.init.xavier_uniform_(self.user_emb.weight)
        nn.init.xavier_uniform_(self.item_emb.weight)

    def forward(self, user, pos_item, neg_item):
        u = self.user_emb(user)
        pi = self.item_emb(pos_item)
        ni = self.item_emb(neg_item)
        pos_score = (u * pi).sum(dim=1)
        neg_score = (u * ni).sum(dim=1)
        loss = -F.logsigmoid(pos_score - neg_score).mean()
        return loss

    def predict(self, user):
        u = self.user_emb(user)
        scores = u @ self.item_emb.weight.T
        return scores


def run_bpr(train_data, test_data, cold_movies, movie_list, movie2idx, n_movies,
            dim=64, epochs=30, lr=0.01, batch_size=512, top_k=10):
    """BPR-MF 矩阵分解基线"""
    print("\n>>> 运行 BPR-MF 基线...")

    # 构建 user -> idx
    user_list = sorted(set(item["userId"] for item in train_data))
    user2idx = {u: i for i, u in enumerate(user_list)}
    n_users = len(user_list)

    # 构建交互对 (user_idx, item_idx)
    interactions = []
    user_items = defaultdict(set)
    for item in train_data:
        uid = user2idx[item["userId"]]
        for m in item["movie_seq"]:
            mid = movie2idx[str(m)]
            interactions.append((uid, mid))
            user_items[uid].add(mid)

    interactions = np.array(interactions)
    print(f"  交互对: {len(interactions)}, 用户: {n_users}, 电影: {n_movies}")

    device = "cpu"
    model = BPRModel(n_users, n_movies, dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        np.random.shuffle(interactions)
        total_loss = 0
        n_batch = 0

        for start in range(0, len(interactions), batch_size):
            batch = interactions[start:start + batch_size]
            users = torch.LongTensor(batch[:, 0]).to(device)
            pos_items = torch.LongTensor(batch[:, 1]).to(device)
            # 向量化负采样
            neg = np.random.randint(0, n_movies, size=len(batch))
            neg_items = torch.LongTensor(neg).to(device)

            loss = model(users, pos_items, neg_items)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batch += 1

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1}/{epochs}, Loss: {total_loss/n_batch:.4f}")

    # 评估
    model.eval()
    results_list = []
    for item in test_data:
        target = str(item["target"])
        uid = item["userId"]
        if uid not in user2idx:
            continue
        uidx = user2idx[uid]

        with torch.no_grad():
            scores = model.predict(torch.LongTensor([uidx]).to(device))[0]

        # 屏蔽已看过的
        seen = set(str(m) for m in item["movie_seq"])
        for m in seen:
            if m in movie2idx:
                scores[movie2idx[m]] = -1e9

        top_indices = torch.topk(scores, top_k).indices.cpu().numpy()
        ranked = [movie_list[i] for i in top_indices]
        results_list.append((target, ranked))

    return compute_metrics(results_list, cold_movies, top_k)


# ────────────────────────────────────────────────────────────
# Baseline 3: SASRec (Self-Attentive Sequential Recommendation)
# ────────────────────────────────────────────────────────────

class SASRecModel(nn.Module):
    def __init__(self, n_items, max_len=50, d_model=64, nhead=2, n_layers=2, dropout=0.1):
        super().__init__()
        self.item_emb = nn.Embedding(n_items + 1, d_model, padding_idx=0)  # 0=pad
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.dropout = nn.Dropout(dropout)
        self.max_len = max_len

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=d_model * 4, dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.output = nn.Linear(d_model, n_items + 1)

    def forward(self, seq):
        """seq: (B, L) item indices, 0-padded on left"""
        B, L = seq.shape
        positions = torch.arange(L, device=seq.device).unsqueeze(0).expand(B, L)
        x = self.item_emb(seq) + self.pos_emb(positions)
        x = self.dropout(x)

        # causal mask
        mask = nn.Transformer.generate_square_subsequent_mask(L).to(x.device)
        # padding mask
        pad_mask = (seq == 0)

        x = self.encoder(x, mask=mask, src_key_padding_mask=pad_mask)
        logits = self.output(x)
        return logits


def run_sasrec(train_data, test_data, cold_movies, movie_list, movie2idx, n_movies,
               max_len=50, d_model=64, epochs=30, lr=1e-3, batch_size=128, top_k=10):
    """SASRec 序列推荐基线"""
    print("\n>>> 运行 SASRec 基线...")

    # 构建训练序列 (item index 从 1 开始，0 是 padding)
    idx_offset = 1  # 留 0 给 padding
    train_seqs = []
    for item in train_data:
        seq = [movie2idx[str(m)] + idx_offset for m in item["movie_seq"] if str(m) in movie2idx]
        if len(seq) >= 3:
            train_seqs.append(seq[-max_len:])

    print(f"  有效训练序列: {len(train_seqs)}")

    device = "cpu"
    model = SASRecModel(n_movies, max_len=max_len, d_model=d_model).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  SASRec 参数量: {total_params:,}")

    # 训练：预测序列中下一个 item
    for epoch in range(epochs):
        model.train()
        np.random.shuffle(train_seqs)
        total_loss = 0
        n_batch = 0

        for start in range(0, len(train_seqs), batch_size):
            batch_seqs = train_seqs[start:start + batch_size]
            # 左 padding 到 max_len
            padded = []
            targets = []
            for seq in batch_seqs:
                inp = seq[:-1]
                tgt = seq[1:]
                pad_len = max_len - len(inp)
                padded.append([0] * pad_len + inp)
                targets.append([-100] * pad_len + tgt)

            inp_t = torch.LongTensor(padded).to(device)
            tgt_t = torch.LongTensor(targets).to(device)

            logits = model(inp_t)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), tgt_t.view(-1), ignore_index=-100)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            n_batch += 1

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1}/{epochs}, Loss: {total_loss/n_batch:.4f}")

    # 评估
    model.eval()
    results_list = []
    for item in test_data:
        target = str(item["target"])
        seq = [movie2idx[str(m)] + idx_offset for m in item["movie_seq"] if str(m) in movie2idx]
        if len(seq) < 2:
            continue
        seq = seq[-max_len:]
        pad_len = max_len - len(seq)
        inp = torch.LongTensor([[0] * pad_len + seq]).to(device)

        with torch.no_grad():
            logits = model(inp)
            scores = logits[0, -1, :]  # 最后一个位置的预测

        # 屏蔽已看过的 和 padding
        seen = set(str(m) for m in item["movie_seq"])
        scores[0] = -1e9  # padding
        for m in seen:
            if m in movie2idx:
                scores[movie2idx[m] + idx_offset] = -1e9

        top_indices = torch.topk(scores, top_k).indices.cpu().numpy()
        ranked = [movie_list[i - idx_offset] if 0 <= i - idx_offset < len(movie_list) else "" for i in top_indices]
        results_list.append((target, ranked))

    return compute_metrics(results_list, cold_movies, top_k)


# ────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────

def main():
    train_data, test_data, cold_movies, movie_list, movie2idx, n_movies = load_data()

    all_results = {}

    # Popularity
    pop_res = run_popularity(train_data, test_data, cold_movies, movie_list)
    print_results("Popularity", pop_res)
    all_results["Popularity"] = pop_res

    # BPR-MF
    bpr_res = run_bpr(train_data, test_data, cold_movies, movie_list, movie2idx, n_movies)
    print_results("BPR-MF", bpr_res)
    all_results["BPR-MF"] = bpr_res

    # SASRec
    sasrec_res = run_sasrec(train_data, test_data, cold_movies, movie_list, movie2idx, n_movies)
    print_results("SASRec", sasrec_res)
    all_results["SASRec"] = sasrec_res

    # 保存
    os.makedirs("results", exist_ok=True)
    with open("results/baseline_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n所有基线结果已保存至 results/baseline_results.json")


if __name__ == "__main__":
    main()
