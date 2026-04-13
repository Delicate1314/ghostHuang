"""对比学习对齐：将文本+视觉多模态表示与协同过滤信号对齐"""
import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class AlignmentModel(nn.Module):
    def __init__(self, text_dim=1024, visual_dim=768, proj_dim=256):
        super().__init__()
        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, 512),
            nn.GELU(),
            nn.Linear(512, proj_dim),
        )
        self.visual_proj = nn.Sequential(
            nn.Linear(visual_dim, 512),
            nn.GELU(),
            nn.Linear(512, proj_dim),
        )
        self.temperature = nn.Parameter(torch.tensor(0.07))

    def forward(self, text_emb, visual_emb):
        text_z = F.normalize(self.text_proj(text_emb), dim=-1)
        visual_z = F.normalize(self.visual_proj(visual_emb), dim=-1)
        return text_z, visual_z

    def get_fused(self, text_emb, visual_emb, alpha=0.6):
        text_z, visual_z = self.forward(text_emb, visual_emb)
        return F.normalize(alpha * text_z + (1 - alpha) * visual_z, dim=-1)


def info_nce_loss(z_i, z_j, temperature=0.07):
    """对称 InfoNCE loss"""
    batch_size = z_i.shape[0]
    z = torch.cat([z_i, z_j], dim=0)  # (2B, D)
    sim = torch.mm(z, z.t()) / temperature  # (2B, 2B)

    # 正样本对: (i, i+B) 和 (i+B, i)
    labels = torch.cat(
        [torch.arange(batch_size, 2 * batch_size),
         torch.arange(0, batch_size)],
        dim=0,
    ).to(z.device)

    # 排除自身相似度
    mask = torch.eye(2 * batch_size, dtype=torch.bool, device=z.device)
    sim = sim.masked_fill(mask, -1e9)

    loss = F.cross_entropy(sim, labels)
    return loss


def build_cooccurrence_pairs(sequences_path: str, max_pairs: int = 500000):
    """从用户序列中构建共现电影对"""
    with open(sequences_path, "r") as f:
        train_data = json.load(f)

    pairs = []
    for item in train_data:
        seq = item["movie_seq"]
        for i in range(len(seq)):
            for j in range(i + 1, min(i + 5, len(seq))):
                pairs.append((seq[i], seq[j]))

    if len(pairs) > max_pairs:
        indices = np.random.choice(len(pairs), max_pairs, replace=False)
        pairs = [pairs[i] for i in indices]

    print(f"共现电影对：{len(pairs)}")
    return pairs


def train_alignment(
    text_emb_path: str = "data/processed/text_embeddings.npy",
    visual_emb_path: str = "data/processed/clip_embeddings.npy",
    text_ids_path: str = "data/processed/text_embeddings_ids.json",
    visual_ids_path: str = "data/processed/clip_embeddings_ids.json",
    sequences_path: str = "data/splits/train.json",
    output_path: str = "data/processed/fused_embeddings.npy",
    epochs: int = 50,
    batch_size: int = 256,
):
    text_emb = np.load(text_emb_path)
    visual_emb = np.load(visual_emb_path)

    with open(text_ids_path) as f:
        text_ids = json.load(f)
    with open(visual_ids_path) as f:
        visual_ids = json.load(f)

    # 对齐 ID：取交集
    common_ids = list(set(text_ids) & set(visual_ids))
    text_idx = {id_: i for i, id_ in enumerate(text_ids)}
    visual_idx = {id_: i for i, id_ in enumerate(visual_ids)}

    text_aligned = np.array([text_emb[text_idx[id_]] for id_ in common_ids])
    visual_aligned = np.array([visual_emb[visual_idx[id_]] for id_ in common_ids])

    print(f"对齐后电影数：{len(common_ids)}")
    print(f"文本维度：{text_aligned.shape}, 视觉维度：{visual_aligned.shape}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AlignmentModel(
        text_dim=text_aligned.shape[1],
        visual_dim=visual_aligned.shape[1],
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)

    text_tensor = torch.tensor(text_aligned, dtype=torch.float32)
    visual_tensor = torch.tensor(visual_aligned, dtype=torch.float32)

    # 构建共现 pair（使用索引）
    id_to_idx = {id_: i for i, id_ in enumerate(common_ids)}
    cooccurrence_pairs = build_cooccurrence_pairs(sequences_path)

    # 映射到对齐后的索引
    valid_pairs = []
    for mid_i, mid_j in cooccurrence_pairs:
        si, sj = str(mid_i), str(mid_j)
        if si in id_to_idx and sj in id_to_idx:
            valid_pairs.append((id_to_idx[si], id_to_idx[sj]))

    print(f"有效共现对：{len(valid_pairs)}")

    for epoch in range(epochs):
        np.random.shuffle(valid_pairs)
        total_loss = 0
        n_batches = 0

        for i in range(0, len(valid_pairs), batch_size):
            batch_pairs = valid_pairs[i : i + batch_size]
            if len(batch_pairs) < 2:
                continue

            idx_i = [p[0] for p in batch_pairs]
            idx_j = [p[1] for p in batch_pairs]

            fused_i = model.get_fused(
                text_tensor[idx_i].to(device),
                visual_tensor[idx_i].to(device),
            )
            fused_j = model.get_fused(
                text_tensor[idx_j].to(device),
                visual_tensor[idx_j].to(device),
            )

            loss = info_nce_loss(fused_i, fused_j, model.temperature)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        if (epoch + 1) % 10 == 0:
            avg = total_loss / max(n_batches, 1)
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg:.4f}")

    # 生成所有电影的对齐后表示
    with torch.no_grad():
        all_fused = (
            model.get_fused(text_tensor.to(device), visual_tensor.to(device))
            .cpu()
            .numpy()
        )

    np.save(output_path, all_fused)
    with open(output_path.replace(".npy", "_ids.json"), "w") as f:
        json.dump(common_ids, f)

    # 保存模型
    model_path = output_path.replace(".npy", "_model.pt")
    torch.save(model.state_dict(), model_path)

    print(f"融合 embedding 完成：{all_fused.shape} → {output_path}")


if __name__ == "__main__":
    train_alignment()
