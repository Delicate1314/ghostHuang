"""生成式推荐模型：基于 Transformer Decoder 的序列预测
输入：用户历史电影的语义 ID 序列
输出：逐 token 预测下一部电影的语义 ID
"""
import json
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SIDGenerativeModel(nn.Module):
    """Transformer Decoder 模型，自回归生成语义 ID"""

    def __init__(
        self,
        vocab_size: int,
        sid_length: int,
        d_model: int = 256,
        nhead: int = 4,
        num_layers: int = 4,
        max_seq_len: int = 50,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.sid_length = sid_length
        self.d_model = d_model
        self.vocab_size = vocab_size

        # token embedding + position embedding
        self.token_emb = nn.Embedding(vocab_size + 2, d_model, padding_idx=0)
        # +2 for <BOS> and <SEP> tokens
        self.bos_token = vocab_size
        self.sep_token = vocab_size + 1

        # 位置编码：序列中每个电影的位置 + SID 内部的 token 位置
        self.movie_pos_emb = nn.Embedding(max_seq_len, d_model)
        self.sid_pos_emb = nn.Embedding(sid_length + 1, d_model)  # +1 for SEP

        # Transformer Decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # 输出头
        self.output_head = nn.Linear(d_model, vocab_size + 2)

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, input_ids, movie_positions, sid_positions):
        """
        input_ids: (B, L) - 展平的 SID token 序列
        movie_positions: (B, L) - 每个 token 对应的电影位置索引
        sid_positions: (B, L) - 每个 token 在 SID 内部的位置
        """
        B, L = input_ids.shape

        # Embedding
        x = self.token_emb(input_ids)
        x = x + self.movie_pos_emb(movie_positions)
        x = x + self.sid_pos_emb(sid_positions)

        # Causal mask
        causal_mask = nn.Transformer.generate_square_subsequent_mask(L).to(x.device)

        # Decoder (self-attention only, no encoder)
        memory = torch.zeros(B, 1, self.d_model, device=x.device)
        output = self.decoder(x, memory, tgt_mask=causal_mask)

        logits = self.output_head(output)
        return logits


class SIDDataset(torch.utils.data.Dataset):
    """将用户行为序列转换为 SID token 序列"""

    def __init__(
        self,
        sequences_path: str,
        semantic_ids: dict,
        sid_length: int,
        bos_token: int,
        sep_token: int,
        max_movies: int = 20,
    ):
        with open(sequences_path, "r") as f:
            data = json.load(f)

        self.samples = []
        self.sid_length = sid_length
        self.bos_token = bos_token
        self.sep_token = sep_token

        for item in data:
            seq = item["movie_seq"]
            # 过滤无 SID 的电影
            valid_seq = [str(m) for m in seq if str(m) in semantic_ids]
            if len(valid_seq) < 3:
                continue

            # 截取最近 max_movies 部
            valid_seq = valid_seq[-max_movies:]
            self.samples.append(valid_seq)

        print(f"有效训练序列：{len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate_fn(batch, semantic_ids, sid_length, bos_token, sep_token):
    """将电影序列展平为 SID token 序列"""
    all_input_ids = []
    all_target_ids = []
    all_movie_pos = []
    all_sid_pos = []

    for seq in batch:
        tokens = [bos_token]
        movie_pos = [0]
        sid_pos = [0]
        targets = []

        for m_idx, mid in enumerate(seq):
            sid = semantic_ids[mid]
            for s_idx, token in enumerate(sid):
                tokens.append(token)
                movie_pos.append(m_idx + 1)
                sid_pos.append(s_idx)
            # SEP token 分隔不同电影
            tokens.append(sep_token)
            movie_pos.append(m_idx + 1)
            sid_pos.append(sid_length)

        # target: 预测下一个 token（移位）
        input_seq = tokens[:-1]
        target_seq = tokens[1:]
        movie_pos_seq = movie_pos[:-1]
        sid_pos_seq = sid_pos[:-1]

        all_input_ids.append(input_seq)
        all_target_ids.append(target_seq)
        all_movie_pos.append(movie_pos_seq)
        all_sid_pos.append(sid_pos_seq)

    # Pad to same length
    max_len = max(len(s) for s in all_input_ids)
    padded_input = torch.zeros(len(batch), max_len, dtype=torch.long)
    padded_target = torch.full((len(batch), max_len), -100, dtype=torch.long)
    padded_movie_pos = torch.zeros(len(batch), max_len, dtype=torch.long)
    padded_sid_pos = torch.zeros(len(batch), max_len, dtype=torch.long)

    for i in range(len(batch)):
        L = len(all_input_ids[i])
        padded_input[i, :L] = torch.tensor(all_input_ids[i])
        padded_target[i, :L] = torch.tensor(all_target_ids[i])
        padded_movie_pos[i, :L] = torch.tensor(all_movie_pos[i])
        padded_sid_pos[i, :L] = torch.tensor(all_sid_pos[i])

    return padded_input, padded_target, padded_movie_pos, padded_sid_pos


def train_model(
    sequences_path: str = "data/splits/train.json",
    sid_path: str = "data/processed/semantic_ids.json",
    vocab_path: str = "data/processed/sid_vocab.json",
    model_path: str = "checkpoints/sid_model.pt",
    epochs: int = 20,
    batch_size: int = 64,
    lr: float = 1e-3,
    max_movies: int = 10,
):
    # 加载语义 ID
    with open(sid_path, "r") as f:
        semantic_ids = json.load(f)
    with open(vocab_path, "r") as f:
        vocab_info = json.load(f)

    vocab_size = vocab_info["total_vocab_size"]
    sid_length = vocab_info["sid_length"]

    print(f"Vocab: {vocab_size}, SID 长度: {sid_length}")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = SIDGenerativeModel(
        vocab_size=vocab_size,
        sid_length=sid_length,
        d_model=128,
        nhead=4,
        num_layers=2,
        max_seq_len=max_movies + 2,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量：{total_params:,}")

    bos_token = model.bos_token
    sep_token = model.sep_token

    dataset = SIDDataset(
        sequences_path, semantic_ids, sid_length, bos_token, sep_token, max_movies
    )

    from functools import partial

    collate = partial(
        collate_fn,
        semantic_ids=semantic_ids,
        sid_length=sid_length,
        bos_token=bos_token,
        sep_token=sep_token,
    )

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, collate_fn=collate
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    best_loss = float("inf")
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        n_batches = 0

        for input_ids, target_ids, movie_pos, sid_pos in dataloader:
            input_ids = input_ids.to(device)
            target_ids = target_ids.to(device)
            movie_pos = movie_pos.to(device)
            sid_pos = sid_pos.to(device)

            logits = model(input_ids, movie_pos, sid_pos)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                target_ids.view(-1),
                ignore_index=-100,
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = total_loss / max(n_batches, 1)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "vocab_size": vocab_size,
                    "sid_length": sid_length,
                    "d_model": 128,
                    "nhead": 4,
                    "num_layers": 2,
                    "best_loss": best_loss,
                },
                model_path,
            )

    print(f"\n训练完成！最佳 Loss: {best_loss:.4f}")
    print(f"模型保存至 {model_path}")


import os

if __name__ == "__main__":
    train_model()
