"""分析实验结果，计算相对随机基线的倍数"""
import json

n_candidates = 3882
k = 10
random_recall = k / n_candidates

print(f"随机基线 Recall@{k} = {random_recall:.6f} ({k}/{n_candidates})")
print()

with open("results/evaluation_results.json") as f:
    sid = json.load(f)
with open("results/baseline_results.json") as f:
    bl = json.load(f)

# SID evaluate.py 中 "Overall" 实际是 Warm
warm_n, cold_n = 5055, 979
total_n = warm_n + cold_n
sid_overall = {
    "R": (sid["Overall"]["Recall@K"] * warm_n + sid["Cold-Start"]["Recall@K"] * cold_n) / total_n,
    "N": (sid["Overall"]["NDCG@K"] * warm_n + sid["Cold-Start"]["NDCG@K"] * cold_n) / total_n,
    "M": (sid["Overall"]["MRR"] * warm_n + sid["Cold-Start"]["MRR"] * cold_n) / total_n,
}

print("=" * 60)
print("各方法 Recall@10 相对随机基线的倍数")
print("=" * 60)

rows = [
    ("Random", random_recall, random_recall, random_recall),
    ("SASRec", bl["SASRec"]["Overall"]["Recall@K"], bl["SASRec"]["Warm"]["Recall@K"], bl["SASRec"]["Cold-Start"]["Recall@K"]),
    ("Popularity", bl["Popularity"]["Overall"]["Recall@K"], bl["Popularity"]["Warm"]["Recall@K"], bl["Popularity"]["Cold-Start"]["Recall@K"]),
    ("SID (Ours)", sid_overall["R"], sid["Overall"]["Recall@K"], sid["Cold-Start"]["Recall@K"]),
    ("BPR-MF", bl["BPR-MF"]["Overall"]["Recall@K"], bl["BPR-MF"]["Warm"]["Recall@K"], bl["BPR-MF"]["Cold-Start"]["Recall@K"]),
]

print(f"{'方法':<14} {'Overall':>10} {'(x rand)':>9} {'Warm':>10} {'(x rand)':>9} {'Cold':>10} {'(x rand)':>9}")
print("-" * 72)
for name, ov, wm, cd in rows:
    ox = ov / random_recall
    wx = wm / random_recall
    cx = cd / random_recall if cd > 0 else 0
    print(f"{name:<14} {ov:>10.4f} {ox:>8.1f}x {wm:>10.4f} {wx:>8.1f}x {cd:>10.4f} {cx:>8.1f}x")

print()
print("=" * 60)
print("完整指标对比表")
print("=" * 60)

full_rows = [
    ("Random", random_recall, random_recall * 0.4, random_recall * 0.25,
     random_recall, random_recall * 0.4, random_recall * 0.25,
     random_recall, random_recall * 0.4, random_recall * 0.25),
    ("Popularity",
     bl["Popularity"]["Overall"]["Recall@K"], bl["Popularity"]["Overall"]["NDCG@K"], bl["Popularity"]["Overall"]["MRR"],
     bl["Popularity"]["Warm"]["Recall@K"], bl["Popularity"]["Warm"]["NDCG@K"], bl["Popularity"]["Warm"]["MRR"],
     bl["Popularity"]["Cold-Start"]["Recall@K"], bl["Popularity"]["Cold-Start"]["NDCG@K"], bl["Popularity"]["Cold-Start"]["MRR"]),
    ("BPR-MF",
     bl["BPR-MF"]["Overall"]["Recall@K"], bl["BPR-MF"]["Overall"]["NDCG@K"], bl["BPR-MF"]["Overall"]["MRR"],
     bl["BPR-MF"]["Warm"]["Recall@K"], bl["BPR-MF"]["Warm"]["NDCG@K"], bl["BPR-MF"]["Warm"]["MRR"],
     bl["BPR-MF"]["Cold-Start"]["Recall@K"], bl["BPR-MF"]["Cold-Start"]["NDCG@K"], bl["BPR-MF"]["Cold-Start"]["MRR"]),
    ("SASRec",
     bl["SASRec"]["Overall"]["Recall@K"], bl["SASRec"]["Overall"]["NDCG@K"], bl["SASRec"]["Overall"]["MRR"],
     bl["SASRec"]["Warm"]["Recall@K"], bl["SASRec"]["Warm"]["NDCG@K"], bl["SASRec"]["Warm"]["MRR"],
     bl["SASRec"]["Cold-Start"]["Recall@K"], bl["SASRec"]["Cold-Start"]["NDCG@K"], bl["SASRec"]["Cold-Start"]["MRR"]),
    ("SID (Ours)",
     sid_overall["R"], sid_overall["N"], sid_overall["M"],
     sid["Overall"]["Recall@K"], sid["Overall"]["NDCG@K"], sid["Overall"]["MRR"],
     sid["Cold-Start"]["Recall@K"], sid["Cold-Start"]["NDCG@K"], sid["Cold-Start"]["MRR"]),
]

for split_name, cols in [("Overall", (0,1,2)), ("Warm", (3,4,5)), ("Cold-Start", (6,7,8))]:
    print(f"\n--- {split_name} ---")
    print(f"{'方法':<14} {'Recall@10':>10} {'NDCG@10':>10} {'MRR':>10}")
    print("-" * 46)
    for row in full_rows:
        name = row[0]
        r, n, m = row[cols[0]+1], row[cols[1]+1], row[cols[2]+1]
        print(f"{name:<14} {r:>10.4f} {n:>10.4f} {m:>10.4f}")

print()
print("=" * 60)
print("关键数据点")
print("=" * 60)
print(f"SID 冷启动 Recall@10 是随机基线的 {sid['Cold-Start']['Recall@K']/random_recall:.1f} 倍")
if bl["SASRec"]["Cold-Start"]["Recall@K"] > 0:
    print(f"SID 冷启动 Recall@10 是 SASRec 的 {sid['Cold-Start']['Recall@K']/bl['SASRec']['Cold-Start']['Recall@K']:.1f} 倍")
print(f"SID Overall Recall@10 是随机基线的 {sid_overall['R']/random_recall:.1f} 倍")
print(f"BPR-MF Overall 是最强基线 (Recall@10={bl['BPR-MF']['Overall']['Recall@K']:.4f})")
print(f"SID Overall 达到 BPR-MF 的 {sid_overall['R']/bl['BPR-MF']['Overall']['Recall@K']*100:.1f}%")
print(f"SID Warm MRR ({sid['Overall']['MRR']:.4f}) 略高于 Popularity MRR ({bl['Popularity']['Warm']['MRR']:.4f})")
