# -*- coding: utf-8 -*-
from __future__ import print_function
import argparse
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from model.pointnet import PointNetSkipAtCls

parser = argparse.ArgumentParser(description="pointnetGPD")
parser.add_argument("--cuda", action="store_true", default=True)
parser.add_argument("--gpu", type=int, default=0)
parser.add_argument("--load-model", type=str, default="./assets/learned_models/PointNet_SkipAt/PointNet_SkipAt_200.model")
parser.add_argument("--model-points", type=int, default=750)
parser.add_argument("--n-repeat", type=int, default=10)
args = parser.parse_args()

# GPU設定
args.cuda = args.cuda if torch.cuda.is_available() else False
device = torch.device(f"cuda:{args.gpu}" if args.cuda else "cpu")
if args.cuda:
    torch.cuda.manual_seed(1)
np.random.seed(int(time.time()))

# モデル読み込み
checkpoint = torch.load(args.load_model, map_location="cpu", weights_only=False)
model = checkpoint.module if hasattr(checkpoint, "module") else checkpoint
model.eval()
model.to(device)
print(f"✅ モデルをロードしました: {args.load_model}")

# ── ここから生の test.npy 前処理付き推論 ──
# 1. 点群をロード
pc_raw = np.load("test.npy")  # shape=(M,3)
M = pc_raw.shape[0]
print(f"Loaded raw point cloud with {M} points")


# 2. Dataset相当の前処理関数を定義
def preprocess_point_cloud(pc, num_points):
    # ① ランダムサンプリング or リプレイスあり抽出
    if pc.shape[0] >= num_points:
        idxs = np.random.choice(pc.shape[0], num_points, replace=False)
    else:
        idxs = np.random.choice(pc.shape[0], num_points, replace=True)
    pc_sampled = pc[idxs, :]  # (num_points, 3)

    # ② 重心を引いてセンタリング
    centroid = np.mean(pc_sampled, axis=0)
    pc_centered = pc_sampled - centroid  # (num_points, 3)

    # ③ 最大距離でスケーリング（半径を1に正規化）
    furthest_dist = np.max(np.linalg.norm(pc_centered, axis=1))
    if furthest_dist > 0:
        pc_normalized = pc_centered / furthest_dist
    else:
        pc_normalized = pc_centered

    # ④ 転置して (3, num_points) に
    return pc_normalized.T.astype(np.float32)


# 3. 複数回繰り返して投げる場合の処理ループ
all_logits = []
all_probs = []
all_preds = []

for _ in range(args.n_repeat):
    # 前処理してテンソル化
    pc_proc = preprocess_point_cloud(pc_raw, args.model_points)  # (3,750)
    pc_tensor = torch.from_numpy(pc_proc).unsqueeze(0).to(device)  # (1,3,750)

    # 推論
    with torch.no_grad():
        logits, _ = model(pc_tensor)  # (1,2)
        probs = F.softmax(logits, dim=-1)  # (1,2)

    # CPUへ戻して numpy に
    logits_np = logits.cpu().numpy()[0]  # shape=(2,)
    probs_np = probs.cpu().numpy()[0]  # shape=(2,)
    pred = int(probs_np.argmax())  # 予測ラベル

    all_logits.append(logits_np)
    all_probs.append(probs_np)
    all_preds.append(pred)

# 4. 結果をまとめて表示
all_logits = np.stack(all_logits, axis=0)  # (n_repeat,2)
all_probs = np.stack(all_probs, axis=0)  # (n_repeat,2)
all_preds = np.array(all_preds)  # (n_repeat,)

# 平均logits, 平均確率, 最頻値（voting）
mean_logits = all_logits.mean(axis=0)
mean_probs = all_probs.mean(axis=0)
from scipy.stats import mode

vote_label = int(mode(all_preds).mode.item())

print("🔍 raw logits（repeat平均）:", mean_logits)
print("✨ softmax probabilities（repeat平均）:", mean_probs)
print("🏷️ voting予測ラベル:", vote_label)
