# -*- coding: utf-8 -*-
from __future__ import print_function
import argparse
import os
import time

import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR

from model.dataset import PointGraspOneViewDataset
from model.pointnet import PointNetSkipAtCls
from tqdm import tqdm

parser = argparse.ArgumentParser(description="pointnetGPD")
parser.add_argument("--tag", type=str, default="PointNet_SkipAt")
parser.add_argument("--epoch", type=int, default=201)
parser.add_argument("--mode", choices=["train", "test"], required=True)
parser.add_argument("--batch-size", type=int, default=64)
parser.add_argument("--cuda", action="store_true")
parser.add_argument("--gpu", type=int, default=0)
parser.add_argument("--lr", type=float, default=0.001)
parser.add_argument("--load-model", type=str, default="")
parser.add_argument("--load-epoch", type=int, default=-1)
parser.add_argument("--model-path", type=str, default="./assets/learned_models", help="pre-trained model path")
parser.add_argument("--log-interval", type=int, default=10)
parser.add_argument("--save-interval", type=int, default=10)

args = parser.parse_args()

# GPU設定
args.cuda = torch.cuda.is_available()
os.makedirs(args.model_path, exist_ok=True)
if args.cuda:
    torch.cuda.manual_seed(1)

# ロガー
logger = SummaryWriter(os.path.join("./assets/log/", args.tag))
np.random.seed(int(time.time()))


def worker_init_fn(pid):
    np.random.seed(torch.initial_seed() % (2**31 - 1))


def my_collate(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)


# ハイパーパラメータ
grasp_points_num = 750
thresh_good = 0.6
thresh_bad = 0.6
point_channel = 3


def test(model, loader):
    model.eval()
    torch.set_grad_enabled(False)
    test_loss = 0
    correct = 0
    dataset_size = 0
    res = []
    for data, target, obj_name in loader:
        dataset_size += data.shape[0]
        data, target = data.float(), target.long().squeeze()
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        output, _ = model(data)  # N*C
        test_loss += F.nll_loss(output, target, reduction="sum").cpu().item()
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.view_as(pred)).long().cpu().sum()
        for i, j, k in zip(obj_name, pred.data.cpu().numpy(), target.data.cpu().numpy()):
            res.append((i, int(j[0]), int(k)))
    test_loss /= len(loader.dataset)
    acc = float(correct) / float(dataset_size)
    return acc, test_loss


def main():
    test_loader = torch.utils.data.DataLoader(
        PointGraspOneViewDataset(
            grasp_points_num=grasp_points_num,
            tag="test",
            grasp_amount_per_file=500,
            thresh_good=thresh_good,
            thresh_bad=thresh_bad,
            with_obj=True,
        ),
        batch_size=1,  # 1サンプルずつ見たいから1に設定
        num_workers=0,
        pin_memory=True,
        shuffle=False,
        worker_init_fn=worker_init_fn,
        collate_fn=my_collate,
    )

    # チェックポイント保存形式からロード
    model = torch.load(args.load_model, map_location="cpu", weights_only=False)

    # .eval() と GPU 設定
    if args.cuda:
        device_id = [args.gpu]
        torch.cuda.set_device(device_id[0])
        model = nn.DataParallel(model, device_ids=device_id).cuda()
        model = model.module  # DataParallel の wrapper を外す
    model.eval()

    print("✅ load model {}".format(args.load_model))

    # ——— ここから Dataset から読み込んで推論してみる！ ———
    print("💎 Datasetから1サンプル読み込んで推論してみるよ〜")
    data, target, obj_name = next(iter(test_loader))
    # data の shape をモデル入力に合わせる
    # Datasetクラスが (batch, N, 3) を返す想定 → (batch, 3, N) に変換
    if data.dim() == 3 and data.shape[1] == grasp_points_num and data.shape[2] == 3:
        data = data.permute(0, 2, 1)  # → (1, 3, 750)
    data = data.float()
    if args.cuda:
        data = data.cuda()

    with torch.no_grad():
        logits, _ = model(data)  # → (1, 2)
    # logits_np = logits.cpu().numpy()
    probs_np = F.softmax(logits, dim=-1).cpu().numpy()
    pred_label = int(np.argmax(probs_np, axis=1)[0])

    print(f"Object: {obj_name[0]}")
    # print("🔍 raw logits:", logits_np)
    # print("✨ softmax probabilities:", probs_np)
    print("✔️ 正解ラベル:", int(target.item()))
    print("🏷️ 予測ラベル:", pred_label)
    # ————————————————————————————————


if __name__ == "__main__":
    main()
