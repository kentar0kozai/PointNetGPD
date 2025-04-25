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

from model.dataset import *
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

args.cuda = torch.cuda.is_available()
os.makedirs(args.model_path, exist_ok=True)
if args.cuda:
    torch.cuda.manual_seed(1)

logger = SummaryWriter(os.path.join("./assets/log/", args.tag))
np.random.seed(int(time.time()))


def worker_init_fn(pid):
    np.random.seed(torch.initial_seed() % (2**31 - 1))


def my_collate(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)


grasp_points_num = 750
thresh_good = 0.6
thresh_bad = 0.6
point_channel = 3


def train(model, loader, epoch):
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=20, gamma=0.7)
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    # scheduler = CosineAnnealingLR(optimizer, T_max=100, eta_min=0)
    model.train()
    torch.set_grad_enabled(True)
    correct = 0
    dataset_size = 0
    pbar = tqdm(enumerate(loader), total=len(loader), desc=f"Train Epoch: {epoch}", leave=False)
    for batch_idx, (data, target) in pbar:
        dataset_size += data.shape[0]
        data, target = data.float(), target.long().squeeze()
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        output, _ = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.view_as(pred)).long().cpu().sum()
        if batch_idx % args.log_interval == 0:
            percentage = 100.0 * batch_idx * args.batch_size / len(loader.dataset)
            pbar.set_postfix(
                {"Progress": f"{batch_idx * args.batch_size}/{len(loader.dataset)} ({percentage:.2f}%)", "Loss": loss.item(), "Tag": args.tag}
            )
            logger.add_scalar("train_loss", loss.cpu().item(), batch_idx + epoch * len(loader))
    scheduler.step()
    return float(correct) / float(dataset_size)


def test(model, loader):
    model.eval()
    torch.set_grad_enabled(False)
    test_loss = 0
    correct = 0
    dataset_size = 0
    da = {}
    db = {}
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
            res.append((i, j[0], k))

    test_loss /= len(loader.dataset)
    acc = float(correct) / float(dataset_size)
    return acc, test_loss


def main():

    train_loader = torch.utils.data.DataLoader(
        PointGraspOneViewDataset(
            grasp_points_num=grasp_points_num,
            tag="train",
            grasp_amount_per_file=6500,
            # grasp_amount_per_file=65,
            thresh_good=thresh_good,
            thresh_bad=thresh_bad,
        ),
        batch_size=args.batch_size,
        num_workers=28,
        pin_memory=True,
        shuffle=True,
        worker_init_fn=worker_init_fn,
        collate_fn=my_collate,
    )

    test_loader = torch.utils.data.DataLoader(
        PointGraspOneViewDataset(
            grasp_points_num=grasp_points_num,
            tag="test",
            grasp_amount_per_file=500,
            # grasp_amount_per_file=5,
            thresh_good=thresh_good,
            thresh_bad=thresh_bad,
            with_obj=True,
        ),
        batch_size=args.batch_size,
        num_workers=28,
        pin_memory=True,
        shuffle=True,
        worker_init_fn=worker_init_fn,
        collate_fn=my_collate,
    )

    is_resume = 0
    if args.load_model and args.load_epoch != -1:
        is_resume = 1

    if is_resume or args.mode == "test":
        checkpoint = torch.load(args.load_model, map_location="cpu", weights_only=False)
        if hasattr(checkpoint, "module"):
            model = checkpoint.module
        else:
            model = checkpoint
        model = model.to("cuda:0")

    else:
        model = PointNetSkipAtCls(num_points=grasp_points_num, input_chann=3, k=2)
    if args.cuda:
        # if args.gpu != -1:
        #     print("========== Use Single GPU! ==========")
        #     torch.cuda.set_device(args.gpu)
        #     model = model.cuda()
        # else:
        print("========== Use Multi GPUs! ==========")
        device_id = [0, 1, 2, 3]
        torch.cuda.set_device(device_id[0])
        model = nn.DataParallel(model, device_ids=device_id).cuda()

    if args.mode == "train":
        for epoch in range(is_resume * args.load_epoch, args.epoch):
            acc_train = train(model, train_loader, epoch)
            print("Train done, acc={}".format(acc_train))
            acc, loss = test(model, test_loader)
            print("Test done, acc={}, loss={}".format(acc, loss))
            logger.add_scalar("train_acc", acc_train, epoch)
            logger.add_scalar("test_acc", acc, epoch)
            logger.add_scalar("test_loss", loss, epoch)
            if epoch % args.save_interval == 0:
                path = os.path.join(args.model_path, args.tag + "_{}.model".format(epoch))
                torch.save(model, path)
                print("Save model @ {}".format(path))
    else:
        print("testing...")
        acc, loss = test(model, test_loader)
        print("Test done, acc={}, loss={}".format(acc, loss))


if __name__ == "__main__":
    main()
