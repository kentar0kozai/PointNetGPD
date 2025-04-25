import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from model.pointnet2_utils import PointNetSetAbstractionMsg, PointNetSetAbstraction
from model.vn_layers import *
from model.vn_pointnet import PointNetEncoder


# チャネルアテンションモジュール（Squeeze-and-Excitationブロック風）
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.fc1 = nn.Conv1d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv1d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: [B, C, L]
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


# セルフアテンションモジュール（Non-local block風）
class SelfAttention(nn.Module):
    def __init__(self, in_dim):
        super(SelfAttention, self).__init__()
        self.query_conv = nn.Conv1d(in_dim, in_dim // 8, 1)
        self.key_conv = nn.Conv1d(in_dim, in_dim // 8, 1)
        self.value_conv = nn.Conv1d(in_dim, in_dim, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x = x.contiguous()  # これで連続メモリに変換！
        B, C, N = x.size()
        proj_query = self.query_conv(x).permute(0, 2, 1)  # [B, N, C//8]
        proj_key = self.key_conv(x)  # [B, C//8, N]
        energy = torch.bmm(proj_query, proj_key)  # [B, N, N]
        attention = self.softmax(energy)  # [B, N, N]
        proj_value = self.value_conv(x)  # [B, C, N]
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))  # [B, C, N]
        out = self.gamma * out + x
        return out


class STN3d(nn.Module):
    def __init__(self, num_points=2500, input_chann=3):
        super(STN3d, self).__init__()
        self.num_points = num_points
        self.conv1 = torch.nn.Conv1d(input_chann, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.mp1 = torch.nn.MaxPool1d(num_points)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.mp1(x)
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.array([1, 0, 0, 0, 1, 0, 0, 0, 1]).astype(np.float32))).view(1, 9).repeat(batchsize, 1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, 3, 3)
        return x


class SimpleSTN3d(nn.Module):
    def __init__(self, num_points=2500, input_chann=3):
        super(SimpleSTN3d, self).__init__()
        self.num_points = num_points
        self.conv1 = torch.nn.Conv1d(input_chann, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 256, 1)
        self.mp1 = torch.nn.MaxPool1d(num_points)
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 9)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(256)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(64)

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.mp1(x)
        x = x.view(-1, 256)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.array([1, 0, 0, 0, 1, 0, 0, 0, 1]).astype(np.float32))).view(1, 9).repeat(batchsize, 1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, 3, 3)
        return x


class DualPointNetfeat(nn.Module):
    def __init__(self, num_points=2500, input_chann=6, global_feat=True):
        super(DualPointNetfeat, self).__init__()
        self.stn1 = SimpleSTN3d(num_points=num_points, input_chann=input_chann // 2)
        self.stn2 = SimpleSTN3d(num_points=num_points, input_chann=input_chann // 2)
        self.conv1 = torch.nn.Conv1d(input_chann, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.mp1 = torch.nn.MaxPool1d(num_points)
        self.num_points = num_points
        self.global_feat = global_feat

    def forward(self, x):
        batchsize = x.size()[0]
        trans1 = self.stn1(x[:, 0:3, :])
        trans2 = self.stn2(x[:, 3:6, :])
        x = x.transpose(2, 1)
        x = torch.cat([torch.bmm(x[..., 0:3], trans1), torch.bmm(x[..., 3:6], trans2)], dim=-1)
        x = x.transpose(2, 1)
        x = F.relu(self.bn1(self.conv1(x)))
        pointfeat = x
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = self.mp1(x)
        x = x.view(-1, 1024)
        if self.global_feat:
            return x, trans1 + trans2
        else:
            x = x.view(-1, 1024, 1).repeat(1, 1, self.num_points)
            return torch.cat([x, pointfeat], 1), trans1 + trans2


class PointNetfeat(nn.Module):
    def __init__(self, num_points=2500, input_chann=3, global_feat=True):
        super(PointNetfeat, self).__init__()
        self.stn = STN3d(num_points=num_points, input_chann=input_chann)
        self.conv1 = torch.nn.Conv1d(input_chann, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.mp1 = torch.nn.MaxPool1d(num_points)
        self.num_points = num_points
        self.global_feat = global_feat

    def forward(self, x):
        batchsize = x.size()[0]
        trans = self.stn(x)
        x = x.transpose(2, 1)
        x = torch.bmm(x, trans)
        # x = torch.cat([torch.bmm(x[..., 0:3], trans), torch.bmm(x[..., 3:6], trans)], dim=-1)
        x = x.transpose(2, 1)
        x = F.relu(self.bn1(self.conv1(x)))
        pointfeat = x
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = self.mp1(x)
        x = x.view(-1, 1024)
        if self.global_feat:
            return x, trans
        else:
            x = x.view(-1, 1024, 1).repeat(1, 1, self.num_points)
            return torch.cat([x, pointfeat], 1), trans


class DualPointNetCls(nn.Module):
    def __init__(self, num_points=2500, input_chann=3, k=2):
        super(DualPointNetCls, self).__init__()
        self.num_points = num_points
        self.feat = DualPointNetfeat(num_points, input_chann=input_chann, global_feat=True)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()

    def forward(self, x):
        x, trans = self.feat(x)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.fc3(x)
        return F.log_softmax(x, dim=-1), trans


class PointNetCls(nn.Module):
    def __init__(self, num_points=2500, input_chann=3, k=2):
        super(PointNetCls, self).__init__()
        self.num_points = num_points
        self.feat = PointNetfeat(num_points, input_chann=input_chann, global_feat=True)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()

    def forward(self, x):
        x, trans = self.feat(x)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.fc3(x)
        return F.log_softmax(x, dim=-1), trans


class PointNetDenseCls(nn.Module):
    def __init__(self, num_points=2500, input_chann=3, k=2):
        super(PointNetDenseCls, self).__init__()
        self.num_points = num_points
        self.k = k
        self.feat = PointNetfeat(num_points, input_chann=input_chann, global_feat=False)
        self.conv1 = torch.nn.Conv1d(1088, 512, 1)
        self.conv2 = torch.nn.Conv1d(512, 256, 1)
        self.conv3 = torch.nn.Conv1d(256, 128, 1)
        self.conv4 = torch.nn.Conv1d(128, self.k, 1)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)

    def forward(self, x):
        batchsize = x.size()[0]
        x, trans = self.feat(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.conv4(x)
        x = x.transpose(2, 1).contiguous()
        x = F.log_softmax(x.view(-1, self.k), dim=-1)
        x = x.view(batchsize, self.num_points, self.k)
        return x, trans


class PointNetAtfeat(nn.Module):
    def __init__(self, num_points=2500, input_chann=3, global_feat=True):
        super(PointNetAtfeat, self).__init__()
        self.stn = STN3d(num_points=num_points, input_chann=input_chann)
        self.conv1 = torch.nn.Conv1d(input_chann, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        # conv2の出力に対して自己アテンションを適用！
        self.attn = SelfAttention(128)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.mp1 = torch.nn.MaxPool1d(num_points)
        self.num_points = num_points
        self.global_feat = global_feat

    def forward(self, x):
        batchsize = x.size()[0]
        trans = self.stn(x)
        x = x.transpose(2, 1)
        x = torch.bmm(x, trans)
        x = x.transpose(2, 1)
        x = F.relu(self.bn1(self.conv1(x)))
        pointfeat = x
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.attn(x)  # ←ここで自己アテンションを適用☆
        x = self.bn3(self.conv3(x))
        x = self.mp1(x)
        x = x.view(-1, 1024)
        if self.global_feat:
            return x, trans
        else:
            x = x.view(-1, 1024, 1).repeat(1, 1, self.num_points)
            return torch.cat([x, pointfeat], 1), trans


class PointNetAtCls(nn.Module):
    def __init__(self, num_points=2500, input_chann=3, k=2):
        super(PointNetAtCls, self).__init__()
        self.num_points = num_points
        self.feat = PointNetAtfeat(num_points, input_chann=input_chann, global_feat=True)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()

    def forward(self, x):
        x, trans = self.feat(x)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.fc3(x)
        return F.log_softmax(x, dim=-1), trans


# PointNet++ の分類器（SSG版）にチャネルアテンションとセルフアテンションを追加！
class PointNetPPCls(nn.Module):
    def __init__(self, num_class=2, input_chann=3):
        """
        PointNet++ の分類器だよ〜🌸
        :param num_point: 入力点群の数
        :param input_chann: 入力チャネル数（通常は3：xyz）
        :param num_class: 分類クラス数
        """
        super(PointNetPPCls, self).__init__()
        self.normal_channel = input_chann > 3
        in_channel = input_chann - 3 if self.normal_channel else 3

        # シングルスケールグルーピング（SSG）のSA層
        self.sa1 = PointNetSetAbstraction(512, 0.2, 32, in_channel, [64, 64, 320], False)
        self.sa2 = PointNetSetAbstraction(128, 0.4, 64, 320 + 3, [128, 128, 640], False)
        self.sa3 = PointNetSetAbstraction(None, None, None, 640 + 3, [256, 512, 1024], True)

        # l2_points にセルフアテンションを適用（チャネル数は640）
        self.self_attn = SelfAttention(640)

        # スキップ接続で結合する特徴は、l3_features (1024) と l2_features (640) の合計で1664チャネル！
        # この特徴にチャネルアテンションを適用して、重要なチャネルを強調するよ♪
        self.channel_attn = ChannelAttention(1024 + 640)

        # fc層
        self.fc1 = nn.Linear(1024 + 640, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(256, num_class)

    def forward(self, xyz):
        """
        :param xyz: 入力点群 [B, C, N]
        :return: 分類結果と中間特徴（l3_points）
        """
        B, C, N = xyz.shape
        if self.normal_channel:
            norm = xyz[:, 3:, :]
            xyz = xyz[:, :3, :]
        else:
            norm = None

        # SA層で特徴抽出
        l1_xyz, l1_points = self.sa1(xyz, norm)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        # ここで l2_points にセルフアテンションを適用するよ〜
        l2_points = self.self_attn(l2_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)

        # l3_points はグローバル集約なので、チャネル次元を squeeze して [B, 1024] にするよ
        l3_features = l3_points.squeeze(-1)
        # l2_points は最大値プーリングして [B, 640] に！
        l2_features = torch.max(l2_points, 2)[0]

        # スキップ接続で特徴を結合
        x = torch.cat([l3_features, l2_features], dim=1)  # shape: [B, 1664]

        # チャネルアテンションを適用するため、x を [B, 1664, 1] に変形
        x_attn = self.channel_attn(x.unsqueeze(-1)).squeeze(-1)
        # アテンション重みを加えた特徴と元の特徴を足して、よりリッチな表現に！
        x = x + x_attn

        x = self.drop1(F.relu(self.bn1(self.fc1(x))))
        x = self.drop2(F.relu(self.bn2(self.fc2(x))))
        x = self.fc3(x)
        x = F.log_softmax(x, dim=-1)
        return x, l3_points


class VnnPointNetCls(nn.Module):
    def __init__(self, args, num_class=2, normal_channel=False):
        super(VnnPointNetCls, self).__init__()
        self.args = args
        if normal_channel:
            channel = 6
        else:
            channel = 3
        self.feat = PointNetEncoder(args, global_feat=True, feature_transform=True, channel=channel)
        self.fc1 = nn.Linear(1024 // 3 * 6, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_class)
        self.dropout = nn.Dropout(p=0.4)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()

    def forward(self, x):
        x, trans, trans_feat = self.feat(x)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.dropout(self.fc2(x))))
        x = self.fc3(x)
        x = F.log_softmax(x, dim=1)
        return x, trans_feat


class PointNetSkipAtfeat(nn.Module):
    def __init__(self, num_points=2500, input_chann=3, global_feat=True):
        super(PointNetSkipAtfeat, self).__init__()
        self.stn = STN3d(num_points=num_points, input_chann=input_chann)
        self.conv1 = torch.nn.Conv1d(input_chann, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        # スキップ用の1×1畳み込みでチャンネル数を合わせるよ！
        self.shortcut_conv = torch.nn.Conv1d(64, 128, 1)
        self.channel_attn = ChannelAttention(128)
        self.attn = SelfAttention(128)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.mp1 = torch.nn.MaxPool1d(num_points)
        self.num_points = num_points
        self.global_feat = global_feat

    def forward(self, x):
        batchsize = x.size()[0]
        trans = self.stn(x)
        x = x.transpose(2, 1)
        x = torch.bmm(x, trans)
        x = x.transpose(2, 1)
        # conv1層
        x1 = F.relu(self.bn1(self.conv1(x)))  # 出力: (batch, 64, num_points)
        # conv1の出力を1×1畳み込みで128チャネルに変換（スキップ用）
        residual = self.shortcut_conv(x1)  # 出力: (batch, 128, num_points)
        # conv2層
        x2 = F.relu(self.bn2(self.conv2(x1)))  # 出力: (batch, 128, num_points)
        # スキップコネクションで残差を加算☆
        x2 = x2 + residual
        # チャネルアテンションで各チャネルの重み付けを調整！
        x2 = self.channel_attn(x2) * x2
        # 自己アテンション適用！
        x2 = self.attn(x2)
        # conv3層
        x3 = self.bn3(self.conv3(x2))
        x3 = self.mp1(x3)
        x3 = x3.view(-1, 1024)
        if self.global_feat:
            return x3, trans
        else:
            x3 = x3.view(-1, 1024, 1).repeat(1, 1, self.num_points)
            return torch.cat([x3, x1], 1), trans


class PointNetSkipAtCls(nn.Module):
    def __init__(self, num_points=2500, input_chann=3, k=2):
        super(PointNetSkipAtCls, self).__init__()
        self.num_points = num_points
        self.feat = PointNetSkipAtfeat(num_points, input_chann=input_chann, global_feat=True)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()

    def forward(self, x):
        x, trans = self.feat(x)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.fc3(x)
        return F.log_softmax(x, dim=-1), trans


if __name__ == "__main__":
    pass
