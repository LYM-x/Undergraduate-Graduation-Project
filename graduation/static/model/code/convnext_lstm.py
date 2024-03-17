# -*- coding:utf-8 -*-
# @Time : 2022-02-08 9:41
# @Author : 肖紫心
# @File : codexzx.py
# @Software : PyCharm

import torch
from torch import Tensor
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear, Sequential, LSTM, ReLU,BatchNorm1d,CrossEntropyLoss,Dropout,Linear,MSELoss
from torch_geometric.nn import GCNConv
from torch_geometric.datasets import Planetoid
import torch.nn.functional as F
import sys
import pandas as pd
import xlrd
from torch.nn import SmoothL1Loss
from sklearn.metrics import r2_score
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import time
import copy
import pickle

from tsfresh import extract_relevant_features,extract_features,select_features
from tsfresh.utilities.dataframe_functions import impute

import warnings
warnings.filterwarnings('ignore')

# 改进网络2：分离网络，参考改进1，和数据不匹配性，tsfresh特征用作分类
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath
from timm.models.registry import register_model


class Block(nn.Module):
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise conv
        self.norm = LayerNorm_new(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x

class ConvNeXt_lstm(nn.Module):
    def __init__(self, in_chans=3, in_channel=20, hidden_channels=20, hidden_channels2=10, out_channels=1, t0=6,
                 depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], drop_path_rate=0.,
                 layer_scale_init_value=1e-6, head_init_scale=1.,
                 ):
        super().__init__()

        self.downsample_layers = nn.ModuleList()  # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=1),
            LayerNorm_new(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                LayerNorm_new(dims[i], eps=1e-6, data_format="channels_first"),
                nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()  # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[Block(dim=dims[i], drop_path=dp_rates[cur + j],
                        layer_scale_init_value=layer_scale_init_value) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)  # final norm layer

        self.head = nn.Linear(dims[-1], hidden_channels2)

        self.apply(self._init_weights)
        self.head.weight.data.mul_(head_init_scale)
        self.head.bias.data.mul_(head_init_scale)

        self.lstm1 = LSTM(in_channel, hidden_channels, batch_first=True)
        self.lstm2 = LSTM(hidden_channels, hidden_channels2, batch_first=True)
        self.flatten = Flatten(start_dim=0, end_dim=-1)
        self.model1 = Sequential(
            Linear(2 * hidden_channels2, hidden_channels2),
            Linear(hidden_channels2, out_channels),
        )

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        return self.norm(x.mean([-2, -1]))  # global average pooling, (N, C, H, W) -> (N, C)

    def forward(self, x1, x2):
        x1 = self.forward_features(x1)
        x1 = self.head(x1)
        x1 = self.flatten(x1)
        out, (data2, c) = self.lstm1(x2)
        out, (data2, c) = self.lstm2(data2)
        data2 = data2.squeeze(dim=0).squeeze(dim=0)
        data = torch.cat((x1, data2))
        output = self.model1(data)

        return output

class LayerNorm_new(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

def Dp(y_pred,y_true,q):
    return max([q*(y_pred-y_true),(q-1)*(y_pred-y_true)])

def Rp_num_den(y_preds,y_trues,q):
    numerator = np.sum([Dp(y_pred,y_true,q) for y_pred,y_true in zip(y_preds,y_trues)])
    denominator = np.sum([np.abs(y_true) for y_true in y_trues])
    return numerator,denominator


# 时间窗口划分函数构建
def win_data(data_raw, win_num, pren,flag):
    # data_raw格式：[time_dim,feature_dim(用于预测的数据)]，为numpy对象
    # win_num：时间窗口大小+1

    # 返回
    # data：tensor 结构[bacesize,时间窗口+1，feature]
    all_time_point = data_raw.shape[0]
    data = []
    data_features = []
    for i in range(all_time_point - win_num):
        data_tmp = data_raw[i:i + win_num]
        if flag==0:
            ynp = np.array(data_tmp[:-pren])
            ydf = pd.DataFrame(ynp,columns=['value'])
            ydf['id'] = [1]*ynp.shape[0]
            ydf['time'] = [i for i in range(ynp.shape[0])]
            features = extract_features(ydf,column_id='id',column_sort='time')
            features = impute(features)
            features = features.dropna(axis=1,how='any')
            features = features.values
            data_features.append(features)
        data.append(data_tmp)
    data = torch.tensor(data, dtype=torch.float32)
    return data, data_features


# 归一化函数
def data_norm(data_raw):
    # data_raw: [time_dim,feature_dim(用于预测的数据)] ,为numpy对象

    # 返回
    # mm：归一化的模型
    # mm_data归一化后的数据，类型为numpy对象,一样的结构
    mm = MinMaxScaler()
    mm_data = mm.fit_transform(data_raw)
    return mm, mm_data


# 训练函数
def traindata(model, optimizer, loss_fn, epoch_num, data0_train, data0_val, file_train, file_val, device, preN):
    lossall = []
    lossallv = []
    stepall = []
    total_train_step = 0
    for epoch in range(epoch_num):
        model.train()
        lossx = 0
        lossxv = 0
        for i, data0x in enumerate(data0_train):
            x_data = data0x[:-preN]
            x_data = x_data.unsqueeze(dim=0)
            y_data = data0x[-preN:]
            y_data = y_data.squeeze(dim=1)

            data = file_train[i]
            data = np.array(data).reshape(16, 16)
            data = torch.tensor(data, dtype=torch.float32)
            data = data.unsqueeze(dim=0).unsqueeze(dim=0)

            data = data.to(device)
            x_data = x_data.to(device)
            y_data = y_data.to(device)

            pred = model(data, x_data)
            loss = loss_fn(pred, y_data)
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            lossx = loss + lossx
        model.eval()
        for i, data0xv in enumerate(data0_val):
            x_datav = data0xv[:-preN]
            x_datav = x_datav.unsqueeze(dim=0)
            y_datav = data0xv[-preN:]
            y_datav = y_datav.squeeze(dim=1)

            datav = file_val[i]
            datav = np.array(datav).reshape(16, 16)
            datav = torch.tensor(datav, dtype=torch.float32)
            datav = datav.unsqueeze(dim=0).unsqueeze(dim=0)

            datav = datav.to(device)
            x_datav = x_datav.to(device)
            y_datav = y_datav.to(device)

            predv = model(datav, x_datav)
            lossv = loss_fn(predv, y_datav)

            lossxv = lossv + lossxv
        total_train_step = total_train_step + 1
        lossx = lossx.cpu().detach().numpy()
        lossall.append(np.mean(lossx))
        lossxv = lossxv.cpu().detach().numpy()
        lossallv.append(np.mean(lossxv))
        #         lossall.append(lossx.cpu().detach().numpy())
        #         lossallv.append(lossxv.cpu().detach().numpy())
        stepall.append(total_train_step)

    return stepall, lossall, lossallv


# 测试数据
def testdata(model, loss_fn, data0_test, file_test, minmax_model, device, preN):
    model.eval()
    lossall = []
    stepall = []
    lossrmse = 0
    numx = 0
    num = 0
    den = 0
    total_test_step = 0
    L1all = []
    L1loss = SmoothL1Loss()
    ypred_list = []
    yture_list = []
    with torch.no_grad():
        for i, data0x in enumerate(data0_test):
            x_data = data0x[:-preN]
            x_data = x_data.unsqueeze(dim=0)
            y_data = data0x[-preN:]
            y_data = y_data.squeeze(dim=1)

            data = file_test[i]
            data = np.array(data).reshape(16, 16)
            data = torch.tensor(data, dtype=torch.float32)
            data = data.unsqueeze(dim=0).unsqueeze(dim=0)

            data = data.to(device)
            y_data = y_data.to(device)
            x_data = x_data.to(device)

            pred = model(data, x_data)
            num_i, den_i = Rp_num_den(pred.cpu(), y_data.cpu(), .5)
            num += num_i
            den += den_i

            pred = pred.unsqueeze(dim=0)
            pred = minmax_model.inverse_transform(pred.cpu())
            y_data = y_data.unsqueeze(dim=0)
            y_data = minmax_model.inverse_transform(y_data.cpu())

            ypred_list.append(pred)
            yture_list.append(y_data)

            y_data = torch.tensor(y_data, dtype=torch.float32)
            pred = torch.tensor(pred, dtype=torch.float32)

            L1all.append(L1loss(pred, y_data))
            loss = loss_fn(pred, y_data)
            lossx = pow(loss, 0.5)  # RMSE
            loss = lossx / y_data  # 误差率
            total_test_step = total_test_step + 1
            lossall.append(loss[0].cpu().detach().numpy())
            stepall.append(total_test_step)
            lossrmse = lossx + lossrmse
            numx = numx + 1

    ypre_l = {}
    num = len(ypred_list)
    for j in range(num):
        lenx = ypred_list[0].shape[1]
        if j == 0:
            for i in range(lenx):
                ypre_l[i] = []
        for i in range(lenx):
            ypre_l[i].append(ypred_list[j][0][i])

    yture_l = {}
    num = len(yture_list)
    for j in range(num):
        lenx = yture_list[0].shape[1]
        if j == 0:
            for i in range(lenx):
                yture_l[i] = []
        for i in range(lenx):
            yture_l[i].append(yture_list[j][0][i])

    r2 = 0
    num = 0
    for key, values in ypre_l.items():
        r2x = r2_score(yture_l[key], ypre_l[key])
        r2 = r2 + r2x
        num = num + 1
    r2 = r2 / num

    L1loss = sum(L1all) / numx
    Rp = (2 * num) / den

    return Rp, lossrmse / numx, L1loss, stepall, lossall, r2


# 预测数据

def pre1(model, data0, datax, minmax_model, device):
    model.eval()
    with torch.no_grad():
        data = np.array(datax).reshape(16, 16)
        data = torch.tensor(data, dtype=torch.float32)
        datax = data.unsqueeze(dim=0).unsqueeze(dim=0)

        data0 = data0.to(device)
        datax = datax.to(device)

        pred = model(datax, data0)
        pred = pred.unsqueeze(dim=0)
        real_pred = minmax_model.inverse_transform(pred.cpu())
        data0 = 0

        return real_pred[0].tolist(), data0


def predict(model_path, data0_pre, file_train, minmax_model, step):
    # data0_pre:tensor类型，tensor.float32,结构为[bach, 时间戳步数，feature]，即为之前训练的data0
    # model_path: 模型路径 pth完整模型文件
    # data1：tensor类型，tensor.float32，[节点数，节点特征数]
    # edge_index：tensor类型，int，[2, 节点连接情况（从0开始给节点取名）]
    # step:预测步数

    # 判断是否是gpu还是cpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with open(model_path, "rb") as get_myprofile:
        model = pickle.load(get_myprofile)
    model.to(device)
    # 返回类型：list数组
    result = []
    data0 = data0_pre[-1, :, :]
    data0 = data0.unsqueeze(dim=0)
    datax = file_train[-1, :]
    for i in range(step):
        res1, data0 = pre1(model, data0, datax, minmax_model, device)
        result.append(res1)
        data0 = torch.tensor(data0, dtype=torch.float32)

    return result