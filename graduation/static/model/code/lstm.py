# -*- coding:utf-8 -*-
# @Time : 2022-03-13 11:46
# @Author : 肖紫心
# @File : lstm.py
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
import numpy as np
from torch.nn import SmoothL1Loss
from sklearn.metrics import r2_score
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import time
import copy
import pickle

def Dp(y_pred,y_true,q):
    return max([q*(y_pred-y_true),(q-1)*(y_pred-y_true)])

def Rp_num_den(y_preds,y_trues,q):
    numerator = np.sum([Dp(y_pred,y_true,q) for y_pred,y_true in zip(y_preds,y_trues)])
    denominator = np.sum([np.abs(y_true) for y_true in y_trues])
    return numerator,denominator

class Lstmx(torch.nn.Module):
    def __init__(self, input_size: int, hidden_size: int,output_size):
        super(Lstmx, self).__init__()
        self.lstm1 = LSTM(input_size, hidden_size,batch_first=True)
        self.lstm2 = LSTM(hidden_size, hidden_size*2, batch_first=True)
        self.flatten = Flatten(start_dim=0, end_dim=-1)
        self.linear = Linear(hidden_size*2, output_size)

    def forward(self, x):
        out, (data2, c) = self.lstm1(x)
        out, (data2, c) = self.lstm2(data2)
        out = self.flatten(data2)
        output = self.linear(out)
        return output


# 时间窗口划分函数构建
def win_data(data_raw, win_num):
    # data_raw格式：[time_dim,feature_dim(用于预测的数据)]，为numpy对象
    # win_num：时间窗口大小+1

    # 返回
    # data：tensor 结构[bacesize,时间窗口+1，feature]
    all_time_point = data_raw.shape[0]
    data = []
    for i in range(all_time_point - win_num):
        data_tmp = data_raw[i:i + win_num]
        data.append(data_tmp)
    data = torch.tensor(data, dtype=torch.float32)
    return data


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
def traindata(model, optimizer, loss_fn, epoch_num, data0_train, data0_val, device, preN):
    lossall = []
    lossallv = []
    stepall = []
    total_train_step = 0
    for epoch in range(epoch_num):
        model.train()
        lossx = 0
        lossxv = 0
        for data0x in data0_train:
            x_data = data0x[:-preN]
            y_data = data0x[-preN:]

            x_data = x_data.unsqueeze(dim=0)
            y_data = y_data.squeeze(dim=1)

            x_data = x_data.to(device)
            y_data = y_data.to(device)

            pred = model(x_data)
            loss = loss_fn(pred, y_data)
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            lossx = loss + lossx
        model.eval()
        for data0xv in data0_val:
            x_datav = data0xv[:-preN]
            y_datav = data0xv[-preN:]
            x_datav = x_datav.unsqueeze(dim=0)
            y_datav = y_datav.squeeze(dim=1)

            x_datav = x_datav.to(device)
            y_datav = y_datav.to(device)

            predv = model(x_datav)
            lossv = loss_fn(predv, y_datav)

            lossxv = lossv + lossxv
        total_train_step = total_train_step + 1
        lossx = lossx.cpu().detach().numpy()
        lossall.append(np.mean(lossx))
        lossxv = lossxv.cpu().detach().numpy()
        lossallv.append(np.mean(lossxv))
        stepall.append(total_train_step)

    return stepall, lossall, lossallv


# 测试数据
def testdata(model, loss_fn, data0_test, minmax_model, device, preN):
    model.eval()
    lossall = []
    stepall = []
    numx = 0
    num = 0
    den = 0
    lossrmse = 0
    total_test_step = 0
    L1loss = SmoothL1Loss()
    L1all = []
    ypred_list = []
    yture_list = []
    with torch.no_grad():
        for data0x in data0_test:
            x_data = data0x[:-preN]
            y_data = data0x[-preN:]
            x_data = x_data.unsqueeze(dim=0)
            y_data = y_data.squeeze(dim=1)

            x_data = x_data.to(device)
            y_data = y_data.to(device)

            pred = model(x_data)
            num_i, den_i = Rp_num_den(pred.cpu(), y_data.cpu(), .5)
            num += num_i
            den += den_i

            pred = pred.unsqueeze(dim=0)
            pred = minmax_model.inverse_transform(pred.cpu())
            y_data = y_data.unsqueeze(dim=0)
            y_data = minmax_model.inverse_transform(y_data.cpu())

            #             print(pred)
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

def pre1(model, data0, minmax_model, device):
    model.eval()
    with torch.no_grad():
        data0 = data0.to(device)

        pred = model(data0)
        pred = pred.unsqueeze(dim=0)
        real_pred = minmax_model.inverse_transform(pred.cpu())

        #         pred=pred.unsqueeze(dim=0)
        #         data0 = torch.cat([data0,pred],dim=1)
        #         data0 = data0[:,1:,:].cpu().numpy()
        data0 = 0

        return real_pred[0].tolist(), data0


def predict(model_path, data0_pre, minmax_model, step):
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
    # model = torch.load(model_path, map_location=device)
    # 返回类型：list数组
    result = []
    data0 = data0_pre[-1, :, :]
    data0 = data0.unsqueeze(dim=0)
    for i in range(step):
        res1, data0 = pre1(model, data0, minmax_model, device)
        result.append(res1)
        data0 = torch.tensor(data0, dtype=torch.float32)

    return result

if __name__ == '__main__':
    pass



