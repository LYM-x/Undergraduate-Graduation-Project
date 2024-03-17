# -*- coding:utf-8 -*-
# @Time : 2022-03-13 11:46
# @Author : 肖紫心
# @File : transformer.py
# @Software : PyCharm
import torch
from torch import nn
from torch.nn import SmoothL1Loss
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
import warnings
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset

warnings.filterwarnings("ignore")
torch.cuda.set_device(0)


class time_series_decoder_paper(Dataset):
    """synthetic time series dataset from section 5.1"""

    def __init__(self, data, t0, listx, N=4500, preN=18, transform=None):
        """
        Args:
            data:tensor类型，[数目，每条要预测的时间戳数目]
            t0: previous t0 data points to predict from
            N: number of data points
            transform: any transformations to be applied to time series
        """

        self.t0 = t0
        self.N = N
        self.transform = None
        temp = []

        # time points

        for i in range(len(listx)):
            m = listx[i]
            x = torch.arange(m, m + t0 + preN).type(torch.float).unsqueeze(0)
            if (i == 0):
                temp = x
            else:
                temp = torch.cat([temp, x], dim=0)

        self.x = temp

        # sinuisoidal signal
        self.fx = data

        self.masks = self._generate_square_subsequent_mask(t0, preN)

        # print out shapes to confirm desired output
        print("x: {}*{}".format(*list(self.x.shape)),
              "fx: {}*{}".format(*list(self.fx.shape)))

    def __len__(self):
        return len(self.fx)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = (self.x[idx, :],
                  self.fx[idx, :],
                  self.masks)

        if self.transform:
            sample = self.transform(sample)

        return sample

    def _generate_square_subsequent_mask(self, t0, preN):
        mask = torch.zeros(t0 + preN, t0 + preN)
        for i in range(0, t0):
            mask[i, t0:] = 1
        for i in range(t0, t0 + preN):
            mask[i, i + 1:] = 1
        mask = mask.float().masked_fill(mask == 1, float('-inf'))  # .masked_fill(mask == 1, float(0.0))
        return mask


class CausalConv1d(torch.nn.Conv1d):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 dilation=1,
                 groups=1,
                 bias=True):
        super(CausalConv1d, self).__init__(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
            dilation=dilation,
            groups=groups,
            bias=bias)

        self.__padding = (kernel_size - 1) * dilation

    def forward(self, input):
        return super(CausalConv1d, self).forward(F.pad(input, (self.__padding, 0)))


class context_embedding(torch.nn.Module):
    def __init__(self, in_channels=1, embedding_size=256, k=5):
        super(context_embedding, self).__init__()
        self.causal_convolution = CausalConv1d(in_channels, embedding_size, kernel_size=k)

    def forward(self, x):
        x = self.causal_convolution(x)
        return F.tanh(x)

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


class TransformerTimeSeries(torch.nn.Module):
    """
    Time Series application of transformers based on paper

    causal_convolution_layer parameters:
        in_channels: the number of features per time point
        out_channels: the number of features outputted per time point
        kernel_size: k is the width of the 1-D sliding kernel

    nn.Transformer parameters:
        d_model: the size of the embedding vector (input)

    PositionalEncoding parameters:
        d_model: the size of the embedding vector (positional vector)
        dropout: the dropout to be used on the sum of positional+embedding vector

    """

    def __init__(self):
        super(TransformerTimeSeries, self).__init__()
        self.input_embedding = context_embedding(2, 512 * 5, 9)
        #         print(shape(self.input_embedding))
        self.positional_embedding = torch.nn.Embedding(512, 512 * 5)
        #         print(self.positional_embedding.shape)

        self.decode_layer = torch.nn.TransformerEncoderLayer(d_model=512 * 5, nhead=8)
        #         print(self.decode_layer.shape)
        self.transformer_decoder = torch.nn.TransformerEncoder(self.decode_layer, num_layers=3)
        #         print(self.transformer_decoder.shape)

        self.fc1 = torch.nn.Linear(512 * 5, 1)

    #         print(self.fc1.shape)

    def forward(self, x, y, attention_masks):
        # concatenate observed points and time covariate
        # (B*feature_size*n_time_points)
        z = torch.cat((y.unsqueeze(1), x.unsqueeze(1)), 1)

        # input_embedding returns shape (Batch size,embedding size,sequence len) -> need (sequence len,Batch size,embedding_size)
        z_embedding = self.input_embedding(z).permute(2, 0, 1)

        # get my positional embeddings (Batch size, sequence_len, embedding_size) -> need (sequence len,Batch size,embedding_size)
        positional_embeddings = self.positional_embedding(x.type(torch.long)).permute(1, 0, 2)

        input_embedding = z_embedding + positional_embeddings

        transformer_embedding = self.transformer_decoder(input_embedding, attention_masks)

        output = self.fc1(transformer_embedding.permute(1, 0, 2))

        return output


# class TransformerTimeSeries(torch.nn.Module):
#     """
#     Time Series application of transformers based on paper
#
#     causal_convolution_layer parameters:
#         in_channels: the number of features per time point
#         out_channels: the number of features outputted per time point
#         kernel_size: k is the width of the 1-D sliding kernel
#
#     nn.Transformer parameters:
#         d_model: the size of the embedding vector (input)
#
#     PositionalEncoding parameters:
#         d_model: the size of the embedding vector (positional vector)
#         dropout: the dropout to be used on the sum of positional+embedding vector
#
#     """
#
#     def __init__(self):
#         d_model = 512 * 5
#         num_encoder_layers = 2
#         num_decoder_layers = 2
#         nhead = 8
#
#         super(TransformerTimeSeries, self).__init__()
#         self.input_embedding = context_embedding(2, d_model, 9)
#         self.positional_embedding = torch.nn.Embedding(512, d_model)
#         # Encoder
#         encoder_layer = nn.TransformerEncoderLayer(d_model, nhead)
#         encoder_norm = nn.LayerNorm(d_model)
#         self.encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)
#         # Decoder
#         decoder_layer = nn.TransformerDecoderLayer(d_model, nhead)
#         decoder_norm = nn.LayerNorm(d_model)
#         self.decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm)
#
#         self.fc1 = torch.nn.Linear(512 * 5, 1)
#
#     def forward(self, x, y, attention_masks):
#         z = torch.cat((y.unsqueeze(1), x.unsqueeze(1)), 1)
#         z_embedding = self.input_embedding(z).permute(2, 0, 1)
#         positional_embeddings = self.positional_embedding(x.type(torch.long)).permute(1, 0, 2)
#         input_embedding = z_embedding + positional_embeddings
#
#         memory = self.encoder(input_embedding)
#         output = self.decoder(input_embedding, memory, tgt_mask=attention_masks)
#         output = self.fc1(output.permute(1, 0, 2))
#         return output


def Dp(y_pred,y_true,q):
    return max([q*(y_pred-y_true),(q-1)*(y_pred-y_true)])

def Rp_num_den(y_preds,y_trues,q):
    numerator = np.sum([Dp(y_pred,y_true,q) for y_pred,y_true in zip(y_preds,y_trues)])
    denominator = np.sum([np.abs(y_true) for y_true in y_trues])
    return numerator,denominator

def train_epoch(model, train_dl, preN,optimizer, criterion,t0=96):
    model.train()
    train_loss = 0
    n = 0
    for step, (x, y, attention_masks) in enumerate(train_dl):
        optimizer.zero_grad()
        output = model(x.cuda(), y.cuda(), attention_masks[0].cuda())
        loss = criterion(output.squeeze(dim=2)[:, (t0 - 1):(t0 + preN - 1)], y.cuda()[:, t0:])  # not missing data
        loss.backward()
        optimizer.step()

        train_loss += (loss.detach().cpu().item() * x.shape[0])
        n += x.shape[0]
    return train_loss / n


def eval_epoch(model, validation_dl, preN, optimizer, criterion,t0=96):
    model.eval()
    eval_loss = 0
    n = 0
    with torch.no_grad():
        for step, (x, y, attention_masks) in enumerate(validation_dl):
            output = model(x.cuda(), y.cuda(), attention_masks[0].cuda())
            loss = criterion(output.squeeze(dim=2)[:, (t0 - 1):(t0 + preN - 1)], y.cuda()[:, t0:])

            eval_loss += (loss.detach().cpu().item() * x.shape[0])
            n += x.shape[0]

    return eval_loss / n


def test_epoch(model, test_dl, preN, t0, minmax_model, criterion):
    with torch.no_grad():
        predictions = []
        observations = []

        lossall = []
        losspall = []
        L1all = []

        model.eval()
        for step, (x, y, attention_masks) in enumerate(test_dl):
            output = model(x.cuda(), y.cuda(), attention_masks[0].cuda())

            for p, o in zip(output.squeeze(dim=2)[:, (t0 - 1):(t0 + preN - 1)].cpu().numpy().tolist(),
                            y.cuda()[:, t0:].cpu().numpy().tolist()):  # not missing data
                predictions.append(p)
                observations.append(o)

        num = 0
        den = 0
        numx = 0
        L1loss = SmoothL1Loss()
        yre_l = {}
        for index, data in enumerate(predictions):
            num = len(data)
            if index == 0:
                for i in range(num):
                    yre_l[i] = []
            for i in range(num):
                yre_l[i].append(data[i])
        ytu_l = {}
        for index, data in enumerate(observations):
            num = len(data)
            if index == 0:
                for i in range(num):
                    ytu_l[i] = []
            for i in range(num):
                ytu_l[i].append(data[i])

        r2 = 0
        num = 0
        for key, values in ytu_l.items():
            r2x = r2_score(ytu_l[key], yre_l[key])
            r2 = r2 + r2x
            num = num + 1
        r2 = r2 / num

        for y_preds, y_trues in zip(predictions, observations):
            num_i, den_i = Rp_num_den(y_preds, y_trues, .5)
            num += num_i
            den += den_i
            numx = numx + 1

            y_preds = minmax_model.inverse_transform([y_preds])
            y_trues = minmax_model.inverse_transform([y_trues])

            y_preds = torch.tensor(y_preds, dtype=torch.float32)
            y_trues = torch.tensor(y_trues, dtype=torch.float32)
            L1all.append(L1loss(y_preds, y_trues))
            loss = criterion(y_preds, y_trues)
            loss = pow(loss, 0.5)  # RMSE
            lossp = loss / y_trues  # 误差率
            losspall.append(lossp[0][0].cpu().detach().numpy())
            lossall.append(loss.cpu().detach().numpy())

        Rp = (2 * num) / den

    return Rp, sum(lossall) / numx, sum(L1all) / numx, losspall, r2

