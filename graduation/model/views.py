# -*- coding:utf-8 -*-
from django.shortcuts import render
from django.contrib.auth.decorators import login_required
from django.http import HttpResponseRedirect,HttpResponse,FileResponse
import asyncio
from model.models import Model_train
from asgiref.sync import sync_to_async
# Create your views here.
from django.utils.http import urlquote
from torch.utils.data import DataLoader
from static.model.code import lstm,gcn_lstm,transformer,Dataloader,convnext_lstm
from data.models import File
import pandas as pd
from django.conf import settings
import numpy as np
import matplotlib.pyplot as plt
import time
import io
import pickle
from django.core.files.base import File as Filexx
import threading
from torch.nn import *
import torch
import copy


from wsgiref.util import FileWrapper
import tempfile
import zipfile
import csv
import os
import json

from PIL import Image

model1_train = 'no'
model2_train = 'no'
model3_train = 'no'
model4_train = 'no'
models_pre = {}

def plt_to_file(fig):
    canvas = fig.canvas
    buffer = io.BytesIO()  # 获取输入输出流对象
    canvas.print_png(buffer)  # 将画布上的内容打印到输入输出流对象
    data = buffer.getvalue()  # 获取流的值
    buffer.write(data)  # 将数据写入buffer
    return buffer

def model_to_file(model):
    model = pickle.dumps(model)
    buffer = io.BytesIO()  # 获取输入输出流对象
    buffer.write(model)  # 将数据写入buffer
    return buffer

@login_required
def model1(request):
    if request.method=='GET':
        global model1_train
        train = model1_train
        user = request.user
        files = user.file_set.all()
        modelsx = Model_train.objects.filter(user=user,model_class='lstm')
        return render(request,'model/model1.html',locals())
    if request.method == 'POST':
        lr = request.POST['lr']
        win = request.POST['win']
        pren = request.POST['pren']
        epoch = request.POST['epoch']
        hid = request.POST['hid']
        fid = request.POST['file']

        user = request.user
        files = user.file_set.all()
        return render(request, 'model/model1.html', locals())

@login_required
def thread_model1(request):
    if request.method == 'POST':
        global model1_train
        model1_train = 'yes'

        lr = request.POST['lr']
        win = request.POST['win']
        pren = request.POST['pren']
        epoch = request.POST['epoch']
        hid = request.POST['hid']
        fid = request.POST['file']
        user = request.user

        time_col = request.POST['time_col']
        pre_col = request.POST['pre_col']
        print('pre_col:'+str(time_col))
        print('pre_col:' + str(pre_col))

        model_name = request.POST['pic_name']
        print('model_name:',model_name)

        code_new = request.POST.get('code_input','no')
        if code_new:
            pass
        else:
            code_new='no'
        print(code_new)

        args = [lr, win, pren, epoch, hid, fid,time_col,pre_col,model_name,code_new,user]

        test = PrintThread(args)
        test.start()
        return HttpResponseRedirect('/model/model1')

class PrintThread(threading.Thread):  #lstm  datapath = data0x
    def __init__(self,args):
        threading.Thread.__init__(self)
        self.args = args

    def run(self):
        self.args[0] = float(self.args[0])
        for i in range(1, len(self.args)-3):
            self.args[i] = int(self.args[i])
        lr, win, pren, epoch, hid, fid,time_col,pre_col,model_name,code_new,user = self.args
        # 准备数据集
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(torch.cuda.is_available())
        print(torch.__version__)
        file = File.objects.get(id=fid)
        path = str(settings.BASE_DIR).replace('\\', '/') + '/media/' + str(file.path)
        data_path = path
        fclass = str(file.name).split('.')[-1]
        if fclass == 'csv':
            try:
                data0x = pd.read_csv(path, encoding='gbk')
            except:
                data0x = pd.read_csv(path, encoding='utf-8')
        elif fclass in ['xls', 'xlsx']:
            data0x = pd.read_excel(path)

        data0x = data0x.iloc[:,pre_col].values.reshape(-1,1)
        mm, data0 = lstm.data_norm(data0x)
        data0 = lstm.win_data(data0, win)

        # 划分训练和测试集（有时可以加个验证集）

        # mask构建 7：3
        data0_size = data0.shape[0]
        test_size = int(data0_size * 0.3)
        listu = np.random.choice([i for i in range(data0_size)], size=test_size, replace=False)
        listu.sort()

        data0m = data0.numpy()
        data0m_test = data0m[listu, :, :]
        data0m_train = np.delete(data0m, listu, axis=0)
        data0_test = torch.tensor(data0m_test, dtype=torch.float32)
        data0_train = torch.tensor(data0m_train, dtype=torch.float32)

        # 设置参数+初始化网络 lstm
        loss_fn = torch.nn.MSELoss()
        loss_fn = loss_fn.to(device)  # gpu
        learing_rate = lr  # 1e-2
        epoch_num = epoch
        preN = pren
        hidden_channels = hid

        in_channels = data0.shape[2]
        out_channels = preN

        if code_new =='no':
            model = lstm.Lstmx(in_channels, hidden_channels, out_channels)
        else:
            # try:
            import sys
            code_new = 'from torch.nn import * \nimport torch \n' + code_new
            path = str(settings.BASE_DIR).replace('\\', '/') + '/media/' + '/{}'.format(user.id)
            with open(path+'/model_code.py', "w") as f:
                f.write(code_new)
            try:
                with open(path + '/model_code.py', "r") as f:
                    code = f.read()
                    exec(code)
                sys.path.append(path)
                from model_code import Lstmx
                model = Lstmx(in_channels, hidden_channels, out_channels)
                # 删除model_code.py文件
                os.remove(path+'/model_code.py')
            except:
                pass
        optimizer = torch.optim.Adam(model.parameters(), lr=learing_rate)
        model = model.to(device)  # gpu

        # 开始训练
        stepallx, lossallx, lossallvx = lstm.traindata(model, optimizer, loss_fn, epoch_num, data0_train, data0_test,
                                                       device,
                                                       preN)


        print('sucesess train')
        # 画loss图
        fig = plt.figure(3)
        plt.xlabel("Epoch")
        plt.ylabel("MSE Loss")
        plt.plot(stepallx, lossallx)
        plt.plot(stepallx, lossallvx)
        plt.legend(['Train Loss', 'Eval Loss'])
        img1 = plt_to_file(fig)
        plt.close(3)

        # 测试数据
        Rp, Rmse, L1loss, stepall, lossall, r2 = lstm.testdata(model, loss_fn, data0_test, mm, device, preN)


        result_data = [stepallx,lossallx,lossallvx,stepall, lossall]
        print('sucesess test')
        # 画结果图
        labels = [i for i in range(1, lossall[0].shape[0] + 1)]
        list1_2 = np.ones(len(stepall))
        list1_2 = list1_2 - 0.5
        fig = plt.figure(4, figsize=(10, 10))
        plt.grid()
        plt.plot(stepall, lossall, label=labels)
        plt.plot(stepall, list1_2, linestyle='--')
        plt.title('Test--rmse:{}'.format(Rmse))
        plt.legend()
        img2 = plt_to_file(fig)
        plt.close(4)

        # Rp, Rmse,L1loss, r2
        index = '{},{},{},{}'.format(Rp, Rmse, L1loss, r2)
        if(model_name):
            pass
        else:
            model_name = 'model_lstm_{}.pth'.format(time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()))

        newfile = Model_train.objects.create(model_class='lstm', model_name=model_name,
                                                 user=user, value_index=index, data_path=data_path,
                                                 train_index=str(win) + ',' + str(preN)
                                                             + ',' + str(time_col) + ',' + str(pre_col))
        file_content = Filexx(img1)
        newfile.loss_pic.save('loss_lstm.png', file_content)
        newfile.save()
        file_content = Filexx(img2)
        newfile.acc_pic.save('acc_lstm.png', file_content)
        newfile.save()

        model = model_to_file(model)
        file_content = Filexx(model)
        newfile.model_path.save('model_lstm.pth', file_content)
        newfile.save()

        result_data = model_to_file(result_data)
        file_content = Filexx(result_data)
        newfile.result_data.save('result_data_lstm.txt', file_content)
        newfile.save()


        global model1_train
        model1_train = 'done'
        print('success---------------------')

class PrintThread3(threading.Thread): #transformer  datapath = data0x
    def __init__(self,args):
        threading.Thread.__init__(self)
        self.args = args

    def run(self):
        self.args[0] = float(self.args[0])
        for i in range(1, len(self.args)-3):
            self.args[i] = int(self.args[i])
        lr, win, pren, epoch, hid, fid,time_col,pre_col,model_name, code_new,user = self.args

        # 要预测的时间步数
        preN = pren
        # 滑动窗口大小
        window_size = win
        # 准备数据集
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(torch.cuda.is_available())
        print(torch.__version__)
        file = File.objects.get(id=fid)
        path = str(settings.BASE_DIR).replace('\\', '/') + '/media/' + str(file.path)
        data_path = path
        fclass = str(file.name).split('.')[-1]
        if fclass == 'csv':
            try:
                data0x = pd.read_csv(path, encoding='gbk')
            except:
                data0x = pd.read_csv(path, encoding='utf-8')
        elif fclass in ['xls', 'xlsx']:
            data0x = pd.read_excel(path)

        # data0x = data0x.values
        data0x = data0x.iloc[:, pre_col].values.reshape(-1, 1)
        mm, data0 = transformer.data_norm(data0x)
        data0 = transformer.win_data(data0, win)

        data0 = torch.tensor(data0, dtype=torch.float32).squeeze(dim=2)
        t0 = data0.shape[1] - preN
        # 划分训练和测试集（有时可以加个验证集）

        # 划分训练和测试集 7：3
        test_scale = 0.3
        data0_size = data0.shape[0]
        test_size = int(data0_size * test_scale)
        list_test = np.random.choice([i for i in range(data0_size)], size=test_size, replace=False)
        list_test.sort()
        list_train = [i for i in range(data0_size) if i not in list_test]

        data0m = data0.numpy()
        data0m_test = data0m[list_test, :]
        data0m_train = np.delete(data0m, list_test, axis=0)
        data0_test = torch.tensor(data0m_test, dtype=torch.float32)
        data0_train = torch.tensor(data0m_train, dtype=torch.float32)

        test_dataset = transformer.time_series_decoder_paper(data0_test, t0, list_test, data0_test.shape[0], preN)
        train_dataset = transformer.time_series_decoder_paper(data0_train, t0, list_train, data0_train.shape[0], preN)
        validation_dataset = train_dataset

        criterion = torch.nn.MSELoss()
        train_dl = DataLoader(train_dataset, batch_size=1, shuffle=True)
        validation_dl = DataLoader(validation_dataset, batch_size=1)
        test_dl = DataLoader(test_dataset, batch_size=1)

        if code_new == 'no':
            model = transformer.TransformerTimeSeries().cuda()
        else:
            # try:
            import sys
            code_new = 'from torch.nn import * \nimport torch\nimport torch.nn.functional as F\n'+ code_new
            path = str(settings.BASE_DIR).replace('\\', '/') + '/media/' + '/{}'.format(user.id)
            with open(path + '/model_code.py', "w") as f:
                f.write(code_new)
            try:
                sys.path.append(path)
                from model_code import TransformerTimeSeries
                model = TransformerTimeSeries().cuda()
                # 删除model_code.py文件
                os.remove(path + '/model_code.py')
            except:
                pass

        lr = lr  # learning rate
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        epochs = epoch

        train_epoch_loss = []
        eval_epoch_loss = []

        for e, epoch in enumerate(range(epochs)):
            train_loss = []
            eval_loss = []

            l_t = transformer.train_epoch(model, train_dl, preN, optimizer, criterion,t0)
            train_loss.append(l_t)

            l_e = transformer.eval_epoch(model, validation_dl, preN, optimizer, criterion,t0)
            eval_loss.append(l_e)

            Rp, Rmse, L1, losspall, r2 = transformer.test_epoch(model, test_dl, preN, t0, mm, criterion)

            train_epoch_loss.append(np.mean(train_loss))
            eval_epoch_loss.append(np.mean(eval_loss))

        print('sucesess train')
        # 画loss图
        fig = plt.figure(6)
        plt.xlabel("Epoch")
        plt.ylabel("MSE Loss")
        plt.plot(train_epoch_loss)
        plt.plot(eval_epoch_loss)
        plt.legend(['Train Loss', 'Eval Loss'])
        img1 = plt_to_file(fig)
        plt.close(6)

        print('sucesess test')
        # 画结果图
        stepall = [i for i in range(1, len(losspall) + 1)]
        list1_2 = np.ones(len(losspall))
        list1_2 = list1_2 - 0.5
        fig = plt.figure(5,figsize=(10, 10))
        plt.grid()
        plt.plot(stepall, losspall)
        plt.plot(stepall, list1_2, linestyle='--')
        plt.title('Test--rmse:{}'.format(Rmse))
        img2 = plt_to_file(fig)
        plt.close(5)

        # Rp, Rmse,L1loss, r2
        index = '{},{},{},{}'.format(Rp, Rmse, L1, r2)
        if(model_name):
            pass
        else:
            model_name = 'model_transformer_{}.pth'.format(time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()))

        newfile = Model_train.objects.create(model_class='transformer', model_name=model_name,
                                             user=user, value_index=index,data_path=data_path,train_index=str(win)+','+str(preN)
                                             +','+str(time_col)+','+str(pre_col))

        file_content = Filexx(img1)
        newfile.loss_pic.save('loss_transformer.png', file_content)
        newfile.save()
        file_content = Filexx(img2)
        newfile.acc_pic.save('acc_transformer.png', file_content)
        newfile.save()

        model = model_to_file(model)
        file_content = Filexx(model)
        newfile.model_path.save('model_transformer.pth', file_content)
        newfile.save()

        global model3_train
        model3_train = 'done'
        print('success---------------------')

class PrintThread2(threading.Thread): #gcn+lstm  datapath = data0x+data1+endge_data
    def __init__(self,args):
        threading.Thread.__init__(self)
        self.args = args

    def run(self):
        self.args[0] = float(self.args[0])
        for i in range(1, len(self.args) - 3):
            self.args[i] = int(self.args[i])
        lr, win, pren, epoch, hid, fid_data0,fid_data1,target,fid_edge,time_col,pre_col,model_name,code_new, user = self.args

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(torch.cuda.is_available())
        print(torch.__version__)
        data_path = ''

        file = File.objects.get(id=fid_data0)
        path = str(settings.BASE_DIR).replace('\\', '/') + '/media/' + str(file.path)
        data_path = path+data_path
        fclass = str(file.name).split('.')[-1]
        if fclass == 'csv':
            try:
                data0x = pd.read_csv(path, encoding='gbk')
            except:
                data0x = pd.read_csv(path, encoding='utf-8')
        elif fclass in ['xls', 'xlsx']:
            data0x = pd.read_excel(path)
        # data0x = data0x.values
        data0x = data0x.iloc[:, pre_col].values.reshape(-1, 1)
        mm, data0 = gcn_lstm.data_norm(data0x)
        data0 = gcn_lstm.win_data(data0, win)

        file = File.objects.get(id=fid_data1)
        path = str(settings.BASE_DIR).replace('\\', '/') + '/media/' + str(file.path)
        data_path = data_path+','+path
        fclass = str(file.name).split('.')[-1]
        if fclass == 'csv':
            try:
                data1 = pd.read_csv(path, encoding='gbk')
            except:
                data1 = pd.read_csv(path, encoding='utf-8')
        elif fclass in ['xls', 'xlsx']:
            data1 = pd.read_excel(path)
        m, data1 = gcn_lstm.data_norm(data1.values)
        data1 = torch.tensor(data1, dtype=torch.float32)

        file = File.objects.get(id=fid_edge)
        path = str(settings.BASE_DIR).replace('\\', '/') + '/media/' + str(file.path)
        data_path = data_path+','+path
        fclass = str(file.name).split('.')[-1]
        if fclass == 'csv':
            try:
                edge_index = pd.read_csv(path, encoding='gbk')
            except:
                edge_index = pd.read_csv(path, encoding='utf-8')
        elif fclass in ['xls', 'xlsx']:
            edge_index = pd.read_excel(path)

        data = edge_index.values
        x = data.shape[1]
        y = data.shape[0]
        arr = []
        for i in range(x):
            for j in range(y):
                if data[j, i] == 1:
                    arr.append([j, i])
        edge_index = torch.tensor(arr).T

        # mask构建 7：3
        data0_size = data0.shape[0]
        test_size = int(data0_size * 0.3)
        listu = np.random.choice([i for i in range(data0_size)], size=test_size, replace=False)
        listu.sort()

        data0m = data0.numpy()
        data0m_test = data0m[listu, :, :]
        data0m_train = np.delete(data0m, listu, axis=0)
        data0_test = torch.tensor(data0m_test, dtype=torch.float32)
        data0_train = torch.tensor(data0m_train, dtype=torch.float32)

        # 设置参数+初始化网络 gcn_lstm
        loss_fn = gcn_lstm.MSELoss()
        loss_fn = loss_fn.to(device)  # gpu
        learing_rate = lr  # 1e-2
        epoch_num = epoch
        preN = pren
        hidden_channels = hid
        targetx = target

        in_channels_gcn = data1.shape[1]
        in_channels_lstm = data0.shape[2]
        out_channels = preN
        numgcnE = 1
        if code_new == 'no':
            model = gcn_lstm.GCN(in_channels_gcn, in_channels_lstm, hidden_channels, out_channels, numgcnE, targetx)
        else:
            # try:
            import sys
            code_new = 'from torch.nn import * \nimport torch \nfrom torch_geometric.nn import GCNConv\nfrom torch import Tensor\n' + code_new
            path = str(settings.BASE_DIR).replace('\\', '/') + '/media/' + '/{}'.format(user.id)
            with open(path + '/model_code.py', "w") as f:
                f.write(code_new)
            try:
                sys.path.append(path)
                from model_code import GCN
                model = GCN(in_channels_gcn, in_channels_lstm, hidden_channels, out_channels, numgcnE, targetx)
                # 删除model_code.py文件
                os.remove(path + '/model_code.py')
            except:
                pass
            # except:
            #     print('失败')
            #     model = gcn_lstm.GCN(in_channels_gcn, in_channels_lstm, hidden_channels, out_channels, numgcnE, targetx)

        optimizer = torch.optim.Adam(model.parameters(), lr=learing_rate)
        model = model.to(device)  # gpu

        # 开始训练
        stepallx, lossallx, lossallvx = gcn_lstm.traindata(model, optimizer, loss_fn, epoch_num, data0_train, data0_test, data1,
                                                  edge_index, device, preN)

        print('sucesess train')
        # 画loss图
        fig = plt.figure(1)
        plt.xlabel("Epoch")
        plt.ylabel("MSE Loss")
        plt.plot(stepallx, lossallx)
        plt.plot(stepallx, lossallvx)
        plt.legend(['Train Loss', 'Eval Loss'])
        img1 = plt_to_file(fig)
        plt.close(1)

        # 测试数据
        Rp, Rmse, L1loss, stepall, lossall, r2 = gcn_lstm.testdata(model, loss_fn, data0_test, mm, data1, edge_index, device,
                                                          preN)

        print('sucesess test')
        # 画结果图
        labels = [i for i in range(1, lossall[0].shape[0] + 1)]
        list1_2 = np.ones(len(stepall))
        list1_2 = list1_2 - 0.5
        fig = plt.figure(2, figsize=(10, 10))
        plt.grid()
        plt.plot(stepall, lossall, label=labels)
        plt.plot(stepall, list1_2, linestyle='--')
        plt.title('Test--rmse:{}'.format(Rmse))
        plt.legend()
        img2 = plt_to_file(fig)
        plt.close(2)

        # Rp, Rmse,L1loss, r2
        index = '{},{},{},{}'.format(Rp, Rmse, L1loss, r2)
        if(model_name):
            pass
        else:
            model_name = 'model_GCN_lstm_{}.pth'.format(time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()))

        newfile = Model_train.objects.create(model_class='GCN_lstm', model_name=model_name,
                                             user=user, value_index=index,data_path=data_path,train_index=str(win)+','+str(preN)
                                             +','+str(time_col)+','+str(pre_col))

        file_content = Filexx(img1)
        newfile.loss_pic.save('loss_GCN_lstm.png', file_content)
        newfile.save()
        file_content = Filexx(img2)
        newfile.acc_pic.save('acc_GCN_lstm.png', file_content)
        newfile.save()

        model = model_to_file(model)
        file_content = Filexx(model)
        newfile.model_path.save('model_GCN_lstm.pth', file_content)
        newfile.save()

        global model2_train
        model2_train = 'done'
        print('success---------------------')

def down(request):
    if request.method == 'GET':
        user = request.user
        mid = request.GET['down']
        print(mid)
        model = Model_train.objects.get(id=mid)
        # print(settings.BASE_DIR)
        path_loss = str(settings.BASE_DIR).replace('\\', '/') + '/media/' + str(model.loss_pic)
        acc_loss = str(settings.BASE_DIR).replace('\\', '/') + '/media/' + str(model.acc_pic)
        index = model.value_index.split(',')

        path_index = str(settings.BASE_DIR).replace('\\', '/') + '/media/' + '/{}/'.format(user.id)+'./index.csv'
        with open(path_index, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["MAE", "RMSE", "Hugloss","R2"])
            writer.writerow(index)

        file_path_list = [
            {'file_name': 'loss.png', 'file_path': path_loss},
            {'file_name': 'acc.png', 'file_path': acc_loss},
            {'file_name': 'index.csv', 'file_path': path_index},
        ]

        temp = tempfile.TemporaryFile()
        archive = zipfile.ZipFile(temp, 'w', zipfile.ZIP_DEFLATED)
        for file_path_dict in file_path_list:
            file_path = file_path_dict.get('file_path', None)
            file_name = file_path_dict.get('file_name', None)
            archive.write(file_path, file_name)  # TODO need check file exist or not

        archive.close()
        lenth = temp.tell()
        temp.seek(0)

        wrapper = FileWrapper(temp)

        # Using HttpResponse
        response = HttpResponse(wrapper, content_type='application/zip')
        response['Content-Disposition'] = 'attachment; filename=result_{}.zip'.format(model.model_name)
        response['Content-Length'] = lenth  # temp.tell()
        return response

@login_required
def model2(request):
    if request.method=='GET':
        global model2_train
        train = model2_train
        user = request.user
        files = user.file_set.all()
        modelsx = Model_train.objects.filter(user=user,model_class='GCN_lstm')
        return render(request,'model/model2.html',locals())

@login_required
def thread_model2(request):
    if request.method == 'POST':
        global model2_train
        model2_train = 'yes'

        lr = request.POST['lr']
        win = request.POST['win']
        pren = request.POST['pren']
        epoch = request.POST['epoch']
        hid = request.POST['hid']
        fid_data0 = request.POST['file_data0']
        fid_data1 = request.POST['file_data1']
        target = request.POST['target']
        fid_edge = request.POST['file_edge']

        time_col = request.POST['time_col']
        pre_col = request.POST['pre_col']
        model_name = request.POST['pic_name']

        code_new = request.POST.get('code_input','no')
        if code_new:
            pass
        else:
            code_new='no'

        user = request.user

        args = [lr, win, pren, epoch, hid, fid_data0,fid_data1,target,fid_edge,time_col,pre_col,model_name,code_new,user]

        test = PrintThread2(args)
        test.start()
        return HttpResponseRedirect('/model/model2')

@login_required
def model3(request):
    if request.method=='GET':
        global model3_train
        train = model3_train
        user = request.user
        files = user.file_set.all()
        modelsx = Model_train.objects.filter(user=user,model_class='transformer')
        return render(request,'model/model3.html',locals())

@login_required
def thread_model3(request):
    if request.method == 'POST':
        global model3_train
        model3_train = 'yes'

        lr = request.POST['lr']
        win = request.POST['win']
        pren = request.POST['pren']
        epoch = request.POST['epoch']
        hid = request.POST['hid']
        fid = request.POST['file']
        user = request.user

        time_col = request.POST['time_col']
        pre_col = request.POST['pre_col']
        model_name = request.POST['pic_name']

        code_new = request.POST.get('code_input', 'no')
        if code_new:
            pass
        else:
            code_new = 'no'

        args = [lr, win, pren, epoch, hid, fid,time_col,pre_col,model_name,code_new,user]

        test = PrintThread3(args)
        test.start()
        return HttpResponseRedirect('/model/model3')

@login_required
def model4(request):
    if request.method=='GET':
        global model4_train
        train = model4_train
        user = request.user
        files = user.file_set.all()
        modelsx = Model_train.objects.filter(user=user,model_class='ConvNext_lstm')
        return render(request,'model/model4.html',locals())

@login_required
def thread_model4(request):
    if request.method == 'POST':
        global model4_train
        model4_train = 'yes'

        lr = request.POST['lr']
        win = request.POST['win']
        pren = request.POST['pren']
        epoch = request.POST['epoch']
        hid1 = request.POST['hid1']
        hid2 = request.POST['hid2']
        fid = request.POST['file']
        user = request.user

        time_col = request.POST['time_col']
        pre_col = request.POST['pre_col']
        model_name = request.POST['pic_name']

        code_new = request.POST.get('code_input', 'no')
        if code_new:
            pass
        else:
            code_new = 'no'

        args = [lr, win, pren, epoch, hid1,hid2 ,fid,time_col,pre_col,model_name,code_new,user]

        test = PrintThread4(args)
        test.start()
        return HttpResponseRedirect('/model/model4')



class PrintThread4(threading.Thread): #ConvNext_lstm
    def __init__(self,args):
        threading.Thread.__init__(self)
        self.args = args

    def run(self):
        self.args[0] = float(self.args[0])
        for i in range(1, len(self.args) - 3):
            self.args[i] = int(self.args[i])
        lr, win, pren, epoch, hid1,hid2 ,fid,time_col,pre_col,model_name,code_new,user = self.args

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(torch.cuda.is_available())
        print(torch.__version__)
        data_path = ''

        file = File.objects.get(id=fid)
        path = str(settings.BASE_DIR).replace('\\', '/') + '/media/' + str(file.path)
        data_path = path+data_path
        fclass = str(file.name).split('.')[-1]
        if fclass == 'csv':
            try:
                data0x = pd.read_csv(path, encoding='gbk')
            except:
                data0x = pd.read_csv(path, encoding='utf-8')
        elif fclass in ['xls', 'xlsx']:
            data0x = pd.read_excel(path)
        # data0x = data0x.values
        data0x = data0x.iloc[:, pre_col].values.reshape(-1, 1)
        mm, data0 = convnext_lstm.data_norm(data0x)

        flag=0
        filex = ''
        if file.feature_path:
            flag=1
        data0,data0_features = convnext_lstm.win_data(data0, win,pren,flag)
        if flag==0:
            data0_features = torch.tensor(data0_features,dtype = torch.float32)
            data0_features = data0_features.squeeze(dim=1).squeeze(dim=1)
            data0_features = np.array(data0_features)
            filex = pd.DataFrame(data0_features)
            filex = filex[filex!=0].dropna(axis=1,how='all')
            print(filex.shape)
            for i in range(256-filex.shape[1]):
                filex['add{}'.format(i)]=0
            filex.fillna(0, inplace=True)
            print(filex.shape)
            path_2 = path.rsplit('/',1)[0]+'/features_{}_{}.csv'.format(file.name,fid)
            filex.to_csv(path_2,index=None)
            file.feature_path = path_2
            file.save()
            data_path = data_path+','+path_2
        elif flag==1:
            filex = pd.read_csv(file.feature_path)


        # mask构建 7：3
        data0_size = data0.shape[0]
        test_size = int(data0_size * 0.3)
        listu = np.random.choice([i for i in range(data0_size)], size=test_size, replace=False)
        listu.sort()

        data0m = data0.numpy()
        data0m_test = data0m[listu, :, :]
        data0m_train = np.delete(data0m, listu, axis=0)
        data0_test = torch.tensor(data0m_test, dtype=torch.float32)
        data0_train = torch.tensor(data0m_train, dtype=torch.float32)

        # features构建
        filem = filex.values
        filem_test = filem[listu, :]
        filem_train = np.delete(filem, listu, axis=0)
        file_test = torch.tensor(filem_test, dtype=torch.float32)
        file_train = torch.tensor(filem_train, dtype=torch.float32)

        # 设置参数+初始化网络 gcn_lstm
        loss_fn = gcn_lstm.MSELoss()
        loss_fn = loss_fn.to(device)  # gpu
        learing_rate = lr  # 1e-2
        epoch_num = epoch
        preN = pren
        hidden_channels = hid1
        hidden_channels2 = hid2
        t0 = win - pren

        if code_new == 'no':
            model = convnext_lstm.ConvNeXt_lstm(hidden_channels=hidden_channels, hidden_channels2=hidden_channels2,
                                                depths=[3, 3, 9, 3],
                                                dims=[48, 96, 192, 384], in_chans=1,
                                                in_channel=data0.shape[2], out_channels=preN, t0=t0)
        else:
            # try:
            import sys
            code_new = 'import torch.nn as nn\nfrom torch.nn import *\nimport torch\nimport torch.nn.functional as F\nfrom timm.models.layers import *\n' + code_new
            path = str(settings.BASE_DIR).replace('\\', '/') + '/media/' + '/{}'.format(user.id)
            with open(path + '/model_code.py', "w") as f:
                f.write(code_new)
            try:
                sys.path.append(path)
                from model_code import ConvNeXt_lstm
                model = ConvNeXt_lstm(hidden_channels=hidden_channels, hidden_channels2=hidden_channels2,
                                                    depths=[3, 3, 9, 3],
                                                    dims=[48, 96, 192, 384], in_chans=1,
                                                    in_channel=data0.shape[2], out_channels=preN, t0=t0)
                # 删除model_code.py文件
                os.remove(path + '/model_code.py')
            except:
                pass

        optimizer = torch.optim.Adam(model.parameters(), lr=learing_rate)
        model = model.to(device)  # gpu

        # 开始训练
        stepallx, lossallx, lossallvx = convnext_lstm.traindata(model, optimizer, loss_fn, epoch_num, data0_train, data0_test,
                                                  file_train, file_test, device, preN)

        print('sucesess train')
        # 画loss图
        fig = plt.figure(1)
        plt.xlabel("Epoch")
        plt.ylabel("MSE Loss")
        plt.plot(stepallx, lossallx)
        plt.plot(stepallx, lossallvx)
        plt.legend(['Train Loss', 'Eval Loss'])
        img1 = plt_to_file(fig)
        plt.close(1)

        # 测试数据
        Rp, Rmse, L1loss, stepall, lossall, r2 = convnext_lstm.testdata(model, loss_fn, data0_test, file_test, mm, device, preN)

        print('sucesess test')
        # 画结果图
        labels = [i for i in range(1, lossall[0].shape[0] + 1)]
        list1_2 = np.ones(len(stepall))
        list1_2 = list1_2 - 0.5
        fig = plt.figure(2, figsize=(10, 10))
        plt.grid()
        plt.plot(stepall, lossall, label=labels)
        plt.plot(stepall, list1_2, linestyle='--')
        plt.title('Test--rmse:{}'.format(Rmse))
        plt.legend()
        img2 = plt_to_file(fig)
        plt.close(2)

        # Rp, Rmse,L1loss, r2
        index = '{},{},{},{}'.format(Rp, Rmse, L1loss, r2)
        if(model_name):
            pass
        else:
            model_name = 'model_ConvNext_lstm_{}.pth'.format(time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()))

        newfile = Model_train.objects.create(model_class='ConvNext_lstm', model_name=model_name,
                                             user=user, value_index=index,data_path=data_path,train_index=str(win)+','+str(preN)
                                             +','+str(time_col)+','+str(pre_col))

        file_content = Filexx(img1)
        newfile.loss_pic.save('loss_ConvNext_lstm.png', file_content)
        newfile.save()
        file_content = Filexx(img2)
        newfile.acc_pic.save('acc_ConvNext_lstm.png', file_content)
        newfile.save()

        model = model_to_file(model)
        file_content = Filexx(model)
        newfile.model_path.save('model_ConvNext_lstm.pth', file_content)
        newfile.save()

        global model4_train
        model4_train = 'done'
        print('success---------------------')


@login_required
def models(request):
    if request.method=='GET':
        global models_pre
        train = models_pre
        print(train)
        user = request.user

        models = user.model_train_set.all()
        files = models
        try:
            fid = request.GET['fid']
            models = Model_train.objects.filter(id=fid)
        except:
            pass

        dataset = {}
        xaxis = {}
        pren = {}
        for pic in models:
            if (pic.pre):
                file_path = settings.BASE_DIR.replace('\\', '/') + '/media/' + str(pic.pre_result)
                with open(file_path, "rb") as get_myprofile:
                    data = pickle.load(get_myprofile)
                dataset['%s' % str(pic.id)] = data.iloc[:,0].values.tolist()
                xaxis['%s' % str(pic.id)] = data.iloc[:, 1].values.tolist()
                pren['%s' % str(pic.id)] = data.columns.values[0]
        return render(request, 'model/models.html', locals())
    if request.method == 'POST':
        type = request.POST['name_type']
        ex_name = request.POST['ex_name']
        user = request.user
        Experiment.objects.create(name=ex_name, ex_class=type,user=user)
        viewpics = user.experiment_set.all()
        files = Model_train.objects.filter(user=user)
        return render(request, 'model/models.html', locals())

class PrintThreads(threading.Thread):  #预测结果均为 numpy array数据类型  [[y1,y2,y3,...]]
    def __init__(self, args):
        threading.Thread.__init__(self)
        self.args = args

    def run(self):
        print('开始线程')
        self.args[1] = int(self.args[1])
        mid,step, user = self.args
        model = Model_train.objects.get(id=mid)
        model_path = str(settings.BASE_DIR).replace('\\', '/') + '/media/' + str(model.model_path)
        data_path = model.data_path
        model_name = model.model_name

        index = model.train_index
        win, preN,time_col,pre_col = index.split(',')
        win = int(win)
        preN = int(preN)
        time_col = int(time_col)
        pre_col = int(pre_col)

        if model.model_class == 'GCN_lstm':
            paths = data_path.split(',')
            # 预测数据 data0x+data1+endge_data
            fclass = str(paths[0]).split('.')[-1]
            if fclass == 'csv':
                try:
                    data0x = pd.read_csv(paths[0], encoding='gbk')
                except:
                    data0x = pd.read_csv(paths[0], encoding='utf-8')
            elif fclass in ['xls', 'xlsx']:
                data0x = pd.read_excel(paths[0])

            fclass = str(paths[1]).split('.')[-1]
            if fclass == 'csv':
                try:
                    data1 = pd.read_csv(paths[1], encoding='gbk')
                except:
                    data1 = pd.read_csv(paths[1], encoding='utf-8')
            elif fclass in ['xls', 'xlsx']:
                data1 = pd.read_excel(paths[1])
            m, data1 = gcn_lstm.data_norm(data1.values)
            data1 = torch.tensor(data1, dtype=torch.float32)

            fclass = str(paths[2]).split('.')[-1]
            if fclass == 'csv':
                try:
                    edge_index = pd.read_csv(paths[2], encoding='gbk')
                except:
                    edge_index = pd.read_csv(paths[2], encoding='utf-8')
            elif fclass in ['xls', 'xlsx']:
                edge_index = pd.read_excel(paths[2])
            data = edge_index.values
            x = data.shape[1]
            y = data.shape[0]
            arr = []
            for i in range(x):
                for j in range(y):
                    if data[j, i] == 1:
                        arr.append([j, i])
            edge_index = torch.tensor(arr).T

            # data0x = data0x.values
            data0x_time = data0x.iloc[:, time_col].values
            data0x = data0x.iloc[:, pre_col].values.reshape(-1, 1)
            minmax_model, data0 = gcn_lstm.data_norm(data0x)
            data0_pre = gcn_lstm.win_data(data0, win)
            result = gcn_lstm.predict(model_path, data0_pre, minmax_model, data1, edge_index, step)
            print('预测结果：'+str(result))
        elif model.model_class == 'ConvNext_lstm':
            flag = 1
            paths = data_path.split(',')
            # 预测数据 data0x+data1+endge_data
            fclass = str(paths[0]).split('.')[-1]
            if fclass == 'csv':
                try:
                    data0x = pd.read_csv(paths[0], encoding='gbk')
                except:
                    data0x = pd.read_csv(paths[0], encoding='utf-8')
            elif fclass in ['xls', 'xlsx']:
                data0x = pd.read_excel(paths[0])

            fclass = str(paths[1]).split('.')[-1]
            if fclass == 'csv':
                try:
                    data1 = pd.read_csv(paths[1], encoding='gbk')
                except:
                    data1 = pd.read_csv(paths[1], encoding='utf-8')
            elif fclass in ['xls', 'xlsx']:
                data1 = pd.read_excel(paths[1])
            m, data1 = gcn_lstm.data_norm(data1.values)
            data1 = torch.tensor(data1, dtype=torch.float32)

            # data0x = data0x.values
            data0x_time = data0x.iloc[:, time_col].values
            data0x = data0x.iloc[:, pre_col].values.reshape(-1, 1)
            minmax_model, data0 = convnext_lstm.data_norm(data0x)

            data0, data0_features = convnext_lstm.win_data(data0, win, preN, flag)
            result = convnext_lstm.predict(model_path, data0, data1, minmax_model, step)
        else:
            # 预测数据
            fclass = str(data_path).split('.')[-1]
            data0x = ''
            if fclass == 'csv':
                try:
                    data0x = pd.read_csv(data_path, encoding='gbk')
                except:
                    data0x = pd.read_csv(data_path, encoding='utf-8')
            elif fclass in ['xls', 'xlsx']:
                data0x = pd.read_excel(data_path)

            data0x_time = data0x.iloc[:, time_col].values
            data0x = data0x.iloc[:, pre_col].values.reshape(-1, 1)
            minmax_model, data0 = gcn_lstm.data_norm(data0x)
            print('------------------------------------------------------')
            if model.model_class == 'lstm':
                data0_pre = gcn_lstm.win_data(data0, win)
                result = lstm.predict(model_path, data0_pre, minmax_model, step)
            elif model.model_class == 'transformer':
                print(model_path)
                with open(model_path, "rb") as get_myprofile:
                    model = pickle.load(get_myprofile)
                data0 = transformer.win_data(data0, win).squeeze(dim=2)
                list_pre = [i for i in range(data0.shape[0])]
                t0 = win - preN
                test_dataset = Dataloader.time_series_decoder_paper(data0, t0, list_pre, data0.shape[0], preN)
                x, y, attention_masks = test_dataset.__getitem__(-1)
                x = x.unsqueeze(dim=0)
                y = y.unsqueeze(dim=0)
                attention_masks = attention_masks.unsqueeze(dim=0)
                output = model(x.cuda(), y.cuda(), attention_masks[0].cuda())
                pre = []
                for i in output:
                    for j in i:
                        pre.append(j[0].cpu().detach().numpy())
                result = minmax_model.inverse_transform([pre])[0][-preN:]
                result = np.expand_dims(result, axis=0)
            print('预测结果：' + str(result))

        if 'list' not in str(type(result)):
            result = result[0].tolist()
        else:
            result = result[0]

        data0x = data0x.T.tolist()[0]
        print(len(data0x))
        data0x.extend(result)
        print(len(data0x))

        # data0x = data0x.T.tolist()[0]
        data0x_time = data0x_time.tolist()
        for i in range(preN):
            data0x_time.append('pre-'+str(i))
        result_plot = pd.DataFrame({
            str(preN):data0x,
            'time':data0x_time
        })

        path = model.pre_result
        if path:
            path_pre = str(settings.BASE_DIR).replace('\\', '/') + '/media/' + str(path)
            os.remove(path_pre)
        result_plot = model_to_file(result_plot)
        file_content = Filexx(result_plot)
        model.pre_result.save('pre_result', file_content)
        model.save()

        model.pre = True
        model.save()

        global models_pre
        models_pre[mid] = 'done'

@login_required
def thread_models(request):
    if request.method == 'POST':
        global models_pre

        step = request.POST.get('step','1')
        if(not step):
            step='1'
        mid = request.POST['mid']
        user = request.user

        models_pre[mid] = 'yes'

        args = [mid,step, user]

        test = PrintThreads(args)
        test.start()
        return HttpResponseRedirect('/model/models')

def models_get(request):
    if request.method=='GET':
        fid = request.GET['file']
        file = File.objects.get(id=fid)
        path = str(settings.BASE_DIR).replace('\\', '/') + '/media/' + str(file.path)
        fclass = str(file.name).split('.')[-1]
        if fclass == 'csv':
            try:
                data = pd.read_csv(path, encoding='utf-8')
            except:
                data = pd.read_csv(path, encoding='gbk')
        elif fclass in ['xls', 'xlsx']:
            data = pd.read_excel(path)
        label = data.columns.values.tolist()

        return HttpResponse(json.dumps({
            "label": label,
        }))

def models_del(request):
    if request.method == 'GET':
        mid = request.GET['delete']
        model = Model_train.objects.get(id=mid)
        path_loss = str(settings.BASE_DIR).replace('\\', '/') + '/media/' + str(model.loss_pic)
        acc_loss = str(settings.BASE_DIR).replace('\\', '/') + '/media/' + str(model.acc_pic)
        model_file = str(settings.BASE_DIR).replace('\\', '/') + '/media/' + str(model.model_path)

        model.delete()
        os.remove(path_loss)
        os.remove(acc_loss)
        os.remove(model_file)

        classname = model.model_class

        if(classname=='lstm'):#GCN_lstm,transformer,ConvNext_lstm
            return HttpResponseRedirect('/model/model1/')
        elif(classname=='GCN_lstm'):
            return HttpResponseRedirect('/model/model2/')
        elif (classname == 'transformer'):
            return HttpResponseRedirect('/model/model3/')
        elif (classname == 'ConvNext_lstm'):
            return HttpResponseRedirect('/model/model4/')

@login_required
def models_code_change(request):
    if request.method == 'POST':
        model_class = request.POST['model']
        code = request.POST['code']
        print(model_class)
        print(code)
        return HttpResponseRedirect('/model/model1')

def dels(request):
    if request.method == 'GET':
        mid = request.GET['delete']
        model = Model_train.objects.get(id=mid)
        path_loss = str(settings.BASE_DIR).replace('\\', '/') + '/media/' + str(model.loss_pic)
        acc_loss = str(settings.BASE_DIR).replace('\\', '/') + '/media/' + str(model.acc_pic)
        model_file = str(settings.BASE_DIR).replace('\\', '/') + '/media/' + str(model.model_path)

        model.delete()
        os.remove(path_loss)
        os.remove(acc_loss)
        os.remove(model_file)

        return HttpResponseRedirect('/model/models/')




