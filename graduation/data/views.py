# -*- coding:utf-8 -*-
import os
from django.views.decorators.csrf import csrf_exempt
from django.shortcuts import render
from django.contrib.auth.decorators import login_required
from data.models import File
from django.http import HttpResponseRedirect,FileResponse,HttpResponse
from django.contrib import messages
from django.conf import settings
import pandas as pd
from django.utils.encoding import escape_uri_path
from django.utils.http import urlquote
import re
import json

from django.http import JsonResponse #返回ajax的数据的话需要先引入JsonResponse包

# Create your views here.
def spider(request):
    return render(request,'data/spider.html')

@login_required()
def manage(request):
    if request.method=='POST':
        user = request.user
        file = request.FILES['newfile']
        fileclass = str(file.name).split('.')[-1]
        if fileclass not in ['csv','xls','xlsx']:
            messages.error(request,'文件格式不正确')
            return HttpResponseRedirect('/data/manage/')
        File.objects.create(name=file.name,path=file,user=user)
        return HttpResponseRedirect('/data/manage/')
    if request.method=='GET':
        user1 = request.user
        files = user1.file_set.all()
        try:
            fid = request.GET['fid']
            files = File.objects.filter(id=fid)
        except:
            pass
        return render(request,'data/manage.html',locals())
@login_required()
def delete(request):
    if request.method=='GET':
        delid = int(request.GET['delete'])
        filedel = File.objects.get(id=delid)
        filedel.delete()
        file = settings.BASE_DIR + '\\media\\' + str(filedel.path).replace('/', '\\')
        os.remove(file)
        if filedel.feature_path:
            os.remove(filedel.feature_path)
        return HttpResponseRedirect('/data/manage/')
@login_required()
def change_del(request):
    if request.method=='GET':
        row = int(request.GET['row'])
        id = int(request.GET['id'])
        file = File.objects.get(id=id)

        path = settings.BASE_DIR.replace('\\', '/') + '/media/' + str(file.path)
        fclass = str(file.name).split('.')[-1]
        if fclass == 'csv':
            try:
                data = pd.read_csv(path, encoding='gbk')
            except:
                try:
                    # data = pd.read_csv(path, encoding='latin-1')
                    data = pd.read_csv(path, encoding='utf-8')
                except:
                    messages.error(request, '文件格式有问题，请删除')
                    return HttpResponseRedirect('/data/manage/')
        elif fclass in ['xls', 'xlsx']:
            data = pd.read_excel(path)

        data = data.drop(data.index[[row]])
        if fclass == 'csv':
            data.to_csv(path,index=False)
        elif fclass in ['xls', 'xlsx']:
            data.to_excel(path,index=False)
        file.save()

        return HttpResponseRedirect("/data/look_file?fileid={}".format(id))
@login_required()
def look_file(request):
    if request.method=='GET':
        fileid = request.GET['fileid']
        change = request.GET.get('change','no')
        file = File.objects.get(id=fileid)
        # print(settings.BASE_DIR)
        path = settings.BASE_DIR.replace('\\','/') + '/media/'+str(file.path)
        fclass = str(file.name).split('.')[-1]
        if fclass =='csv':
            try:
                data = pd.read_csv(path,encoding='utf-8')
            except:
                try:
                    # data = pd.read_csv(path, encoding='latin-1')
                    data = pd.read_csv(path, encoding='gbk')
                except:
                    messages.error(request,'文件格式有问题，请删除')
                    return HttpResponseRedirect('/data/manage/')
        elif fclass in ['xls','xlsx']:
            data = pd.read_excel(path)

        #把pandas数据放到前端
        # print(data.head())
        tables = data.values
        label = data.columns.values
        print('标签：',label)
        label_type=[]
        for m in data.dtypes.values:
            label_type.append(m.name)
        # print(type(label_type))
        # print(type(label_type[0]))
        # print(label_type)
        # print(type(label_type[0].values))
        #完------------------------------
        return render(request,'data/fileinfo.html',locals())
    if request.method == 'POST':
        fileid = request.POST['id']
        data = request.POST['changes']
        row = int(request.POST['row'])
        datalist = data.split(',')

        file = File.objects.get(id=fileid)
        # print(settings.BASE_DIR)
        path = settings.BASE_DIR.replace('\\', '/') + '/media/' + str(file.path)
        fclass = str(file.name).split('.')[-1]
        if fclass == 'csv':
            try:
                data = pd.read_csv(path, encoding='gbk')
            except:
                try:
                    # data = pd.read_csv(path, encoding='latin-1')
                    data = pd.read_csv(path, encoding='utf-8')
                except:
                    messages.error(request, '文件格式有问题，请删除')
                    return HttpResponseRedirect('/data/manage/')
        elif fclass in ['xls', 'xlsx']:
            data = pd.read_excel(path)

        i = 0
        try:
            for classx in data.dtypes.values:
                classx = str(classx)
                if 'int' in classx:
                    datalist[i] = int(datalist[i])
                elif 'float' in classx:
                    datalist[i] = float(datalist[i])
                elif 'object' in classx:
                    datalist[i] = str(datalist[i])
                i = i + 1
        except:
            messages.error(request, '数据填写类型有误')
            print('error change')
            tables = data.values
            label = data.columns.values
            # 完------------------------------
            return render(request, 'data/fileinfo.html', locals())

        data.iloc[row] = datalist
        if fclass == 'csv':
            data.to_csv(path, index=False)
        elif fclass in ['xls', 'xlsx']:
            data.to_excel(path, index=False)
        print('success change')
        # ['罗店镇', '宝山区x', '127.80518163200001', '2020', '121.353397885', '31.4074956989999']
        tables = data.values
        label = data.columns.values
        file.save()

        dict = {'tables':tables,
                'label':label,
                'fileid':fileid}

        return render(request,'data/fileinfo.html',dict)

@login_required()
def change_add(request):
    if request.method=='GET':
        fileid = request.GET['fileid']
        change = request.GET.get('change','no')
        file = File.objects.get(id=fileid)
        # print(settings.BASE_DIR)
        path = settings.BASE_DIR.replace('\\','/') + '/media/'+str(file.path)
        fclass = str(file.name).split('.')[-1]
        if fclass =='csv':
            try:
                data = pd.read_csv(path,encoding='gbk')
            except:
                try:
                    # data = pd.read_csv(path, encoding='latin-1')
                    data = pd.read_csv(path, encoding='utf-8')
                except:
                    messages.error(request,'文件格式有问题，请删除')
                    return HttpResponseRedirect('/data/manage/')
        elif fclass in ['xls','xlsx']:
            data = pd.read_excel(path)

        #把pandas数据放到前端
        tables = data.values
        label = data.columns.values
        label_type=[]
        for m in data.dtypes.values:
            label_type.append(m.name)
        # print(type(label_type))
        # print(type(label_type[0]))
        # print(label_type)
        # print(type(label_type[0].values))
        #完------------------------------
        return render(request,'data/fileinfo.html',locals())
    if request.method == 'POST':
        fileid = request.POST['id']
        data = request.POST['changes']
        datalist = data.split(',')

        file = File.objects.get(id=fileid)
        # print(settings.BASE_DIR)
        path = settings.BASE_DIR.replace('\\', '/') + '/media/' + str(file.path)
        fclass = str(file.name).split('.')[-1]
        if fclass == 'csv':
            try:
                data = pd.read_csv(path, encoding='gbk')
            except:
                try:
                    # data = pd.read_csv(path, encoding='latin-1')
                    data = pd.read_csv(path, encoding='utf-8')
                except:
                    messages.error(request, '文件格式有问题，请删除')
                    return HttpResponseRedirect('/data/manage/')
        elif fclass in ['xls', 'xlsx']:
            data = pd.read_excel(path)

        i = 0
        try:
            for classx in data.dtypes.values:
                classx = str(classx)
                if 'int' in classx:
                    datalist[i] = int(datalist[i])
                elif 'float' in classx:
                    datalist[i] = float(datalist[i])
                elif 'object' in classx:
                    datalist[i] = str(datalist[i])
                i = i + 1
        except:
            messages.error(request, '数据填写类型有误')
            print('error change')
            tables = data.values
            label = data.columns.values
            # 完------------------------------
            return render(request, 'data/fileinfo.html', locals())

        print(data.shape[0])
        print(datalist)
        data.loc[data.shape[0]] = datalist
        if fclass == 'csv':
            data.to_csv(path, index=False)
        elif fclass in ['xls', 'xlsx']:
            data.to_excel(path, index=False)
        print('success add')
        print(data.iloc[-1])

        file.save()

        return HttpResponseRedirect("/data/look_file?fileid={}".format(fileid))

def down(request):
    if request.method=='GET':
        fid = request.GET['down']
        file = File.objects.get(id=fid)
        name = file.name
        # print(settings.BASE_DIR)
        path = str(settings.BASE_DIR).replace('\\', '/') + '/media/' + str(file.path)
        file = open(path, 'rb')
        # name = str(file.name).split('/')[-1]
        response = FileResponse(file)
        response['Content-Type'] = 'application/octet-stream'
        response['Content-Disposition'] = 'attachment;filename="%s"' % urlquote(name)
        print('attachment;filename="%s"' % urlquote(name))
        return response

def data_get(request):
    if request.method=='GET':
        fid = request.GET['file']
        file = File.objects.get(id=fid)
        path = str(settings.BASE_DIR).replace('\\', '/') + '/media/' + str(file.path)
        fclass = str(file.name).split('.')[-1]
        if fclass == 'csv':
            try:
                data = pd.read_csv(path, encoding='gbk')
            except:
                data = pd.read_csv(path, encoding='utf-8')
        elif fclass in ['xls', 'xlsx']:
            data = pd.read_excel(path)
        label = data.columns.values.tolist()

        return HttpResponse(json.dumps({
            "label": label,
        }))
