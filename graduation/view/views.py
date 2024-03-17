# -*- coding:utf-8 -*-
import os.path

import numpy as np
from django.shortcuts import render
from django.conf import settings
from django.contrib.auth.decorators import login_required
from django.http import HttpResponseRedirect,HttpResponse
from view.models import Viewpic,Viewpic3d,Map
from index.models import Viewleft,Viewright
import pandas as pd
from model.models import Model_train
from data.models import File
import io
from django.core.files.base import File as Filexx
import pickle
from django.db.models import Q

def model_to_file(model):
    model = pickle.dumps(model)
    buffer = io.BytesIO()  # 获取输入输出流对象
    buffer.write(model)  # 将数据写入buffer
    return buffer

@login_required()
def IndexView(request):
    if request.method == 'GET':
        user = request.user
        viewpics = user.viewpic_set.all()
        dataset={}
        label={}
        for pic in viewpics:
            if(pic.data_file):
                file_path = settings.BASE_DIR.replace('\\', '/') + '/media/' + str(pic.data_file)
                fclass = str(pic.data_file).split('.')[-1]
                data = []
                if fclass == 'csv':
                    try:
                        data = pd.read_csv(file_path, encoding='utf-8')
                    except:
                        try:
                            # data = pd.read_csv(path, encoding='latin-1')
                            data = pd.read_csv(file_path, encoding='gbk')
                        except:
                            return HttpResponseRedirect('/data/manage/')
                elif fclass in ['xls', 'xlsx']:
                    data = pd.read_excel(file_path)
                dataset['%s'%str(pic.id)] = data.fillna(0).values.T.tolist()
                label['%s' % str(pic.id)] = data.columns.values.tolist()
        return render(request, 'view/view2d.html',locals())
    if request.method == 'POST':
        pic_type = request.POST['name_type']
        pic_name = request.POST['pic_name']
        print('选择添加的图片信息为：'+pic_type+pic_name)
        user = request.user
        Viewpic.objects.create(class_name=pic_type,user=user,pic_name=pic_name)
        viewpics = user.viewpic_set.all()
        return render(request, 'view/view2d.html',locals())

def pic2d_change(request):
    if request.method=='GET':
        return render(request, 'view/view2d.html')
    if request.method=='POST':
        file = request.FILES.get('upfile')
        id = request.POST['id']
        type_name = request.POST['type_name']
        fileclass = str(file.name).split('.')[-1]
        pic = Viewpic.objects.get(id=id)
        if(pic.data_file):
            print('有了')
            filex = settings.BASE_DIR + '\\media\\' + str(pic.data_file).replace('/', '\\')
            os.remove(filex)
        pic.data_file = file
        pic.save()

        return HttpResponseRedirect('/view/pic2d/')

def pic2d_del(request):
    if request.method == 'POST':
        del_id = request.POST['id']
        pic = Viewpic.objects.get(id=del_id)
        picl = Viewleft.objects.filter(viewpic=del_id)
        picr = Viewright.objects.filter(viewpic=del_id)
        if picl:
            for piclx in picl:
                piclx.viewpic='-1'
                piclx.save()
        if picr:
            for picrx in picr:
                picrx.viewpic='-1'
                picrx.save()
        if pic.data_file:
            filex = settings.BASE_DIR + '\\media\\' + str(pic.data_file).replace('/', '\\')
            os.remove(filex)
        pic.delete()
        return HttpResponseRedirect('/view/pic2d')

def pic2d_model(request):
    if request.method == 'POST':
        data_type = request.POST['data_type']
        pic_type = request.POST['pic_type']
        mid = request.POST['mid']
        model = Model_train.objects.get(id=mid)
        with open(model.result_data, "rb") as get_myprofile:
            list_all = pickle.load(get_myprofile)
            stepallx, lossallx, lossallvx, stepall, lossall = list_all
        if data_type=='loss':
            data = pd.DataFrame({'step':stepallx, 'train':lossallx, 'val':lossallvx})
        elif data_type=='acc':
            data = pd.DataFrame({'step':stepall, 'acc':lossall})
        else:
            ch = model.value_index.split(',')
            temp = [ float(x) for x in ch ]
            Rp, Rmse, L1loss, r2 = temp
            data = pd.DataFrame({'Rp':Rp, 'Rmse':Rmse,'L1loss':L1loss,'r2':r2})

        
        user = request.user
        Viewpic.objects.create(class_name=pic_type, user=user, pic_name=pic_name)
        return HttpResponseRedirect('/view/pic2d')


@login_required()
def pic3d(request):
    if request.method == 'GET':
        user = request.user
        viewpics = user.viewpic3d_set.all()
        dataset={}
        label={}
        for pic in viewpics:
            if(pic.data_file):
                file_path = settings.BASE_DIR.replace('\\', '/') + '/media/' + str(pic.data_file)
                fclass = str(pic.data_file).split('.')[-1]
                data = []
                if fclass == 'csv':
                    try:
                        data = pd.read_csv(file_path, encoding='utf-8')
                    except:
                        try:
                            # data = pd.read_csv(path, encoding='latin-1')
                            data = pd.read_csv(file_path, encoding='gbk')
                        except:
                            return HttpResponseRedirect('/data/manage/')
                elif fclass in ['xls', 'xlsx']:
                    data = pd.read_excel(file_path)
                dataset['%s'%str(pic.id)] = data.fillna(0).values.T.tolist()
                label['%s' % str(pic.id)] = data.columns.values.tolist()
        return render(request, 'view/view3d.html',locals())
    if request.method == 'POST':
        pic_type = request.POST['name_type']
        pic_name = request.POST['pic_name']
        print('选择添加的图片信息为：'+pic_type+pic_name)
        user = request.user
        Viewpic3d.objects.create(class_name=pic_type,user=user,pic_name=pic_name)
        viewpics = user.viewpic3d_set.all()
        return render(request, 'view/view3d.html',locals())

def pic3d_del(request):
    if request.method == 'POST':
        del_id = request.POST['id']
        pic = Viewpic3d.objects.get(id=del_id)
        if pic.data_file:
            filex = settings.BASE_DIR + '\\media\\' + str(pic.data_file).replace('/', '\\')
            os.remove(filex)
        pic.delete()
        return HttpResponseRedirect('/view/pic3d')

def pic3d_change(request):
    if request.method=='GET':
        return render(request, 'view/view3d.html')
    if request.method=='POST':
        file = request.FILES.get('upfile')
        id = request.POST['id']
        pic = Viewpic3d.objects.get(id=id)
        if(pic.data_file):
            print('有了')
            filex = settings.BASE_DIR + '\\media\\' + str(pic.data_file).replace('/', '\\')
            os.remove(filex)
        pic.data_file = file
        pic.save()

        return HttpResponseRedirect('/view/pic3d/')

def pic3d_change_map(request):
    if request.method=='GET':
        return render(request, 'view/view3d.html')
    if request.method=='POST':
        map_name = request.POST['map_name']
        id = request.POST['id']
        pic = Viewpic3d.objects.get(id=id)
        pic.other_index = map_name
        pic.save()

        return HttpResponseRedirect('/view/pic3d/')


@login_required()
def pic(request):
    if request.method == 'GET':
        user = request.user
        experiments = user.model_train_set.all()
        maps = Map.objects.filter(user=user)[0].exs_id
        tmp = maps.split(',')
        max_pren = max(tmp)
        dataset = {}
        xaxis = {}
        pren = {}
        for ex in experiments:
            if (ex.pre):
                file_path = settings.BASE_DIR.replace('\\', '/') + '/media/' + str(ex.pre_result)
                with open(file_path, "rb") as get_myprofile:
                    data = pickle.load(get_myprofile)
                dataset['%s' % str(ex.id)] = data.iloc[:, 0].values.tolist()
                xaxis['%s' % str(ex.id)] = data.iloc[:, 1].values.tolist()
                pren['%s' % str(ex.id)] = data.columns.values[0]

                # model = Model_train.objects.get(model_name=ex.model_name)
                # if('list' in str(type(model.data_path))):
                #     data_path = model.data_path.split(',')[0]
                # else:
                #     data_path = model.data_path

        return render(request, 'view/viewmap.html', locals())
    if request.method == 'POST':
        user = request.user
        exs_id = request.POST.getlist('exs_id')
        print(exs_id)
        ids = ''
        for i in range(len(exs_id)):
            if(i==0):
               ids = ''+exs_id[i]
            else:
                ids = ids +','+exs_id[i]
        print(ids)
        tmp = Map.objects.filter(user=user)
        for data in tmp:
            data.exs_id = ids
            data.save()

        return render(request, 'view/viewmap.html', locals())
@login_required()
def model(request):
    if request.method == 'POST':
        user = request.user
        pic_class = request.POST['pic_type']
        mid = request.POST['mid']
        model = Model_train.objects.get(id=mid)
        model_class = model.model_class
        label = ['Rp', 'Rmse', 'L1loss', '1-R2']
        values = [label]
        if (pic_class == 'radar'):
            index=['max']
        else:
            index=[]
        models = Model_train.objects.filter(model_class=model_class,user=user)
        for model_o in models:
            index.append(model_o.model_name)
            tmp = model_o.value_index.split(',')
            data = [float(x) for x in tmp]
            data[3] = 1-data[3]
            values.append(data)
        path = str(settings.BASE_DIR).replace('\\', '/') + '/media/' + '/{}/'.format(user.id) + '/view/' + '/data1.csv'
        if(pic_class=='radar'):
            x = np.array(values[1:]).max(0).tolist()
            y = values[1:]
            y[:0] = [x]
            df = pd.DataFrame(y,columns=values[0],index=index).T
            df.to_csv(path,index_label='value')
        else:
            df = pd.DataFrame(values[1:], columns=values[0], index=index).T
            df.to_csv(path, index_label='name')
        with open(path, 'r') as f:
            print(type(f))
            file_content = Filexx(f)
            x = Viewpic.objects.filter(pic_name=pic_class + '_' + model.model_class)
            if (x):
                for m in x:
                    path = str(settings.BASE_DIR).replace('\\', '/') + '/media/' + str(m.data_file)
                    os.remove(path)
                    m.data_file.save('data.csv', file_content)
                    m.save()
            else:
                newpic = Viewpic.objects.create(class_name=pic_class, pic_name=pic_class + '_' + model.model_class,
                                                user=user)
                newpic.data_file.save('data.csv', file_content)
                newpic.save()

        return HttpResponseRedirect('/view/pic2d')

@login_required()
def file(request):
    if request.method == 'POST':
        user = request.user
        pic_class = request.POST['pic_type']
        pic_name = request.POST['pic_name']
        fid = request.POST['fid']
        if (pic_class == 'pie_pile'):
            X_axis = request.POST.getlist('X_axis')
            X_axis = [int(x) for x in X_axis]
        else:
            X_axis = int(request.POST['X_axis'])

        if(pic_class=='pie'):
            Y_axis = int(request.POST['Y_axis'])
        else:
            Y_axis = request.POST.getlist('Y_axis')
            Y_axis = [int(x) for x in Y_axis]

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

        path1 = str(settings.BASE_DIR).replace('\\', '/') + '/media/' + '/{}/'.format(user.id) + '/view'
        isExists = os.path.exists(path1)
        if not isExists:
            os.makedirs(path1)
        path = path1 +'//data2.csv'
        if (pic_class in ['line','bar','bar_pile']):
            Y_axis.insert(0,X_axis)
            newdata = data.iloc[:, Y_axis]
        elif(pic_class=='radar'):
            tmp = data.iloc[:, Y_axis].values
            tmp = np.max(tmp, axis=1)
            Y_axis.insert(0, X_axis)
            newdata = data.iloc[:, Y_axis]
            col_name = newdata.columns.tolist()
            col_name.insert(1, 'MAX')
            newdata = newdata.reindex(columns=col_name)
            newdata.iloc[:,1]=tmp
        elif(pic_class=='pie'):
            newdata = data.iloc[:, [X_axis,Y_axis]]
        elif(pic_class=='pie_pile'):
            newdata = data.iloc[:, [X_axis[0],Y_axis[0],X_axis[1],Y_axis[1]]]
        elif(pic_class=='box' or pic_class=='3d_scatter'):
            newdata = data.iloc[:, Y_axis]
            print(newdata.head())

        newdata.to_csv(path, index=False)
        with open(path, 'r',encoding='utf-8') as f:
            file_content = Filexx(f)
            if(pic_name):
                pic_namex = pic_name
            else:
                pic_namex = pic_class + '_' + str(file.name).split('.')[0]

            if(pic_class=='3d_scatter'):
                x = Viewpic3d.objects.filter(pic_name=pic_namex,user=user)
                if (x):
                    for m in x:
                        path = str(settings.BASE_DIR).replace('\\', '/') + '/media/' + str(m.data_file)
                        os.remove(path)
                        m.data_file.save('data2.csv', file_content)
                        m.save()
                else:
                    newpic = Viewpic3d.objects.create(class_name=pic_class, pic_name=pic_namex,
                                                    user=user)
                    newpic.data_file.save('data2.csv', file_content)
                    newpic.save()
            else:
                x = Viewpic.objects.filter(pic_name=pic_namex,user=user)
                if (x):
                    for m in x:
                        path = str(settings.BASE_DIR).replace('\\', '/') + '/media/' + str(m.data_file)
                        os.remove(path)
                        m.data_file.save('data2.csv', file_content)
                        m.save()
                else:
                    newpic = Viewpic.objects.create(class_name=pic_class, pic_name=pic_namex,
                                                    user=user)
                    newpic.data_file.save('data2.csv', file_content)
                    newpic.save()

        return HttpResponse('yes')
