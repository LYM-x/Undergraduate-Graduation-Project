from django.shortcuts import render
from django.contrib.auth.decorators import login_required
from index.models import Viewleft,Viewright
from view.models import Viewpic
from django.conf import settings
import pandas as pd
from django.http import HttpResponseRedirect,HttpResponse
from django.views.decorators.csrf import csrf_exempt

# Create your views here.
@login_required
def index(request):
    user = request.user
    viewleft_tmp = user.viewleft_set.all()
    datasetl = {}
    labell = {}
    classl={}
    flagl={}
    for picx in viewleft_tmp:
        if (picx.viewpic!='-1'):
            pic_id = int(picx.viewpic)
            pic = Viewpic.objects.get(id=pic_id)
            classl['%s' % str(picx.id)] = pic.class_name
            if(pic.data_file):
                file_path = settings.BASE_DIR.replace('\\', '/') + '/media/' + str(pic.data_file)
                fclass = str(pic.data_file).split('.')[-1]
                data = []
                if fclass == 'csv':
                    try:
                        data = pd.read_csv(file_path, encoding='utf-8')
                    except:
                        try:
                            data = pd.read_csv(file_path, encoding='gbk')
                        except:
                            return HttpResponseRedirect('/')
                elif fclass in ['xls', 'xlsx']:
                    data = pd.read_excel(file_path)
                datasetl['%s' % str(picx.id)] = data.fillna(0).values.T.tolist()
                labell['%s' % str(picx.id)] = data.columns.values.tolist()
                flagl['%s' % str(picx.id)] = 1
            else:
                flagl['%s' % str(picx.id)] = 0
        else:
            flagl['%s' % str(picx.id)] = 0


    viewright_tmp = user.viewright_set.all()
    datasetr = {}
    labelr = {}
    classr={}
    flagr={}
    for picx in viewright_tmp:
        if (picx.viewpic != '-1'):
            pic_id = int(picx.viewpic)
            pic = Viewpic.objects.get(id=pic_id)
            classr['%s' % str(picx.id)] = pic.class_name
            if (pic.data_file):
                file_path = settings.BASE_DIR.replace('\\', '/') + '/media/' + str(pic.data_file)
                fclass = str(pic.data_file).split('.')[-1]
                data = []
                if fclass == 'csv':
                    try:
                        data = pd.read_csv(file_path, encoding='utf-8')
                    except:
                        try:
                            data = pd.read_csv(file_path, encoding='gbk')
                        except:
                            return HttpResponseRedirect('/')
                elif fclass in ['xls', 'xlsx']:
                    data = pd.read_excel(file_path)
                datasetr['%s' % str(picx.id)] = data.fillna(0).values.T.tolist()
                labelr['%s' % str(picx.id)] = data.columns.values.tolist()
                flagr['%s' % str(picx.id)] = 1
            else:
                flagr['%s' % str(picx.id)] = 0
        else:
            flagr['%s' % str(picx.id)] = 0

    pics_all = user.viewpic_set.all()

    return render(request, 'index/index.html',locals())

@csrf_exempt
@login_required
def pic(request):
    if request.method=='POST':
        # name_type: name_type,
        # pic_name: pic_name,
        # info: info
        user = request.user
        pic_name = request.POST['pic_name']
        pic_id = request.POST['pic_id']
        info = request.POST['info']
        index = info.find('select') + 6
        id = int(info[index:])

        if('left' in info):
            pic = Viewleft.objects.get(id=id,user=user)
        else:
            pic = Viewright.objects.get(id=id,user=user)
        pic.viewpic = int(pic_id)
        pic.vic_name = pic_name
        pic.save()

        return HttpResponseRedirect('/')


