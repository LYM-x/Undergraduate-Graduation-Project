# -*- coding:utf-8 -*-
import os

from django.shortcuts import render
from django.contrib.auth import authenticate,login as loginx,logout
from django.http import HttpResponseRedirect
# from user.models import Last_url
from django.contrib import messages
# from django.contrib.auth.models import User
from django.contrib.auth.decorators import login_required
from user.models import NewUser
from django.conf import settings
from data.models import File
import shutil
from index.models import Viewleft,Viewright
from view.models import Map


def login(request):
    if request.method=='GET':
        last_url = request.GET.get('next')
        logoutx = request.GET.get('logout')
        if logoutx=='true':
            logout(request)
        return render(request, 'user/login.html')
    if request.method=='POST':
        username = request.POST['username']
        password = request.POST['password']
        user = authenticate(username=username,password=password)
        if not user:
            messages.info(request, '用户名或密码不正确')
            return HttpResponseRedirect('/user/login')
        else:#通过
            loginx(request,user)
            return HttpResponseRedirect('/index')

def register(request):
    if request.method=='GET':
        return render(request, 'user/register.html')
    if request.method=='POST':
        email = request.POST['email']
        username = request.POST.get('username','')
        password1 = request.POST['password1']
        password2 = request.POST['password2']
        if password2 != password1:
            messages.info(request, '两次密码输入不相等')
            return HttpResponseRedirect('/user/register')
        if not username:
            messages.info(request, '请输入用户名')
            return HttpResponseRedirect('/user/register')
        u = NewUser.objects.filter(username=username)
        if u.count() != 0:
            messages.info(request, '已有相同用户名')
            return HttpResponseRedirect('/user/register')
        user = NewUser.objects.create_user(username=username,email=email,password=password1)
        Viewleft.objects.create(vic_name='pic1', user=user)
        Viewleft.objects.create(vic_name='pic2', user=user)
        Viewright.objects.create(vic_name='pic3', user=user)
        Viewright.objects.create(vic_name='pic4', user=user)
        Map.objects.create(user=user)
        messages.success(request, '创建成功！')
        return HttpResponseRedirect('/user/login')

@login_required
def setting(request):
    #简介info,头像myimg,用户名username,选择class（student，teacher）,邮箱email
    if request.method=='POST':
        info = request.POST['info']
        username = request.POST['username']
        # classx = request.POST['class']
        email = request.POST['email']
        myimg = request.FILES.get('myimg','')
        u = NewUser.objects.get(username=username)
        u.info = info
        u.email = email
        if myimg:
            if u.img != '' and u.img:
                old_pic = settings.BASE_DIR + '\\media\\' + str(u.img).replace('/', '\\')
                os.remove(old_pic)
            u.img = myimg
        u.save()
        return HttpResponseRedirect('/index/')
    if request.method=='GET':
        user = request.user
        return render(request,'user/setting.html',locals())


def user(request):
    return HttpResponseRedirect('/user/setting')

def superuser(request):
    # 简介info,头像myimg,用户名username,选择class（student，teacher）,邮箱email
    if request.method=='GET':
        user = request.user
        if(user.is_superuser):
            users = NewUser.objects.all()
            return render(request, 'user/superuser.html',locals())
        else:
            return HttpResponseRedirect('/index/')

def change_user(request):
    if request.method=='POST':
        uid = request.POST['id']
        name_type = request.POST['name_type']
        user=NewUser.objects.get(id=uid)
        print(name_type)
        if name_type=='superuser':
            user.is_superuser = True
        else:
            user.is_superuser=False
        user.save()
        print(user.is_superuser)
        return HttpResponseRedirect('/user/superuser')
    if request.method=='GET':
        return HttpResponseRedirect('/user/superuser')

def del_userno(request):
    if request.method == 'POST':
        uid = request.POST['id']
        user = NewUser.objects.get(id=uid)
        x = user.is_active
        if(x):
            user.is_active = False
        else:
            user.is_active = True
        user.save()
        return HttpResponseRedirect('/user/superuser')
    if request.method=='GET':
        return HttpResponseRedirect('/user/superuser')

def del_useryes(request):
    if request.method == 'POST':
        uid = request.POST['id']
        user = NewUser.objects.get(id=uid)
        user.delete()
        data_path = settings.BASE_DIR + '\\media\\'+str(uid)
        try:
            shutil.rmtree(data_path)
        except:
            pass
        return HttpResponseRedirect('/user/superuser')
    if request.method=='GET':
        return HttpResponseRedirect('/user/superuser')


def add_user(request):
    if request.method=='POST':
        name_type = request.POST['name_type']
        data = request.POST['data']
        datalist = data.split(',')

        u = NewUser.objects.filter(username=datalist[0])
        if u.count() != 0:
            messages.info(request, '已有相同用户名')
            print('已有相同用户名')
            return HttpResponseRedirect('/user/superuser')

        if name_type=='superuser':
            user = NewUser.objects.create_user(username=datalist[0], email=datalist[1], info=datalist[2],is_superuser=True,password='12345')
        else:
            user = NewUser.objects.create_user(username=datalist[0], email=datalist[1], info=datalist[2], is_superuser=False,password='12345')
        Viewleft.objects.create(vic_name='pic1',user=user)
        Viewleft.objects.create(vic_name='pic2', user=user)
        Viewright.objects.create(vic_name='pic3', user=user)
        Viewright.objects.create(vic_name='pic4', user=user)
        Map.objects.create(user=user)

        print(datalist)
        return HttpResponseRedirect('/user/superuser')
    if request.method=='GET':
        return HttpResponseRedirect('/user/superuser')


