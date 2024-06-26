## 1.项目简介
目前市面上环境数据分析平台较少，大多数平台的主要功能为环境数据的获取、收集和可视化展示，对于数据的分析力度不够 ，导致数据管理、可视化、分析三个功能不能很好的集中在一个应用上。这类平台大多存在可视化模式单一、数据分析模型智能化低、平台管理系统不够完善等问题。因此，设计一个集成数据管理、智能算法分析、可视化于一体的智能化环境数据分析平台是十分必要的。

平台使用传统的Django前后端响应方式，通过ORM框架与Mysql数据库连接，网站内嵌基于Pytorch构建的机器学习模型对数据进行预测分析。模型主要包括LSTM、GCN、Transformer等。

## 2.基础包版本
Django=3.2，Python=3.7.1，Pytorch=1.7.1+cu110

其余包在requirements.txt文件中有写

## 3.使用教程

网站运行的代码环境：requirements.txt文件

修改数据库信息：在根文件目录/graduation/settings.py文件中修改

（可选，创建管理员账号）在命令行的根文件目录下输入：python manage.py createsuperuser

运行python manage.py makemigrations

运行python manage.py migrate

运行平台python manage.py runserver

进入平台，首页可注册普通账号

## 4.使用示例
首先进入网站，登陆页面如下：

![登陆网站](./image/登陆.jpg)

进入主页，可以看见部分可视化结果：

![主页](./image/主页.jpg)

上传查看数据：

![上传数据](./image/上传数据.jpg)

![查看数据](./image/查看数据.jpg)

选择机器学习模型进行参数设置，然后训练，预测：

![模型管理](./image/模型管理.jpg)

![模型训练](./image/模型训练.jpg)

![训练结果](./image/训练结果.jpg)

![训练结果2](./image/训练结果2.jpg)

![预测可视化](./image/预测可视化.jpg)

还可以对上传数据或者预测结果进行更丰富的可视化：

![可视化1](./image/可视化1.jpg)

![地图可视化](./image/地图可视化.jpg)

![地图可视化2](./image/地图可视化2.jpg)