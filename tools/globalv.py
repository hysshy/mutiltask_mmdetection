#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import threading
import queue
import pynvml
import socket
import configparser
import os
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
cf = configparser.ConfigParser()
cf.read(str(rootPath)+"/config.ini")
#program_name = cf.get("Server-Info", "program_name")
program_name = rootPath.split('/')[2]

from flask import Flask
#cuda_num = cf.get("Server-Info", "CUDA_NUM")
pynvml.nvmlInit()
cuda_num = pynvml.nvmlDeviceGetCount()
#cuda_num = 1
scene_mark = cf.get("Server-Info", "scene_mark")
#0:结构化1:目标检测2:人脸结构化3:人体结构化4:车辆结构化5:简化版
if scene_mark == '0':
    scene = 'scenestarter'
elif scene_mark == '1':
    scene = 'scenestarter_detect'
elif scene_mark == '2':
    scene = 'scenestarter_face'
elif scene_mark == '3':
    scene = 'scenestarter_person'
elif scene_mark == '4':
    scene = 'scenestarter_car'
else:
    scene = 'scenestarter_simple'

server_id = cf.get("Server-Info", "server_id")
host_ip = cf.get("Server-Info", "server_ip")
#算法服务器列表
hostlist = []
hostnames = cf.options("Node-IP")
for i, name in enumerate(hostnames):
    hostlist.append(cf.get("Node-IP", name))
#database
POOL = None
sql_ip = cf.get("Sql-Database", "host")
sql_usr = cf.get("Sql-Database", "user")
sql_pwd = cf.get("Sql-Database", "password")
sql_port = int(cf.get("Sql-Database", "port"))
sql_database = cf.get("Sql-Database", "database")
# mq信息
mq_server = cf.get("MQ-Server", "mq_server")
mq_pro = None
mq_queue = queue.Queue()
#torch锁
max_gpu = int(cuda_num)
torch_lock_gpu = []
#各算法计数表
facecounter = []
mmcounter = []
recocounter = []
carcounter = []
#根据gpu数量动态建锁及创建均衡计数器
for i in range(max_gpu):
    torch_lock_gpu.append(threading.Lock())
    facecounter.append(0)
    mmcounter.append(0)
    recocounter.append(0)
    carcounter.append(0)
#已启动算法模型列表
detection_list = {}
#数据锁
globallock = threading.Lock()
#使用qnet的scene_mark
qnet_list = ['0', '2', '3']
qnet = None
#设备信息
points = {}
points_car = {}
graph = []
status = {}
delete = {}
rtsps = {}
deviceTypes = {}
cameraInfo = {}
cap = {}
#人脸信息
faces = {}
faces_cap = {}#与人脸匹配成功的抓拍模板
featuresList = []
featuresArray = []
faceID = []
tracker_threshold = cf.get("Recognition-Args", "tracker_threshold")
processing_threshold = cf.get("Recognition-Args", "processing_threshold")
#多对1版本使用参数
server_version = cf.get("Server-Info", "server_version")
mq_list = []
if server_version == '1':
    faceID = {}
    mq_names = cf.options("MQ-Server")
    for i, name in enumerate(mq_names):
        mq_list.append(cf.get("MQ-Server", name))
sql_list = mq_list
POOL_list = []
mq_pro_list = []
mq_queue_list = []
for i in mq_list:
    mq_queue_list.append(queue.Queue())

#公安版本使用
http_server = cf.get("Http-Server", "http_server")
token = ''
cap_unv = {}
cap_hk = {}
feat_dict = {}
brand_dict = {}
device_name = {}
SDKPATH = "/chase/{}/common/SDKFile/".format(program_name)
#公安相关设备信息,格式:0 01203001 01203 192.168.137.5(设备类型0海康,1宇视,2rtsp 设备code 小区code ip)
#cameraFile = '/chase/{}/police/camera.txt'.format(program_name)
cameraFile = '/home/chase/Desktop/camera.txt_back5'
rtsp_format = 'rtsp://{}:{}@{}'
rtsp_format_unv = 'rtsp://{}:{}@{}/media'
camera_passwd = 'admin123'
app = Flask('flask')
face_queue = queue.Queue()
car_queue = queue.Queue()
alarm_face_queue = queue.Queue()
alarm_car_queue = queue.Queue()
#弃用参数
lock1 = threading.Lock()
lock2 = threading.Lock()
socket_send_queue = queue.Queue()
facequeue = queue.Queue()
facequeuereturn = queue.Queue()