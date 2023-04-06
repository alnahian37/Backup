# %%
import json
import time
import pickle
import kinectpy as kpy
import ast
from kinectpy.k4a._k4atypes import K4A_CALIBRATION_TYPE_DEPTH
import gc
import statistics as st



import scipy.io
import numpy as np
import csv
import copy

from scipy.optimize import linear_sum_assignment

import os
import csv

import pickle
import json
import ast

import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from math import inf

import sys
sys.path.insert(1, './../')

from timeit import repeat

import kinectpy as kpy
from kinectpy.k4abt._k4abtTypes import K4ABT_SEGMENT_PAIRS
from kinectpy.k4abt import _k4abt
from kinectpy.k4a import _k4a
from kinectpy.k4a._k4atypes import K4A_CALIBRATION_TYPE_DEPTH, K4A_CALIBRATION_TYPE_COLOR
import logging
from matplotlib import animation
import time
from datetime import datetime
import scipy.io
import time
import socket
from _thread import *

#from functions import *





all_indexes=range(60)
#Table
x1=np.linspace(2090,3115,3)
z1=np.linspace(0,740,3)
y1=np.linspace(1570,8850,3)
x1=np.array(x1).reshape(len(x1),1)
y1=np.array(y1).reshape(len(y1),1)
z1=np.array(z1).reshape(len(z1),1)
x,y=np.meshgrid(x1,y1)
z=737*np.ones((len(x1),len(y1)))
z_floor=np.zeros((len(x1),len(y1)))
xx,zz=np.meshgrid(x1,z1)
yy=1570*np.ones((len(x1),len(z1)))
yy2=8850*np.ones((len(x1),len(z1)))
x2=np.linspace(0,4500,3)
z2=np.linspace(0,3000,3)
y2=np.linspace(0,10000,3)
x2=np.array(x2).reshape(len(x2),1)
y2=np.array(y2).reshape(len(y2),1)
z2=np.array(z2).reshape(len(z2),1)
x_floor,y_floor=np.meshgrid(x2,y2)
z_floor=np.zeros((len(x2),len(y2)))
xxx,zzz=np.meshgrid(x2,z2)
yyy=np.zeros((len(x2),len(z2)))
y3,z3=np.meshgrid(y2,z2)
x3=4500*np.ones((len(y2),len(z2)))
y4,z4=np.meshgrid(y2,z2)
x4=np.zeros((len(y2),len(z2)))
y5=10000*np.ones((len(x2),len(z2)))
z2=np.linspace(0,2000,3)
y2=np.linspace(8200,9000,3)
y2=np.array(y2).reshape(len(y2),1)
z2=np.array(z2).reshape(len(z2),1)
y6,z6=np.meshgrid(y2,z2)
x6=np.zeros((len(y2),len(z2)))

def initialize_plots():
    fig = plt.figure(figsize=(10, 10),dpi=200)

    # Create axis
    axes = fig.add_subplot(111, projection='3d')  
    #axes.set_aspect('auto') 
    (x_min, x_max), (y_min, y_max), (z_min, z_max) = ((0, 4500), (0,10400), (0, 3000)) 
   
    #Door
    axes.plot_surface(x6, y6, z6, color='k',alpha=1,edgecolor=None)

    #Wall
    axes.plot_surface(xxx,y5,zzz,alpha=0.5,color='green',edgecolor=None)
    axes.plot_surface(x4,y4,z4,alpha=0.5,color='blue',edgecolor=None)
    axes.plot_surface(x3,y3,z3,alpha=0.5,color='blue',edgecolor=None)
    axes.plot_surface(xxx,yyy,zzz,alpha=0.5,color='green',edgecolor=None)    
    #Table
    axes.plot_surface(xx,yy,zz,color='magenta',alpha=0.5,edgecolor=None)
    axes.plot_surface(xx,yy2,zz,color='magenta',alpha=0.5,edgecolor=None)
    axes.plot_surface(x,y,z,color='magenta',alpha=0.3,edgecolor=None)
    axes.plot_surface(x_floor,y_floor,z_floor,color='blue',alpha=0.2,edgecolor=None)    
    
    axes.set_xlabel('X')
    axes.set_ylabel('Y')
    axes.set_zlabel('Z')
    axes.set_xlim(left=x_min, right=x_max)
    axes.set_ylim(bottom=y_min, top=y_max)
    axes.set_zlim(bottom=z_min, top=z_max)
    axes.axis('off')
    #axes.grid(True)
    #axes.set_axis_off()
    axes.view_init(azim = -115, elev = 35)
    axes.w_zaxis.set_pane_color((0, 1, 0, 0.2))
    axes.get_proj = lambda: np.dot(Axes3D.get_proj(axes), np.diag([0.5, 1, 0.3, 1]))
    plots = {index: axes.plot([0], [0], [0], 'bo-', markersize = 0)[0] for index in all_indexes}

    return fig, axes, plots





def activity_detection(cur_frame):    
    if (sum(cur_frame[:,3])==0):
        #print('No activity detected')
        activ=np.zeros((cur_frame.shape[0],1))
        activ[:]=4 #no activity when no person is present and only zeros are sent here
        return activ
    
    poses=(cur_frame[:,3]>1300).astype(int).reshape(cur_frame.shape[0],1)
    
    activ=copy.deepcopy(poses)    

    ##Activity 1: Random
    if sum(poses)==cur_frame.shape[0]: #All standing        
        activ[:]=1

    ##Activity 2: Table
    elif sum(poses)==0: #All sitting
        activ[:]=2
    
    ##Activity 3: Presentation or Activity 2: Table or Activity 1: Random
    elif sum(poses)>0 and sum(poses)<cur_frame.shape[0]: #Some standing, some sitting
        y_values=np.zeros((cur_frame.shape[0],1))
        x_values=np.zeros((cur_frame.shape[0],1))
        z_values=np.zeros((cur_frame.shape[0],1))
        flag_presentation=0
        
        
        if sum(poses)<=0.5*cur_frame.shape[0]: #Most people sitting
            for j in range(cur_frame.shape[0]):
                y_values[j]=cur_frame[j,2]
                x_values[j]=cur_frame[j,1]
                z_values[j]=cur_frame[j,3]

            for j in range(cur_frame.shape[0]):
                if y_values[j]>8650 and x_values[j]>1850 and x_values[j]<4200 and z_values[j]>1300: # Presenter is standing, most people sitting
                    flag_presentation=+1
        if flag_presentation>0:
            ######print("PRESENTATION")  ##Activity 3: Presentation
            activ[:]=3
            #flag_presentation=0
        else:
            ######print("Random Standing")
            if (sum(poses)>0.5*cur_frame.shape[0]): #More people standing
                activ[:]=1 #Random
            else:
                activ[:]=2 #Table for majority sitting
    ######print("activ ",activ)
    activ=np.reshape(activ,(cur_frame.shape[0],1))
    global temporal_activ

    temporal_activ[:len(temporal_activ)-1]=temporal_activ[1:]
    temporal_activ[-1]=activ[0,0]
    activ[:]=int(st.mode(temporal_activ)) #mode of the last n seconds. so, activity needs to be stable for n/2 seconds to be valid
    
    return activ

def convert_to_unity(cur_frame):
    unity_data=np.zeros((cur_frame.shape[0],11))

    unity_data[:,0]=cur_frame[:,0]
    unity_data[:,1]=np.round(cur_frame[:,1]/1000,2)-5.1*(cur_frame[:,1]>0)-(cur_frame[:,1]==0)*10  #unity x
    unity_data[:,2]=((cur_frame[:,3]>1300).astype(float)*0.5+1)*(cur_frame[:,3]>0).astype(int) #unity y
    unity_data[:,3]=np.round(cur_frame[:,2]/1000,2)-10.8*(cur_frame[:,2]>0) #unity z
    unity_data[:,4]=cur_frame[:,11]
    unity_data[:,5]=cur_frame[:,13]
    unity_data[:,6]=cur_frame[:,12]
    unity_data[:,7]=cur_frame[:,15].astype(int) #activity
    unity_data[:,8]=(cur_frame[:,3]>1300).astype(int)+1 #pose
    unity_data[:,9]=cur_frame[:,16].astype(int) #sTAtionary
    unity_data[:,10]=0 #glare
    
    return unity_data
def send_to_unity(unity_data):
    global temporal_activ
    global prev_frame_activ
    send_string=''

    if unity_data[0][7]==temporal_activ[-2] or unity_data[0][7]==prev_frame_activ:
        send_command=False
    else:
        send_command=True
        prev_frame_activ=unity_data[0][7]
        if unity_data[0][7]==1:
            send_string="swap ies profiles to SetForget BVLS"
        elif unity_data[0][7]==2:
            send_string="swap ies profiles to Table BVLS"
        elif unity_data[0][7]==3:
            send_string="swap ies profiles to Blackboard BVLS"
        else:
            send_string="swap ies profiles to Dim"

        all_occu=''
        all_gaz_dir=''
        for i in range(unity_data.shape[0]):
            occu=str(tuple(np.round(unity_data[i][1:4],2)))
            gaz_dir=str(tuple(np.round(unity_data[i][4:7],2)))
            all_occu=all_occu+occu
            all_gaz_dir=all_gaz_dir+gaz_dir
        send_string=send_string+'$'+'eco'+'$'+'1 2'+'$'+all_occu+'$'+all_gaz_dir



    #data=unity_data[:,1:].reshape(-1)
    #send_string='$'.join(map(str, data))
    #send_string='Avatar$'+send_string
    return send_command,send_string

def get_camera_data(num_bodies4,bodyFrame4,W1,b1):
    cam4s=[]
    for body_id in range(num_bodies4):

        body3d4 = bodyFrame4.get_body3d(body_id, dest_camera = K4A_CALIBRATION_TYPE_DEPTH)
    
        all_joints4=[]
        for joint in body3d4.joints:

            join=joint.get_coordinates()
            #if join[2]>far_threshold:
                #far_flag=1
                
            join=join@W1.T+b1
            join.reshape(3,1)
            con=np.array([joint.confidence_level]).reshape(1,1)
            join_con=np.concatenate((join,con),axis=1)
            all_joints4.append(join_con)
        all_joints4=np.array(all_joints4).reshape(32,4)
        nose_pos=all_joints4[27,0:3]
        tmp=copy.deepcopy(nose_pos)
        nose_pos[1]=tmp[2]
        nose_pos[2]=tmp[1]
        nose_pos[1]=room_length-nose_pos[1]
        
        confidence=np.mean(all_joints4[:,3])
        ear_left=all_joints4[29,0:3]
        tmp=copy.deepcopy(ear_left)
        ear_left[1]=tmp[2]
        ear_left[2]=tmp[1]
        ear_left[1]=room_length-ear_left[1]

        ear_right=all_joints4[31,0:3]
        tmp=copy.deepcopy(ear_right)
        ear_right[1]=tmp[2]
        ear_right[2]=tmp[1]
        ear_right[1]=room_length-ear_right[1]

        
        p1=(ear_left+ear_right)/2
        v=nose_pos-p1 
        p2=500*v / np.linalg.norm(v)+p1 ## 10*front_point-9*back_point
        position_vector=p2-p1
        
        #if nose_pos[1]>z_threshold: # y distance from origin
        
        if 0==0:
            cam4=np.concatenate((np.array([frame]),nose_pos,np.array([confidence]),p1,p2,position_vector))
            cam4s.append(cam4)
            #necks4.append(neck)
        else:
            #far_flag=0
            num_bodies4-=1
        
    if num_bodies4>0:
        cam4s=np.array(cam4s)
        del_idx=np.where(cam4s[:,1]<0)[0]
        print("** index to delete= ",del_idx,"")
    #del_idx=np.where(prev_frame_data[:,-3]==unique_ids[ind])[0]
        cam4s=np.delete(cam4s,del_idx,0)
    return cam4s


def showing_plot(plot_data):
    poin1s=[]
    poin2s=[]
    poin3s=[]
    poin4s=[]
    for i in range(plot_data.shape[0]):
        x1=plot_data[i][5]
        y1=plot_data[i][6]
        z1=plot_data[i][7]
        x2=plot_data[i][8]
        y2=plot_data[i][9]
        z2=plot_data[i][10]

        poin1=np.array([x1,y1,z1])
        poin2=np.array([x2,y2,z2])
        poin3=np.array([x1,y1,0])
        poin1s.append(poin1)
        poin2s.append(poin2)
        poin3s.append(poin3)
        poin4s.append(plot_data[i][16])
    for l in range(len(all_indexes)):
        plots[l].set_xdata(np.asarray(0))
        plots[l].set_ydata(np.asarray(0))
        plots[l].set_3d_properties(np.asarray(0))
    
    poin1s=np.array(poin1s)
    poin2s=np.array(poin2s)
    poin3s=np.array(poin3s)
    poin4s=np.array(poin4s)
    for i in range(poin1s.shape[0]):
        if i==0:
            color = ((1, 0, 0))
        elif i==1:
            color = ((0, 1, 0))
        elif i==2:
            color = ((0, 0, 1))
        elif i==3:
            color = ((1, 1, 0))
        elif i==4:
            color = ((1, 0, 1))
        elif i==5:
            color = ((0, 1, 1))
        elif i==6:
            color = ((0.75, 0.75, 0.75))
        elif i==7:
            color = ((0.5, 0, 0))
        elif i==8:
            color = ((0, 0.5, 0))
        elif i==9:
            color = ((0, 0, 0.5))
        elif i==10:
            color = ((0.5, 0.5, 0))
        elif i==11:
            color = ((0.5, 0, 0.5))
        elif i==12:
            color = ((0, 0.5, 0.5))
        elif i==13:
            color = ((0.5, 0.5, 0.5))

        p1=poin1s[i]
        p2=poin2s[i]
        p3=poin3s[i]
        p4=poin4s[i]
        plots[i].set_xdata(np.asarray((p2[0],p1[0])))
        plots[i].set_ydata(np.asarray((p2[1],p1[1])))
        plots[i].set_3d_properties(np.asarray((p2[2],p1[2])))
        plots[i].set_markersize(2)
        plots[i].set_color(color)
        plots[i+25].set_xdata(np.asarray((p1[0])))
        plots[i+25].set_ydata(np.asarray((p1[1])))
        plots[i+25].set_3d_properties(np.asarray((p1[2])))
    
        if p1[2]>1300 or p1[2]==0:
            if p4==1:
                plots[i+25].set_color((1,0,0))
            else:

                plots[i+25].set_color((0,0,0))
        elif p1[2]<=1300:
            if p4==1:
                plots[i+25].set_color((1,0,0))
            else:
                plots[i+25].set_color((0.1,0.9,0.1))
        #plots[i+15].set_color((0,0,0))
        plots[i+25].set_markersize(5)
        plots[i+40].set_xdata(np.asarray((p3[0],p1[0])))
        plots[i+40].set_ydata(np.asarray((p3[1],p1[1])))
        plots[i+40].set_3d_properties(np.asarray((p3[2],p1[2])))
        plots[i+40].set_markersize(2)
        if p1[2]>1300 or p1[2]==0:
            plots[i+40].set_color((0,0,0))
        elif p1[2]<=1300:# and p1[2]!=0:
            plots[i+40].set_color((0.1,0.9,0.1))
    fig.canvas.draw()
    plt.pause(0.001)
    #txt="\n\nFrame {}\n".format(frame) #default
    txt=''
    txt2=''
    if plot_data[0][-2]==1:
        txt2='default'
    elif plot_data[0][-2]==2:
        txt2='table'
    elif plot_data[0][-2]==3:
        txt2='presentations'
    elif plot_data[0][-2]==4:
        txt2='no activity'
    txt=txt+txt2         
    axes.title.set_text(txt)
    axes.title.set_fontsize(10)




#Load the calibrated parameters
p4=scipy.io.loadmat('new_cam4_params_umeyama.mat')['name']
W4=p4[0:3,:]
b4=p4[3:,:]

p3=scipy.io.loadmat('new_cam3_params_umeyama.mat')['name']
W3=p3[0:3,:]
b3=p3[3:,:]

p2=scipy.io.loadmat('new_cam2_params_umeyama.mat')['name']
W2=p2[0:3,:]
b2=p2[3:,:]

p1=scipy.io.loadmat('new_cam1_params_umeyama.mat')['name']
W1=p1[0:3,:]
b1=p1[3:,:]


#END of Function Declarations above

print("enter 1 for online and 2 for offline")
on_off=int(input())

# Initialize the library, if the library is not found, add the library path as argument
kpy.initialize_libraries(track_body=True)

if on_off==1:

    #Start the devices
    device_config = kpy.default_configuration	
    #device_config.color_resolution = pykinect.K4A_COLOR_RESOLUTION_OF
    device_config.color_resolution=0 
    print("DDD", device_config)
    print("HHH",device_config.color_resolution)
    #device_config.depth_mode = pykinect.K4A_DEPTH_MODE_WFOV_2X2BINNED

    device_config.depth_mode = 2
    device_config.wired_sync_mode=0
    print("GGG",device_config.depth_mode)
    playback4 = kpy.start_device(device_index=0,config=device_config)

    device_config1 = kpy.default_configuration
    device_config1.color_resolution = 0 
    device_config1.depth_mode=2
    device_config1.wired_sync_mode=0
    playback3 = kpy.start_device(device_index=1,config=device_config1)


    ###
    device_config2 = kpy.default_configuration	
    device_config2.color_resolution=0 #2160p
    device_config2.depth_mode = 2
    device_config2.wired_sync_mode=0
    print("GGG",device_config2.depth_mode)
    playback2 = kpy.start_device(device_index=2,config=device_config2)


    device_config3 = kpy.default_configuration
    device_config3.color_resolution = 0 #2160p
    device_config3.depth_mode=2
    device_config3.wired_sync_mode=0
    playback1 = kpy.start_device(device_index=3,config=device_config3)



    playback_config1 = device_config3
    playback_calibration1 = playback1.calibration
    playback_config2 = device_config2
    playback_calibration2 = playback2.calibration
    playback_config3 = device_config1 #playback3.get_record_configuration()
    playback_calibration3 = playback3.calibration
    playback_config4 = device_config #playback4.get_record_configuration()
    playback_calibration4 = playback4.calibration

elif on_off==2:
    video1_filename=r"D:\Kinect Tools\Nahian\Code Backup Mar 29\All_code_f\Neo kinect calibration\New Calibration\Jan 19\videos_1\output1_4.mkv"
    video2_filename=r"D:\Kinect Tools\Nahian\Code Backup Mar 29\All_code_f\Neo kinect calibration\New Calibration\Jan 19\videos_1\output1_3.mkv"
    video3_filename=r"D:\Kinect Tools\Nahian\Code Backup Mar 29\All_code_f\Neo kinect calibration\New Calibration\Jan 19\videos_1\output1_2.mkv"
    video4_filename=r"D:\Kinect Tools\Nahian\Code Backup Mar 29\All_code_f\Neo kinect calibration\New Calibration\Jan 19\videos_1\output1_1.mkv"
    playback1 = kpy.start_playback(video1_filename)
    playback2 = kpy.start_playback(video2_filename)
    playback3 = kpy.start_playback(video3_filename)
    playback4 = kpy.start_playback(video4_filename)

    playback_config1 = playback1.get_record_configuration()
    playback_calibration1 = playback1.get_calibration()
    playback_config2 = playback2.get_record_configuration()
    playback_calibration2 = playback2.get_calibration()
    playback_config3 = playback3.get_record_configuration()
    playback_calibration3 = playback3.get_calibration()
    playback_config4 = playback4.get_record_configuration()
    playback_calibration4 = playback4.get_calibration()




bodyTracker1 = kpy.start_body_tracker(calibration=playback_calibration1)
bodyTracker2 = kpy.start_body_tracker(calibration=playback_calibration2)
bodyTracker3 = kpy.start_body_tracker(calibration=playback3.calibration)
bodyTracker4 = kpy.start_body_tracker(calibration=playback4.calibration)



fig, axes, plots = initialize_plots()    

plt.ion()



frame=0
flag_enter=0
flag_exit=0
this_frame_exit=0



prev_frame_enter=1
prev_frame_exit=1

data_col=17
data_col_less=14

prev_frame_data=np.array([]).reshape(-1,data_col_less)
#prev_neck_data=np.array([]).reshape(-1,3)
all_data=np.array([]).reshape(-1,data_col)

flag2=0

smooth_window=0


#FOr Person ID

latest_person_id=0
person_id_list=np.array([]).reshape(-1,1)
far_threshold=6500
same_threshold=500

entering_frame=[]

room_length=11000






print("enter number")
person_present=int(input())
print("where to show? 1 for python and 0 for unity and 2 for none")
plot_flag=int(input())
send_to_unity_flag=not plot_flag

if  send_to_unity_flag:
    """
    host, port = "127.0.0.1",65432
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((host, port))
    """
    ServerSocket = socket.socket()
    host = '127.0.0.1'
    port = 65432
    ThreadCount = 0
    try:
        ServerSocket.bind((host, port))
    except socket.error as e:
        print(str(e))

    print('Waitiing for a Connection..')
    ServerSocket.listen(5)
    add=0
    sock, add=ServerSocket.accept()
    print('Connected to: ' + add[0] + ':' + str(add[1]))
    sock.sendall(bytes("Hello from server",'utf-8'))
    #start_new_thread(threaded_client, (sock,))






    #c=socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    #c.connect(("127.0.0.1", 65430))




T=time.time()

if plot_flag==1:
    fps=5
elif plot_flag and plot_flag!=1:
    fps=10
else:
    fps=10

prev_frame_activ=4

file_number=1
dtime=datetime.now()
dtime = dtime.strftime("%d_%m_%Y %H_%M_%S")
#dtime=dtime[:-10]
direc=os.getcwd()
direc=os.path.join(direc,'data collected')

path=os.path.join(direc,dtime)

frame_total=150*fps #default 300 seconds
remember_time=120 #How many previous seconds to remember for saving. Make it less than frame_total seconds
total_files_to_save=1000

close_to_door=[]
exit_event_frame=-remember_time*fps
enter_event_frame=-remember_time*fps
exit_frame_counter=0


os.mkdir(path)


gc.enable()

temporal_activ=4*np.ones(6*fps) #No activity initialization

while True:

    print("frame ",frame," ,all data shape= ",all_data.shape)
    
    print("Person present ",person_present)
    #print("frame",frame)
    if file_number==total_files_to_save+1:
        break
    elap=time.time()-T  
    T=time.time()
    

    if frame==frame_total:# and person_present:
        if all_data.shape[0]:
            take_index_from=np.int32(np.max((all_data[0,0],frame-remember_time*fps)))
            print(" Take index FROM ",take_index_from)
            #take_index=np.min(np.where(all_data[:,0]==take_index_from)[0])
            take_index=np.where(all_data[:,0]==take_index_from)[0]
            
            if take_index.shape[0]:
                temporary_all=copy.deepcopy(all_data[take_index[0]:,:])
            else:                
                temporary_all=np.array([]).reshape(-1,data_col)
           
            all_data=copy.deepcopy(temporary_all)
            if np.mean(np.abs(all_data[:,1])):
            #if all_data.shape[0]:
                filename=path+'/'+f'{file_number:05d}'+'.npy'
                np.save(filename,all_data)
                print("     saving")
                file_number+=1
            else:
                print("    Not SAVING")
            all_data[:,0]-=remember_time*fps
            #print("HAHA ",all_data[0,0])
            entering_frame=[max((x-remember_time*fps),1) for x in entering_frame]#-10*fps
            print(entering_frame)
            prev_frame_enter=max(prev_frame_enter- remember_time*fps,1)
            prev_frame_exit=max(prev_frame_exit-remember_time*fps,1)
        frame-=remember_time*fps
        exit_event_frame=np.max((exit_event_frame-remember_time*fps,-remember_time*fps))
        gc.collect()

        
        

    
    if all_data.shape[0]>0 and person_present>0:
        recent_frame_data=np.where(all_data[:,0]==all_data[-1,0])[0]
        if recent_frame_data.shape[0]>0:
            plot_data=all_data[recent_frame_data,:]
            unity_data=convert_to_unity(plot_data)
            #print("unity_data= ",unity_data[:,1:10])
            if plot_flag==1:
                showing_plot(plot_data)
            if send_to_unity_flag:
                send_command,data_string=send_to_unity(unity_data)
                #print("data_string= ",data_string[:20])
                
                if send_command:# or True:
                    T1=time.time()
                    sock.sendall(data_string.encode("UTF-8"))
                    #receivedData = sock.recv(1024).decode("UTF-8")
                    elap=1000*(time.time()-T1)
                    print("Socekt comm takes miliseconds", elap)
                    #print(receivedData)
                    send_command=False
                
    if person_present==0 and plot_flag==1:
        plot_data=np.zeros((1,data_col))
        showing_plot(plot_data)

        
    frame+=1
    if frame==exit_event_frame+1 and flag_exit==1:
        close_to_door=[]
        exit_frame_counter=0

    if frame==1 and person_present:
        ######print("  frame: ",frame, "person_present: ",person_present)
        for i in range(person_present):
            person_id_list=np.vstack((person_id_list,i+1))
            entering_frame.append(frame)
        prev_frame_data=np.zeros((person_present,data_col_less))
        prev_frame_data[:,0]=frame
        prev_frame_data=np.hstack((prev_frame_data,person_id_list))
        all_data=np.zeros((person_present,data_col))
        all_data[:,0]=frame
        all_data[:,-3]=person_id_list.reshape(-1)
        latest_person_id=person_present
        activ=activity_detection(prev_frame_data)
        ######print("\n\n activ: ",activ,"\n\n")
        all_data[:,-2]=activ.reshape(-1)
        
        continue

    
    time_start = time.time()

    if on_off==2:
        playback1.update()
        playback1.update()
        playback2.update()
        playback2.update()
        playback3.update()
        playback3.update()
        playback4.update()
        playback4.update()

    capture1 = playback1.update()
    bodyFrame1 = bodyTracker1.update(capture=capture1)

    
    capture2 = playback2.update()
    bodyFrame2 = bodyTracker2.update(capture=capture2)

   
    capture3 = playback3.update()
    bodyFrame3 = bodyTracker3.update(capture=capture3)

    
    capture4 = playback4.update()
    bodyFrame4 = bodyTracker4.update(capture=capture4)

    num_bodies1 = bodyFrame1.get_num_bodies()   
    num_bodies2 = bodyFrame2.get_num_bodies()    
    num_bodies3 = bodyFrame3.get_num_bodies()
    num_bodies4 = bodyFrame4.get_num_bodies()
    



    cam1s=[]
    cam2s=[]
    cam3s=[]
    cam4s=[]   


    if num_bodies1 > 0:
        
        cam1s=get_camera_data(num_bodies1,bodyFrame1,W4,b4)
        del_short=np.where(np.logical_and(cam1s[:,3]<1100,cam1s[:,3]>0))[0]
        cam1s=np.delete(cam1s,del_short,0)
        
    if num_bodies2 > 0:
        far_flag=0
        cam2s=get_camera_data(num_bodies2,bodyFrame2,W3,b3)
        del_short=np.where(np.logical_and(cam2s[:,3]<1100,cam2s[:,3]>0))[0]
        cam2s=np.delete(cam2s,del_short,0)
       
        
    if num_bodies3 > 0:
        far_flag=0
        cam3s=get_camera_data(num_bodies3,bodyFrame3,W2,b2) 
        del_short=np.where(np.logical_and(cam3s[:,3]<1100,cam3s[:,3]>0))[0]
        cam3s=np.delete(cam3s,del_short,0)
    
    
    if num_bodies4 > 0:
        far_flag=0
        cam4s=get_camera_data(num_bodies4,bodyFrame4,W1,b1)
        del_short=np.where(np.logical_and(cam4s[:,3]<1100,cam4s[:,3]>0))[0]
        cam4s=np.delete(cam4s,del_short,0)
        

        close_noses=np.array([]).reshape(-1,3)
        for noses in cam4s:
            nose=noses[1:4].reshape(1,3)[0]
            if nose[0]<900 and nose[0]>-100 and nose[1]>9500 and nose[1]<room_length:
                close_noses=np.concatenate((close_noses,nose.reshape(1,3)),axis=0)
        if close_noses.shape[0]>0: #Possible Entry Event
            print("****WE HAVE A POSSIBLE ENTRY EVENT")
            if person_present==0:
                person_present+=close_noses.shape[0]
                prev_frame_enter=frame
                
                flag_enter=1
                for nose in close_noses:
                    nose_enter=nose
                    print("Person entered")
                    print("Nose position: ",nose_enter)

                    latest_person_id+=1
                    person_id_list=np.vstack((person_id_list,latest_person_id))
                    #person_id_list.append(latest_person_id)
                    entering_frame.append(frame)

                    prev_frame_data=np.vstack((prev_frame_data,np.zeros(prev_frame_data.shape[1])))
                    prev_frame_data[-1,1:4]=nose_enter
                    prev_frame_data[-1,5:8]=[500,8500,1500]
                flag_enter=0

            else: #Previous person/ persons already present
                print("")
                for nose in close_noses:

                    indices=np.where(all_data[:,0]==all_data[-1,0])[0]

                    data_taken=all_data[indices,:]
                    #print("   Enter flag on and data taken shape=",data_taken.shape)
                    

                    dist_enter=np.zeros(data_taken.shape[0])
                    for i in range(data_taken.shape[0]):
                        #dist_exit[i]=np.linalg.norm(prev_frame_data[i,1:4]-nose_exit)
                        dist_enter[i]=np.linalg.norm(data_taken[i,1:4]-nose)
                        print("  Enter Distances: ",dist_enter, "for ID: ",data_taken[i,-3])

                    lowest_dist=np.min(dist_enter)
                    lowest_dist_idx=np.argmin(dist_enter)
                    if lowest_dist>550 and (data_taken[lowest_dist_idx,1]>nose[0] or data_taken[lowest_dist_idx,1]==0): #new person x may be less than previous person x
                        print("  ### New person entered")
                        print("## Enter distance: ",lowest_dist)
                        print("## Entering Nose position: ",nose)
                        latest_person_id+=1
                        person_present+=1
                        print("Now Present: ",person_present," nose position: ",nose)
                        person_id_list=np.vstack((person_id_list,latest_person_id))
                        #person_id_list.append(latest_person_id)
                        entering_frame.append(frame)
                        prev_frame_data=np.vstack((prev_frame_data,np.zeros(prev_frame_data.shape[1])))
                        prev_frame_data[-1,1:4]=nose
                        prev_frame_data[-1,5:8]=[500,8500,1500]

        
    
        for noses in cam4s: #Exit Prediction
            nose=noses[1:4].reshape(1,3)[0]
            p1=noses[5:8].reshape(1,3)[0]
            posvect=nose-p1
            posvect=posvect.reshape((1,3))

            posvect2=np.array([posvect[0,0],posvect[0,1]]).reshape(1,2)        


            #angle=np.arccos(np.dot(posvect,np.array([1,0,0]))/(np.linalg.norm(posvect)*np.linalg.norm(np.array([1,0,0]))))[0]
            angle=np.arccos(np.dot(posvect2,np.array([1,0]))/(np.linalg.norm(posvect2)*np.linalg.norm(np.array([1,0]))))[0]
            a=angle*180/np.pi


            #Exit  
            if 180/np.pi*angle>110 and nose[0]<1000 and nose[0]>-100 and nose[1]>9500 and nose[1]<room_length:
                #if frame-prev_frame_exit>fps and frame-prev_frame_enter>fps:
                if 1==1:   
                    nose_exit=nose
                    
                    flag_exit=1
                    exit_event_frame=frame
                    print("EXIT PREDICTED AT FRAME ",frame," WITH ANGLE ",a)

    frame_data=np.array([]).reshape(-1,data_col_less)
    temporary_frame_data=np.array([]).reshape(-1,data_col_less)
    
    
    if num_bodies1==0 and num_bodies2==0 and num_bodies3==0 and num_bodies4==0: #Either no person present or dropout by kinects
        ######print('no body')
        if person_id_list.shape[0]: #Means person present but dropped out
            
           
            frame_data=copy.deepcopy(prev_frame_data)
           
            frame_data[:,0]=frame
            prev_frame_data=copy.deepcopy(frame_data) #For next frame
    
            frame_data2=np.hstack((frame_data,np.zeros((frame_data.shape[0],2))))
            activ=activity_detection(frame_data)
            frame_data2[:,-2]=activ.reshape(-1)
            all_data=np.vstack((all_data,frame_data2))

            #To check if person is standing still
            if frame>2*fps:
                for m in range(person_id_list.shape[0]):
                    if frame-entering_frame[m]>2*fps:
                        considered_indices=np.where(all_data[:,0]==frame-2*fps)[0]
                        ######print("CONsidered_indices ",considered_indices)
                        for n in considered_indices:
                            if all_data[n,-3]==person_id_list[m]:
                                considered_indices2=n
                        
                        cons_data=copy.deepcopy(all_data[considered_indices2,1:4])
                        considered_positions=cons_data #np.mean(cons_data,axis=0)
                        current_indices=np.where(all_data[:,0]==frame)[0]
                        for n in current_indices:
                            if all_data[n,-3]==person_id_list[m]:
                                current_indices2=n
                        ######print("CURrent_indices2 ",current_indices2)
                        cur_dat=copy.deepcopy(all_data[current_indices2,1:4])
                        current_positions=cur_dat
                       
                        if np.linalg.norm(considered_positions-current_positions)<1000: #1 meter
                            frame_data2[m,-1]=1
                            all_data[current_indices2,-1]=1
                            
        


        if flag_exit==1 and frame>exit_event_frame+1:
            indices=np.where(all_data[:,0]==all_data[-1,0])[0]
            indices_prev=np.where(all_data[:,0]==all_data[-1,0]-1)[0]
            data_taken=all_data[indices,:]
            data_taken_prev=all_data[indices_prev,:]
            if data_taken.shape[0]>data_taken_prev.shape[0]:
                data_taken_prev=np.vstack((data_taken_prev,np.zeros((data_taken.shape[0]-data_taken_prev.shape[0],data_taken_prev.shape[1]))))
            #print("   Exit flag on and data taken shape=",data_taken.shape)
            

            dist_exit=np.zeros(data_taken.shape[0])
            data_exit=np.zeros((data_taken.shape[0],2))
            
            for i in range(data_taken.shape[0]):
                if data_taken[i,1]<1800 and data_taken[i,2]>8500:
                    dist_exit[i]=np.linalg.norm(data_taken[i,1:3]-data_taken_prev[i,1:3])+1
                    print("Temp data taken= ",data_taken[i,1:3],"")
                    print("Temp Exit distance= ",dist_exit[i],"  ID= ",data_taken[i,-3],"")
                    if dist_exit[i]==1:
                        close_to_door.append(data_taken[i,-3]) #Take the person ID
                        print(" AAdding to close to door distance= ",dist_exit[i], "  ID= ",data_taken[i,-3])
                    else:
                        print("  Not adding still")


                
                
                
            #print("         DISTANCE=,",dist_exit)
            exit_frame_counter+=1

            if exit_frame_counter==10*fps:
                print("*****Exit frame counter reached")
                print(" **Close to door list length= ",len(close_to_door))
                exit_frame_counter=0
                flag_exit=0
                unique_ids=np.unique(close_to_door)
                unique_ids.sort()
                unique_ids=unique_ids[::-1]
                for ind in range(len(unique_ids)):
                    print("****close to door count=",close_to_door.count(unique_ids[ind]),"  ID= ",unique_ids[ind])
                    if close_to_door.count(unique_ids[ind])>3*fps:
                        print("**Deleting person ID= ",unique_ids[ind])
                        del_idx=np.where(data_taken[:,-3]==unique_ids[ind])[0]
                        print("** index to delete= ",del_idx,"")
                        #del_idx=np.where(prev_frame_data[:,-3]==unique_ids[ind])[0]
                        prev_frame_data=np.delete(prev_frame_data,del_idx,0)
                        person_id_list=np.delete(person_id_list,del_idx,0)
                close_to_door=[]
                person_present=prev_frame_data.shape[0]
                print(" **Person present after deleting= ",person_present)
        continue
    else:
        if num_bodies1>0:
            temporary_frame_data=np.vstack((temporary_frame_data,cam1s))
            
        if num_bodies2>0:
            temporary_frame_data=np.vstack((temporary_frame_data,cam2s))
            
        if num_bodies3>0:
            temporary_frame_data=np.vstack((temporary_frame_data,cam3s))
            
        if num_bodies4>0:
            temporary_frame_data=np.vstack((temporary_frame_data,cam4s))
            
        
        #The following few lines are to remove the same person detected by different cameras
        allidx=np.array(list(range(temporary_frame_data.shape[0])))
        for k in range(temporary_frame_data.shape[0]):
            same=[]
            if k in allidx:
                for l in range(k+1,temporary_frame_data.shape[0]):                        
                    if np.abs(np.linalg.norm(temporary_frame_data[k,1:3]-temporary_frame_data[l,1:3]))<same_threshold:# or np.abs(np.linalg.norm(temporary_neck_data[k,:]-temporary_neck_data[l,:]))<600:
                        same.append(l)
                        same.append(k)
              
                same=np.unique(same).reshape(-1)
                if list(same)==[]:
                    frame_data=np.vstack((frame_data,temporary_frame_data[k,:]))
                    
                else:
                    confidences=temporary_frame_data[same,4]
                    max_confidence=np.argmax(confidences)
                    frame_data=np.vstack((frame_data,temporary_frame_data[same[max_confidence],:]))
                    allidx=[x for x in allidx if x not in same]
                   


    

    

    
    
    #if frame==exit_event_frame and flag_exit==1:
    #    close_to_door=[]
    #    exit_frame_counter=0

    if flag_exit==1 and frame>exit_event_frame+1 and person_present:
        indices=np.where(all_data[:,0]==all_data[-1,0])[0]
        indices_prev=np.where(all_data[:,0]==all_data[-1,0]-1)[0]
        data_taken=all_data[indices,:]
        data_taken_prev=all_data[indices_prev,:]
        if data_taken.shape[0]>data_taken_prev.shape[0]:
                data_taken_prev=np.vstack((data_taken_prev,np.zeros((data_taken.shape[0]-data_taken_prev.shape[0],data_taken_prev.shape[1]))))
        print("   Exit flag on and data taken shape=",data_taken.shape,"  prev frame data shape= ",data_taken_prev.shape)
        

        dist_exit=np.zeros(data_taken.shape[0])
        data_exit=np.zeros((data_taken.shape[0],2))
        
        for i in range(data_taken.shape[0]):
            if data_taken[i,1]<1800 and data_taken[i,2]>8500:
                dist_exit[i]=np.linalg.norm(data_taken[i,1:3]-data_taken_prev[i,1:3])+1
                print("Temp data taken= ",data_taken[i,1:3],"")
                print("Temp Exit distance= ",dist_exit[i],"  ID= ",data_taken[i,-3],"")
                if dist_exit[i]==1:
                    close_to_door.append(data_taken[i,-3]) #Take the person ID
                    print(" AAdding to close to door distance= ",dist_exit[i], "  ID= ",data_taken[i,-3])
                else:
                    print("  Not adding still")
        exit_frame_counter+=1

        if exit_frame_counter==10*fps:
            print("*****Exit frame counter reached")
            exit_frame_counter=0
            flag_exit=0
            unique_ids=np.unique(close_to_door)
            unique_ids.sort()
            unique_ids=unique_ids[::-1]
            for ind in range(len(unique_ids)):
                print("****close to door count=",close_to_door.count(unique_ids[ind]),"  ID= ",unique_ids[ind])
                if close_to_door.count(unique_ids[ind])>3*fps:
                    print("**Deleting person ID= ",unique_ids[ind])
                    del_idx=np.where(data_taken[:,-3]==unique_ids[ind])[0]
                    print("** index to delete= ",del_idx,"")
                    prev_frame_data=np.delete(prev_frame_data,del_idx,0)
                    person_id_list=np.delete(person_id_list,del_idx,0)
            close_to_door=[]
            person_present=prev_frame_data.shape[0]

     
       
        
    
    #Give Person ID
    #ID Assignment Hungarian Algorithm
    distances=np.zeros((frame_data.shape[0],prev_frame_data.shape[0]))
    ######print("distances shape: ",distances.shape)
    for i in range(frame_data.shape[0]):
        for j in range(prev_frame_data.shape[0]):
            distances[i,j]=np.linalg.norm(frame_data[i,1:4]-prev_frame_data[j,1:4])
           
    row,col=linear_sum_assignment(distances)

    temp=copy.deepcopy(frame_data)


    if frame_data.shape[0]==prev_frame_data.shape[0]:
        frame_data[col,:]=temp[:,:]
        frame_data=np.hstack((frame_data,person_id_list))
      
        #prev_frame_data=copy.deepcopy(frame_data)

        ######print("  changed_data added to all",frame_data[:,1:4])
        
    
    elif frame_data.shape[0]<prev_frame_data.shape[0]: #Sudden Disappearance      

        frame_data=copy.deepcopy(prev_frame_data[:,0:-1])
        frame_data[:,0]=frame
        frame_data[col[0:frame_data.shape[0]],:]=temp[:,:]
        frame_data=np.hstack((frame_data,person_id_list))
        #prev_frame_data=copy.deepcopy(frame_data)


        
    

    elif frame_data.shape[0]>prev_frame_data.shape[0]: #Sudden Apperearance
        ######print("SUDDENNNNNN Apperearance")
        temp=copy.deepcopy(frame_data)
        frame_data=temp[row,:]
        temp=copy.deepcopy(frame_data)
        frame_data[col,:]=temp[:,:]
        now_idx=list(range(frame_data.shape[0]))
        del_idx=[x for x in now_idx if x not in col]
        frame_data=np.delete(frame_data,del_idx,0)
        frame_data=np.hstack((frame_data,person_id_list))
        #prev_frame_data=copy.deepcopy(frame_data)
    dist_calc=np.linalg.norm(frame_data[:,1:3]-prev_frame_data[:,1:3],axis=1)
    """
    if np.max(dist_calc)>2500:
        print("Frame: ",frame,"  Max dist: ",np.max(dist_calc))
        print("Maybe ghosting")
        frame_data=copy.deepcopy(prev_frame_data)
        frame_data[:,0]=frame
    """
    jump_prevent_idx=np.where(dist_calc>1500)[0]
    frame_data[jump_prevent_idx,1:14]=copy.deepcopy(prev_frame_data[jump_prevent_idx,1:14]) #Prevent sudden jumps in the data
    #frame_data[jump_prevent_idx,-1]=person_id_list[jump_prevent_idx]
    #frame_data[:,0]=frame
    
    prev_frame_data=copy.deepcopy(frame_data) #For next frame. data_col_less+1 shape

    
    frame_data2=np.hstack((frame_data,np.zeros((frame_data.shape[0],2))))
    activ=activity_detection(frame_data)
    ######print("\n\n activ: ",activ,"\n\n")
    frame_data2[:,-2]=activ.reshape(-1)

    
    all_data=np.vstack((all_data,frame_data2))

    ### TRY adding stationary flag for a person who is not moving for 2*fps frames
    
    if frame>2*fps:
        #print("Hello")
        for m in range(person_id_list.shape[0]):
            if frame-entering_frame[m]>2*fps:
                considered_indices=np.where(all_data[:,0]==frame-2*fps)[0]
                #print("CONsidered_indices ",considered_indices)
                considered_indices2=m
                for n in considered_indices:
                    if all_data[n,-3]==person_id_list[m]:
                        considered_indices2=n
                
                ######print("CONsidered_indices2 ",considered_indices2)
                cons_data=copy.deepcopy(all_data[considered_indices2,1:4])
                ######print("cons_data ",cons_data[1:4])
                considered_positions=cons_data
                current_indices=np.where(all_data[:,0]==frame)[0]
                for n in current_indices:
                    if all_data[n,-3]==person_id_list[m]:
                        current_indices2=n
                
                ######print("CURrent_indices2 ",current_indices2)
                cur_dat=copy.deepcopy(all_data[current_indices2,1:4])
                current_positions=cur_dat
                ######print("for id=",m+1, "dist=", np.linalg.norm(considered_positions-current_positions))
                if np.linalg.norm(considered_positions-current_positions)<1000: #2 meter
                    frame_data2[m,-1]=1
                    
                    all_data[current_indices2,-1]=1
                    
    
    prev_frame_data=copy.deepcopy(frame_data) #For next frame. Now prev_frame_data.shape[1]=data_col_less+1   
    
    #smoothing
    
    if frame>smooth_window and smooth_window!=0:

        b=[]
        for i in range(frame-smooth_window,frame):
            a=np.where(all_data[:,0]==i)
            
            if a[0].size>0:

                get=np.array(a[0]).reshape(-1)
                
                for j in get:
                    b.append(j)

               
        b=np.array(b).reshape(-1)
        
        if b.size>0:
            considered_data=all_data[b,:].reshape(-1,all_data.shape[1])
            ######print("considered_data ",considered_data.shape)
            
            for i in range(int(np.max(person_id_list)+1)):
                c=[]
                c=np.where(considered_data[:,-1]==i)[0]
                
                if c!=[]:
                    d1=np.mean(considered_data[c,1:4],axis=0)
                    d2=np.mean(considered_data[c,11:data_col_less],axis=0)


                    e=b[c]
                    all_data[e,1:4]=d1
                    all_data[e,11:14]=d2
    
    
    
######print("FINISHED")
playback1.close()
playback2.close()
playback3.close()
playback4.close()

            

# %%



