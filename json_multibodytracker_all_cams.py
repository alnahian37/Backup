import pickle
import json
import ast
from timeit import repeat
import numpy as np

from base64 import b16decode
import cv2
import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from math import inf

import sys
sys.path.insert(1, './../')




import os
######## Change it according to python script and data location ########

cwd=os.path.realpath(__file__)
script_name=os.path.basename(cwd)
cwd=cwd.replace(script_name,'')
print('Current Working Directory: ',cwd)
os.chdir(cwd) #Change it to your own directory
print(cwd)



import kinectpy as kpy
from kinectpy.k4abt._k4abtTypes import K4ABT_SEGMENT_PAIRS
from kinectpy.k4abt import _k4abt
from kinectpy.k4a import _k4a
from kinectpy.k4a._k4atypes import K4A_CALIBRATION_TYPE_DEPTH, K4A_CALIBRATION_TYPE_COLOR
import logging
from matplotlib import animation
import time


import scipy.io


p4=scipy.io.loadmat('cam4_params_umeyama.mat')['name']
W4=p4[0:3,:]
b4=p4[3:,:]


p3=scipy.io.loadmat('cam3_params_umeyama.mat')['name']
W3=p3[0:3,:]
b3=p3[3:,:]

p2=scipy.io.loadmat('cam2_params_umeyama.mat')['name']
W2=p2[0:3,:]
b2=p2[3:,:]

p1=scipy.io.loadmat('cam1_params_umeyama.mat')['name']
W1=p1[0:3,:]
b1=p1[3:,:]




all_indexes = range(len(K4ABT_SEGMENT_PAIRS)*4)
#all_indexes = range(len(K4ABT_SEGMENT_PAIRS)*2)



def get_bones(joints):
    xs_list = []
    ys_list = []
    zs_list = []
    print("JOINT SHAPE",joints.shape)
    joints = np.transpose(joints)
    # Draws bones
    for segmentId in range(len(K4ABT_SEGMENT_PAIRS)):
        segment_pair = K4ABT_SEGMENT_PAIRS[segmentId]
        x1 = joints[segment_pair[0]][0]
        x2 = joints[segment_pair[1]][0]
        #xs = np.linspace(x1, x2)

        y1 = joints[segment_pair[0]][1]
        y2 = joints[segment_pair[1]][1]
        #ys = np.linspace(y1, y2)

        z1 = joints[segment_pair[0]][2]
        z2 = joints[segment_pair[1]][2]
        #zs = np.linspace(z1, z2)
            
        xs = (x1, x2)
        ys = (y1, y2)
        zs = (z1, z2)
        xs_list.append(xs)
        ys_list.append(ys)
        zs_list.append(zs)

    return (xs_list, ys_list, zs_list)


def initialize_plots():
    fig = plt.figure()

    # Create axis

    

    axes = fig.add_subplot(111, projection='3d')  
    #axes.set_aspect('auto') 
    (x_min, x_max), (y_min, y_max), (z_min, z_max) = ((0, 4500), (0,10400), (0, 3000)) 

    #Table
    x1=np.linspace(1890,3115,20)
    z1=np.linspace(0,740,20)
    y1=np.linspace(1570,8850,20)

    x1=np.array(x1).reshape(len(x1),1)
    y1=np.array(y1).reshape(len(y1),1)
    z1=np.array(z1).reshape(len(z1),1)


    x,y=np.meshgrid(x1,y1)
    z=737*np.ones((len(x1),len(y1)))
    

    xx,zz=np.meshgrid(x1,z1)
    yy=1570*np.ones((len(x1),len(z1)))
    yy2=8850*np.ones((len(x1),len(z1)))




    x2=np.linspace(0,4500,20)
    z2=np.linspace(0,3000,20)
    y2=np.linspace(0,10000,20)

    x2=np.array(x2).reshape(len(x2),1)
    y2=np.array(y2).reshape(len(y2),1)
    z2=np.array(z2).reshape(len(z2),1)

    xxx,zzz=np.meshgrid(x2,z2)
    yyy=np.zeros((len(x2),len(z2)))

    y3,z3=np.meshgrid(y2,z2)
    x3=4500*np.ones((len(y2),len(z2)))

    y4,z4=np.meshgrid(y2,z2)
    x4=np.zeros((len(y2),len(z2)))

    y5=10000*np.ones((len(x2),len(z2)))


    #Door
    z2=np.linspace(0,2000,20)
    y2=np.linspace(8200,9000,20)
    y2=np.array(y2).reshape(len(y2),1)
    z2=np.array(z2).reshape(len(z2),1)
    y6,z6=np.meshgrid(y2,z2)
    x6=np.zeros((len(y2),len(z2)))
    axes.plot_surface(x6, y6, z6, color='k',alpha=1,edgecolor='k')

    #Wall
    axes.plot_surface(xxx,y5,zzz,alpha=0.7,color='green')
    axes.plot_surface(x4,y4,z4,alpha=0.7,color='blue')
    axes.plot_surface(x3,y3,z3,alpha=0.5,color='blue')
    axes.plot_surface(xxx,yyy,zzz,alpha=0.5,color='green',edgecolor='green')
    
    #Table
    axes.plot_surface(xx,yy,zz,color='magenta')
    axes.plot_surface(xx,yy2,zz,color='magenta')
    axes.plot_surface(x,y,z,color='white',alpha=1)
    
    

    
    
    
    #(x_min, x_max), (y_min, y_max), (z_min, z_max) = ((0, 4500), (0,10400), (0, 3000))
    axes.set_xlabel('X')
    axes.set_ylabel('Y')
    axes.set_zlabel('Z')
    axes.set_xlim(left=x_min, right=x_max)
    axes.set_ylim(bottom=y_min, top=y_max)
    axes.set_zlim(bottom=z_min, top=z_max)
    #axes.grid(True)
    #axes.set_axis_off()
    axes.view_init(azim = -115, elev = 35)
    #axes.view_init(azim = 80, elev = 160)
    
    #axes.w_yaxis.set_pane_color((1.0, 0, 0, 0.2))
    axes.w_zaxis.set_pane_color((0, 1, 0, 0.2))
    #axes.w_xaxis.set_pane_color((0, 0, 1, 0.2))
    axes.get_proj = lambda: np.dot(Axes3D.get_proj(axes), np.diag([0.5, 1, 0.3, 1]))
    plots = {index: axes.plot([0], [0], [0], 'bo-', markersize = 0)[0] for index in all_indexes}

    return fig, axes, plots


def make_video(join1, join2,join3,join4,frame_number, fig, ax, plots, out_filename = "outfile.mp4", show = True):
    '''Creates an animation from skeleton data
    Args:
    skeleton_data ((skeleton, offset) list): The data to animate
    out_filename (string): The file in which to save the video
    show (bool): Whether or not to preview the video before saving

    Returns:
    Nothing
    '''
    # Note that we make the assumption that the interval between frames is equal throughout. This
    # seems to hold mostly true, empirically.
    interval_microseconds = 1
    interval1 = int(interval_microseconds / 1e3)
    #(playback1, playback2, playback3, playback4) = all_playbacks
    #(playback1, playback2) = all_playbacks
    
    #(bodyTracker1, bodyTracker2, bodyTracker3, bodyTracker4) = body_trackers
    #(bodyTracker1, bodyTracker2) = body_trackers

    def init():
        '''Initializes the animation'''
        for index in all_indexes:
            plots[index].set_xdata(np.asarray([0]))
            plots[index].set_ydata(np.asarray([0]))
            plots[index].set_3d_properties(np.asarray([0]))
        return iter(tuple(plots.values()))

    

    def animate(i):
        '''Render each frame'''
        time_start = time.time()
        #join1=joins1[cou]
        #join2=joins2[cou]
        #join3=joins3[cou]
        #join4=joins4[cou]
        

        
        time2 = time.time()
        #print("it takes {} seconds to GET ALL frame".format(time2-time_start))

        #num_bodies1 = bodyFrame1.get_num_bodies()
        #num_bodies2 = bodyFrame2.get_num_bodies()
        #num_bodies3 = bodyFrame3.get_num_bodies()
        #num_bodies4 = bodyFrame4.get_num_bodies()
        
        #print("# number of bodies: \n cam-1: {0}, \n cam-2: {1}, \n cam-3: {2}, \n cam-4: {3}".format(num_bodies1, num_bodies2, num_bodies3, num_bodies4, i))

        #print("# number of bodies: \n cam-1: {0}, \n cam-2: {1}".format(num_bodies1, num_bodies2, i))
        
        joints1 = []
        joints2 = []
        joints3 = []
        joints4 = []
        bones1 = []
        bones2 = []
        bones3 = []
        bones4 = []

        # Extract joint coordinates
        
            #print("SHAPE",_joints1.shape)
        """
        if join1==None:
            join1=[]
        if join2==None:
            join2=[]
        if join3==None:
            join3=[]
        if join4==None:
            join4=[]
        """
        print("JOIN1",join1.shape)
        if 1: 
            #if len(join1)>0:
            
            _joints1 =W4@join1.T+b4.T
            print("SHAPE",_joints1.shape)
            joints1.append(_joints1)
        
        if 1: 
        #& len(join2)>0:
            _joints2 =W3@join2.T+b3.T
            joints2.append(_joints2)
        
        if 1:
        #len(join3)>0:
            _joints3 =W2@join3.T+b2.T
            joints3.append(_joints3)
        if 1:
        #if len(join4)>0:
            _joints4 =W1@join4.T+b1.T
            joints4.append(_joints4)

        
        

        
            

        #if len(joints1) !=1000:
        if len(joints1)>0:
            print('joints1: ', len(joints1))
            bones1 = get_bones(joints1[0])
            print('bones1: ', len(bones1[0]))
            
            #print("bones1: ", len(bones1[0]))
            #print("bones2: ", len(bones1[1]))
            #print("bones3: ", len(bones1[2]))

        if len(joints2) >0 :
            bones2 = get_bones(joints2[0])
        if len(joints3) >0:
            bones3 = get_bones(joints3[0])
        if len(joints4)>0:
            bones4 = get_bones(joints4[0])
        
        # Draws bones

        print("Frame: ", frame_number)
        bones_from_all_cameras = [bones1, bones2, bones3, bones4]
        for b, set_of_bones in enumerate(bones_from_all_cameras):
            color = ((b + 1)*0.25, (b + 1)* 0.1, (b + 1)*0.1)
            if set_of_bones != []:
                x_orientations = np.asarray(set_of_bones[0])
                y_orientations = np.asarray(set_of_bones[1])
                z_orientations = np.asarray(set_of_bones[2])
                print("x_orientations: ", x_orientations.shape)
                
                for bone_id in range(len(x_orientations)):
                    if b == 0:
                        index = bone_id
                        plots[index].set_xdata(x_orientations[bone_id])
                        plots[index].set_ydata(y_orientations[bone_id])
                        plots[index].set_3d_properties(z_orientations[bone_id])
                        plots[index].set_markersize(1)
                        plots[index].set_color(color)
                    if b == 1:
                        index = 31 + bone_id
                        plots[index].set_xdata(x_orientations[bone_id])
                        plots[index].set_ydata(y_orientations[bone_id])
                        plots[index].set_3d_properties(z_orientations[bone_id])
                        plots[index].set_markersize(1)
                        plots[index].set_color(color)
                    
                    
                    if b == 2:
                        index = 31*2 + bone_id
                        plots[index].set_xdata(x_orientations[bone_id])
                        plots[index].set_ydata(y_orientations[bone_id])
                        plots[index].set_3d_properties(z_orientations[bone_id])
                        plots[index].set_markersize(1)
                        plots[index].set_color(color)
                    if b == 3:
                        index = 31*3 + bone_id
                        plots[index].set_xdata(x_orientations[bone_id])
                        plots[index].set_ydata(y_orientations[bone_id])
                        plots[index].set_3d_properties(z_orientations[bone_id])
                        plots[index].set_markersize(1)
                        plots[index].set_color(color)
                    

        print("it takes {} seconds to plot a frame".format(time.time()-time2))
        return iter(tuple(plots.values()))

    #log.info('Creating animation')

    
    cou=0
    
    video = animation.FuncAnimation(
        fig,
        animate, init,
        interval=100,
        blit=True,cache_frame_data=False,repeat=False
    )
    if show:
        plt.show()
    
    

    #log.info(f'Saving video to {out_filename}')
    #video.save(out_filename, fps=5, extra_args=['-vcodec', 'libx264'])

def make_plot(join1, join2,join3,join4, fig, ax, plots, out_filename = "outfile.mp4", show = True):
    '''Render each frame'''
    
   
    
    joints1 = []
    joints2 = []
    joints3 = []
    joints4 = []
    bones1 = []
    bones2 = []
    bones3 = []
    bones4 = []

    # Extract joint coordinates
    
        #print("SHAPE",_joints1.shape)
    if len(join1)>0:
        
        _joints1 =W4@join1.T+b4.T
        joints1.append(_joints1)
    
    if len(join2)>0:
        _joints2 =W3@join2.T+b3.T
        joints2.append(_joints2)
    
    if len(join3)>0:
        _joints3 =W2@join3.T+b2.T
        joints3.append(_joints3)
    if len(join4)>0:
        _joints4 =W1@join4.T+b1.T
        joints4.append(_joints4)
   

    #if len(joints1) !=1000:
    if len(joints1)>0:
        #print('joints1: ', len(joints1))
        bones1 = get_bones(joints1[0])
        #print('bones1: ', len(bones1[0]))
        
        #print("bones1: ", len(bones1[0]))
        #print("bones2: ", len(bones1[1]))
        #print("bones3: ", len(bones1[2]))

    if len(joints2) >0 :
        bones2 = get_bones(joints2[0])
    if len(joints3) >0:
        bones3 = get_bones(joints3[0])
    if len(joints4)>0:
        bones4 = get_bones(joints4[0])
    
    # Draws bones
    bones_from_all_cameras = [bones1, bones2, bones3, bones4]
    for b, set_of_bones in enumerate(bones_from_all_cameras):
        color = ((b + 1)*0.25, (b + 1)* 0.1, (b + 1)*0.1)
        if set_of_bones != []:
            x_orientations = np.asarray(set_of_bones[0])
            y_orientations = np.asarray(set_of_bones[1])
            z_orientations = np.asarray(set_of_bones[2])
            #print("x_orientations: ", x_orientations.shape)
            
            for bone_id in range(len(x_orientations)):
                if b == 0:
                    index = bone_id
                    plots[index].set_xdata(x_orientations[bone_id])
                    plots[index].set_ydata(y_orientations[bone_id])
                    plots[index].set_3d_properties(z_orientations[bone_id])
                    plots[index].set_markersize(1)
                    plots[index].set_color(color)
                if b == 1:
                    index = 31 + bone_id
                    plots[index].set_xdata(x_orientations[bone_id])
                    plots[index].set_ydata(y_orientations[bone_id])
                    plots[index].set_3d_properties(z_orientations[bone_id])
                    plots[index].set_markersize(1)
                    plots[index].set_color(color)               
                
                if b == 2:
                    index = 31*2 + bone_id
                    plots[index].set_xdata(x_orientations[bone_id])
                    plots[index].set_ydata(y_orientations[bone_id])
                    plots[index].set_3d_properties(z_orientations[bone_id])
                    plots[index].set_markersize(1)
                    plots[index].set_color(color)
                if b == 3:
                    index = 31*3 + bone_id
                    plots[index].set_xdata(x_orientations[bone_id])
                    plots[index].set_ydata(y_orientations[bone_id])
                    plots[index].set_3d_properties(z_orientations[bone_id])
                    plots[index].set_markersize(1)
                    plots[index].set_color(color)
    
    #fig.show()

    #fig.canvas.draw()
    #fig.canvas.flush_events()
                
    #plt.show()
        #print("it takes {} seconds to plot a frame".format(time.time()-time2))
    return fig, axes, plots





file=open("body_data.pkl","rb")
theta_load=pickle.load(file)
print(len(theta_load))
alu1=ast.literal_eval(theta_load[0][len(theta_load[0])-1])
frame1=alu1.get('frame')
alu2=ast.literal_eval(theta_load[1][len(theta_load[1])-1])
frame2=alu2.get('frame')
alu3=ast.literal_eval(theta_load[2][len(theta_load[2])-1])
frame3=alu3.get('frame')
alu4=ast.literal_eval(theta_load[3][len(theta_load[3])-1])
frame4=alu4.get('frame')

print(frame1,frame2,frame3,frame4)
total_frames=max(frame1,frame2,frame3,frame4)
print(total_frames)

#cam1=ast.literal_eval(theta_load[0])
#cam2=ast.literal_eval(theta_load[1])
#cam3=ast.literal_eval(theta_load[2])
#cam4=ast.literal_eval(theta_load[3])



file.close()


print(len(theta_load[0]))
#print(theta_load[3][0])

cam1=[]
cam2=[]
cam3=[]
cam4=[]

for k in range(len(theta_load)):
    
    for i in range(len(theta_load[k])): #for all frames

        a=ast.literal_eval(theta_load[k][i])
        #print('frame: ',a.get('frame'))

        join=[]

        frame_num=a.get('frame')

        #print(frame_num)

        for j in range(len(a.get('Joints'))):

            join.append(a.get('Joints')[j].get('xyz'))
        #print(join)
        join=np.array(join)
        #print(join.shape)
        dict={'frame':frame_num,'xyz':join}
        if k==0:
            cam1.append(dict)
        elif k==1:
            cam2.append(dict)
        elif k==2:
            cam3.append(dict)
        elif k==3:
            cam4.append(dict)


        #print(a.get('Joints')[0].get('pos').get('x'))
        #print(len(a.get('Joints')))
        #print(a.get('Joints')[0].get('confidence'))

#for i in range(10):
    #print(cam1[i].get('frame'),cam2[i].get('frame'),cam3[i].get('frame'),cam4[i].get('frame'))

l1=[]
l2=[]
l3=[]
l4=[]

for i in range(total_frames):
    l1.append(np.zeros((32,3)))
    l2.append(np.zeros((32,3)))
    l3.append(np.zeros((32,3)))
    l4.append(np.zeros((32,3)))
    


print(cam1[len(cam1)-1].get('frame'))

for i in range(len(cam1)-1):
    l1[cam1[i].get('frame')]=cam1[i].get('xyz')





for i in range(len(cam2)-1):
    l2[cam2[i].get('frame')]=cam2[i].get('xyz')



for i in range(len(cam3)-1):
    l3[cam3[i].get('frame')]=cam3[i].get('xyz')



for i in range(len(cam4)-1):
    l4[cam4[i].get('frame')]=cam4[i].get('xyz')


print("FLAGS")
print(len(l1))
print(len(l2))
print(len(l3))
print(len(l4))
print("FLAGS END")

if __name__ == "__main__":
    
    

    # Initialize the library, if the library is not found, add the library path as argument
    kpy.initialize_libraries(track_body=True)

    plt.ion()

    fig, axes, plots = initialize_plots()
    plt.show(block=False)
    plt.pause(2)
    plt.close()
    frame = 1


    fig, axes, plots = initialize_plots()
    
    while frame<total_frames:

        
        

        joint1=l1[frame]
        joint2=l2[frame]
        joint3=l3[frame]
        joint4=l4[frame]

        

        print("L1 shape",l1[0].shape) 
        

            
        
        
        print("frame number: ",frame)
        
        time1=time.time()

        fi,a,p=make_plot(joint1,joint2,joint3,joint4,fig, axes, plots,out_filename = "outfile.mp4", show = True)
        print("TAKES {} secods per loop".format(time.time()-time1))

        time1=time.time()
        fi.canvas.draw()
        fi.canvas.flush_events()
        time2=time.time()
        print("it takes {} seconds to plot a frame".format(time2-time1))
        #fi.canvas.flush_events()

        #plt.show(block=False)
        #plt.pause(2)
        #plt.close()

        #make_video(l1,l2,l3,l4,frame, fig, axes, plots,out_filename = "outfile.mp4", show = True)
        frame+=3  
        #time.sleep(0.1)
        #k+=1










