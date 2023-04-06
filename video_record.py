import json
import time
import pickle
import kinectpy as kpy
import ast
from kinectpy.k4a._k4atypes import K4A_CALIBRATION_TYPE_DEPTH
import gc




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


kpy.initialize_libraries()

fps=int(input("Enter the fps: 0 for 5fps, 1 for 15fps, 2 for 30fps:"))

#Start the devices
device_config = kpy.default_configuration	
device_config.color_resolution=1 
device_config.depth_mode = 2
device_config.wired_sync_mode=0
device_config.camera_fps=fps


device_config1 = kpy.default_configuration
device_config1.color_resolution = 0 
device_config1.depth_mode=2
device_config1.wired_sync_mode=0
device_config1.camera_fps=fps



###
device_config2 = kpy.default_configuration	
device_config2.color_resolution=0 #2160p
device_config2.depth_mode = 2
device_config2.wired_sync_mode=0
device_config2.camera_fps=fps



device_config3 = kpy.default_configuration
device_config3.color_resolution = 0 #2160p
device_config3.depth_mode=2
device_config3.wired_sync_mode=0
device_config3.camera_fps=fps




# Start device

single_file_time=100 #Seconds

direc=os.getcwd()
direc=os.path.join(direc,'video collected')


record_time=int(input("Enter the time (minutes) to record in multiples of 5 minutes: "))

record_time=record_time*60 #Seconds

total_file=record_time//single_file_time #Total number of files to be recorded




i=0


while i<total_file:
    path=os.path.join(direc,'videos_'+str(i+1))
    os.mkdir(path)
    video_filename4 = os.path.join(path,'output'+str(i+1)+'_4.mkv')
    video_filename3 = os.path.join(path,'output'+str(i+1)+'_3.mkv')
    video_filename2 = os.path.join(path,'output'+str(i+1)+'_2.mkv')
    video_filename1 = os.path.join(path,'output'+str(i+1)+'_1.mkv')

    playback4 = kpy.start_device(device_index=0,config=device_config, record=True, record_filepath=video_filename4)
    playback3 = kpy.start_device(device_index=1,config=device_config1, record=True, record_filepath=video_filename3)
    playback2 = kpy.start_device(device_index=2,config=device_config2, record=True, record_filepath=video_filename2)
    playback1 = kpy.start_device(device_index=3,config=device_config3, record=True, record_filepath=video_filename1)

    t1=time.time()

    while time.time()-t1<single_file_time:
        playback4.update()
        playback3.update()
        playback2.update()
        playback1.update()
    i+=1
    playback4.close()
    playback3.close()
    playback2.close()
    playback1.close()

