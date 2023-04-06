
from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import scipy.io


import os
######## Change it according to python script and data location ########
cwd=os.path.realpath(__file__)
script_name=os.path.basename(cwd)
cwd=cwd.replace(script_name,'')
print('Current Working Directory: ',cwd)
#os.chdir(r'C:\\Users\\nahia\Google Drive (nahian.buet11@gmail.com)\\Spring-22 Drive Folder\\ECSE 6850\\HW\\Programming 1')
os.chdir(cwd) #Change it to your own directory


def umeyama(from_points, to_points):
    
    assert len(from_points.shape) == 2, \
        "from_points must be a m x n array"
    assert from_points.shape == to_points.shape, \
        "from_points and to_points must have the same shape"
    
    N, m = from_points.shape
    
    mean_from = from_points.mean(axis = 0)
    mean_to = to_points.mean(axis = 0)
    
    delta_from = from_points - mean_from # N x m
    delta_to = to_points - mean_to       # N x m
    
    sigma_from = (delta_from * delta_from).sum(axis = 1).mean()
    sigma_to = (delta_to * delta_to).sum(axis = 1).mean()
    
    cov_matrix = delta_to.T.dot(delta_from) / N
    
    U, d, V_t = np.linalg.svd(cov_matrix, full_matrices = True)
    cov_rank = np.linalg.matrix_rank(cov_matrix)
    S = np.eye(m)
    
    if cov_rank >= m - 1 and np.linalg.det(cov_matrix) < 0:
        S[m-1, m-1] = -1
    elif cov_rank < m-1:
        raise ValueError("colinearility detected in covariance matrix:\n{}".format(cov_matrix))
    
    R = U.dot(S).dot(V_t)
    c = (d * S.diagonal()).sum() / sigma_from
    t = mean_to - c*R.dot(mean_from)
    
    return c*R, t


if __name__ == "__main__":

    # Physical space
    # physical_points = np.array([[0.7556, 0, 3.985],
    #                         [0.7430, 0, 4.530],
    #                         [0.7200, 0, 4.944],
    #                         [1.1870, 0, 3.600],
    #                         [1.2000, 0, 3.978],
    #                         [1.1870, 0, 4.524],
    #                         [1.1680, 0, 4.943],
    #                         [1.5870, 0, 3.953]])

    # Unity points
    # to_points = np.array([[-4.600, 0.127, -6.618],
    #                       [-4.590, 0.127, -6.182],
    #                       [-4.597, 0.127, -5.840],
    #                       [-4.250, 0.127, -6.876],
    #                       [-4.250, 0.127, -6.621],
    #                       [-4.250, 0.127, -6.182],
    #                       [-4.250, 0.127, -5.849],
    #                       [-3.954, 0.127, -6.621]])

    bs=[]
    with open('cam4con1.txt') as f:
        for line in f:
            a=line.strip().split(',')
            b=[int(x) for x in a]
            bs.append(b)
    f.close()

    with open('cam4con2.txt') as f:
        for line in f:
            a=line.strip().split(',')
            b=[int(x) for x in a]
            bs.append(b)
    
    #print(bs)
    bs=np.array(bs)
    print(bs.shape)

    XYZ=bs[:,2:5]
    print(XYZ.shape)


    xyz1=scipy.io.loadmat('kinect_cam4_con1.mat')['name']
    #print(xyz1)
    print(xyz1.shape)
    xyz2=scipy.io.loadmat('kinect_cam4_con2.mat')['name']
    print(xyz2.shape)

    xyz=np.concatenate((xyz1,xyz2),axis=0)
    print(xyz.shape)

    #Camera
    from_points = xyz

    #Physical
    to_points = XYZ




    M, t = umeyama(from_points, to_points)
    #t = np.reshape(t, (3, 1))

    #M_h = np.concatenate((M, t), axis = 1)
    #M_h = np.concatenate((M_h, np.reshape(np.array([0, 0, 0, 1]), (1, 4))), axis = 0)
    #print("M_h",M_h)

    #ex_pt = np.array([1.1870, 0, 4.524, 1])
    #ex_pt = np.transpose(np.reshape(ex_pt, (1, 4)))
    #tr_point = M_h @ ex_pt

    xyz_w=M@xyz.T+t.reshape(3,1)
    xyz_w=xyz_w.T

    print(xyz_w[0:5,:])
    print(XYZ[0:5,:])

    #print("Shape",t.shape)
    #print("M",M)
    #print(tr_point)




