# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
import os.path as osp
import cv2
import numpy as np
import matplotlib
import torch
#matplotlib.use('tkagg')
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib as mpl
#from main.config import cfg
from main import config as cfg
from PIL import Image, ImageDraw
import gc

def get_keypoint_rgb(skeleton):
    rgb_dict= {}
    for joint_id in range(len(skeleton)):
        joint_name = skeleton[joint_id]['name']

        if joint_name.endswith('thumb_null'):
            rgb_dict[joint_name] = (255, 0, 0)
        elif joint_name.endswith('thumb3'):
            rgb_dict[joint_name] = (255, 51, 51)
        elif joint_name.endswith('thumb2'):
            rgb_dict[joint_name] = (255, 102, 102)
        elif joint_name.endswith('thumb1'):
            rgb_dict[joint_name] = (255, 153, 153)
        elif joint_name.endswith('thumb0'):
            rgb_dict[joint_name] = (255, 204, 204)
        elif joint_name.endswith('index_null'):
            rgb_dict[joint_name] = (0, 255, 0)
        elif joint_name.endswith('index3'):
            rgb_dict[joint_name] = (51, 255, 51)
        elif joint_name.endswith('index2'):
            rgb_dict[joint_name] = (102, 255, 102)
        elif joint_name.endswith('index1'):
            rgb_dict[joint_name] = (153, 255, 153)
        elif joint_name.endswith('middle_null'):
            rgb_dict[joint_name] = (255, 128, 0)
        elif joint_name.endswith('middle3'):
            rgb_dict[joint_name] = (255, 153, 51)
        elif joint_name.endswith('middle2'):
            rgb_dict[joint_name] = (255, 178, 102)
        elif joint_name.endswith('middle1'):
            rgb_dict[joint_name] = (255, 204, 153)
        elif joint_name.endswith('ring_null'):
            rgb_dict[joint_name] = (0, 128, 255)
        elif joint_name.endswith('ring3'):
            rgb_dict[joint_name] = (51, 153, 255)
        elif joint_name.endswith('ring2'):
            rgb_dict[joint_name] = (102, 178, 255)
        elif joint_name.endswith('ring1'):
            rgb_dict[joint_name] = (153, 204, 255)
        elif joint_name.endswith('pinky_null'):
            rgb_dict[joint_name] = (255, 0, 255)
        elif joint_name.endswith('pinky3'):
            rgb_dict[joint_name] = (255, 51, 255)
        elif joint_name.endswith('pinky2'):
            rgb_dict[joint_name] = (255, 102, 255)
        elif joint_name.endswith('pinky1'):
            rgb_dict[joint_name] = (255, 153, 255)
        else:
            rgb_dict[joint_name] = (230, 230, 0)
        """
        elif joint_name.endswith('wrist'):
        elif joint_name.endswith('wrist_twist'):
        elif joint_name.endswith('forearm'):
        else:
            print('Unsupported joint name: ' + joint_name)
            assert 0
        """
        
    return rgb_dict

def vis_keypoints(img, kps, score, skeleton, filename, score_thr=0.4, line_width=3, circle_rad = 3):
    
    rgb_dict = get_keypoint_rgb(skeleton)
    _img = Image.fromarray(img.transpose(1,2,0).astype('uint8')) 
    draw = ImageDraw.Draw(_img)
    for i in range(len(skeleton)):
        joint_name = skeleton[i]['name']
        pid = skeleton[i]['parent_id']
        parent_joint_name = skeleton[pid]['name']
        
        kps_i = (kps[i][0].astype(np.int32), kps[i][1].astype(np.int32))
        kps_pid = (kps[pid][0].astype(np.int32), kps[pid][1].astype(np.int32))

        if score[i] > score_thr and score[pid] > score_thr and pid != -1:
            draw.line([(kps[i][0], kps[i][1]), (kps[pid][0], kps[pid][1])], fill=rgb_dict[parent_joint_name], width=line_width)
        if score[i] > score_thr:
            draw.ellipse((kps[i][0]-circle_rad, kps[i][1]-circle_rad, kps[i][0]+circle_rad, kps[i][1]+circle_rad), fill=rgb_dict[joint_name])
        if score[pid] > score_thr and pid != -1:
            draw.ellipse((kps[pid][0]-circle_rad, kps[pid][1]-circle_rad, kps[pid][0]+circle_rad, kps[pid][1]+circle_rad), fill=rgb_dict[parent_joint_name])

    _img.save(osp.join(cfg.cfg.vis_dir, filename))


def vis_3d_keypoints(kps_3d, score, skeleton, filename, score_thr=0.4, line_width=3, circle_rad=3):

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    rgb_dict = get_keypoint_rgb(skeleton)
    
    for i in range(len(skeleton)):
        joint_name = skeleton[i]['name']
        pid = skeleton[i]['parent_id']
        parent_joint_name = skeleton[pid]['name']

        x = np.array([kps_3d[i,0], kps_3d[pid,0]])
        y = np.array([kps_3d[i,1], kps_3d[pid,1]])
        z = np.array([kps_3d[i,2], kps_3d[pid,2]])

        if score[i] > score_thr and score[pid] > score_thr and pid != -1:
            ax.plot(x, z, -y, c = np.array(rgb_dict[parent_joint_name])/255., linewidth = line_width)
        if score[i] > score_thr:
            ax.scatter(kps_3d[i,0], kps_3d[i,2], -kps_3d[i,1], c = np.array(rgb_dict[joint_name]).reshape(1,3)/255., marker='o')
        if score[pid] > score_thr and pid != -1:
            ax.scatter(kps_3d[pid,0], kps_3d[pid,2], -kps_3d[pid,1], c = np.array(rgb_dict[parent_joint_name]).reshape(1,3)/255., marker='o')

    #plt.show()
    #cv2.waitKey(0)
    
    fig.savefig(osp.join(cfg.cfg.vis_dir, filename), dpi=fig.dpi)



def draw_joints_mano_42(filename, J0, valid=None):
    if torch.is_tensor(J0):
        J = J0.detach().cpu().numpy()
    else:
        J = J0
    color = [
        (0, 0, 0), 
        (50, 0, 0), (100, 0, 0), (150, 0, 0), (200, 0, 0), #(250, 0, 0), 
        (0, 60, 0), (0, 120, 0), (0, 180, 0), (0, 240, 0),
        (0, 0, 60), (0, 0, 120), (0, 0, 180), (0, 0, 240),
        (60, 60, 0), (120, 120, 0), (180, 180, 0), (240, 240, 0),
        (0, 60, 60), (0, 120, 120), (0, 180, 180), (0, 240, 240),
        
        (255, 255, 255), 
        (255, 255, 255), (255, 255, 255), (255, 255, 255), (255, 255, 255),
        (255, 255, 255), (255, 255, 255), (255, 255, 255), (255, 255, 255),
        (255, 255, 255), (255, 255, 255), (255, 255, 255), (255, 255, 255),
        (255, 255, 255), (255, 255, 255), (255, 255, 255), (255, 255, 255),
        (255, 255, 255), (255, 255, 255), (255, 255, 255), (255, 255, 255),
        
    ]
    skeleton = [
        [1, 5, 9, 13, 17], [2], [3], [4], [], 
        [6], [7], [8], [],
        [10], [11], [12], [],
        [14], [15], [16], [],
        [18], [19], [20], [],
    ]
    joint_num = 21
    with open(filename, 'w') as f:
        for i in range(J.shape[0]):
            if i >= joint_num and valid is not None and valid[i-joint_num][0] == 0:
                print('v', J[i-joint_num][0], J[i-joint_num][1], J[i-joint_num][2], color[i][0], color[i][1], color[i][2], file=f)
            else:  
                print('v', J[i][0], J[i][1], J[i][2], color[i][0], color[i][1], color[i][2], file=f)
                    
        for i in range(J.shape[0]):
            if i < joint_num:
                for child in skeleton[i]:
                    print('l', i + 1, child + 1, file=f)
            else:
                for child in skeleton[i - joint_num]:
                    print('l', i + 1, child + joint_num + 1, file=f)


def draw_joints_mano_21(filename, J0, valid=None):
    if torch.is_tensor(J0):
        J = J0.detach().cpu().numpy()
    else:
        J = J0
    color = [
        (0, 0, 0), 
        (50, 0, 0), (100, 0, 0), (150, 0, 0), (200, 0, 0), #(250, 0, 0), 
        (0, 60, 0), (0, 120, 0), (0, 180, 0), (0, 240, 0),
        (0, 0, 60), (0, 0, 120), (0, 0, 180), (0, 0, 240),
        (60, 60, 0), (120, 120, 0), (180, 180, 0), (240, 240, 0),
        (0, 60, 60), (0, 120, 120), (0, 180, 180), (0, 240, 240),
    ]
    skeleton = [
        [1, 5, 9, 13, 17], [2], [3], [4], [], 
        [6], [7], [8], [],
        [10], [11], [12], [],
        [14], [15], [16], [],
        [18], [19], [20], [],
    ]
    joint_num = 21
    with open(filename, 'w') as f:
        for i in range(J.shape[0]):
            print('v', J[i][0], J[i][1], J[i][2], color[i][0], color[i][1], color[i][2], file=f)
                    
        for i in range(J.shape[0]):
            for child in skeleton[i]:
                print('l', i + 1, child + 1, file=f)

def save_obj(file, vertices, faces, color=None):
    with open(file, 'w+') as f:
        for i in range(vertices.shape[0]):
            if color is None:
                print('v', vertices[i][0], vertices[i][1], vertices[i][2], file=f)
            else:
                print('v', vertices[i][0], vertices[i][1], vertices[i][2], color[i][0], color[i][1], color[i][2], file=f)
        for i in range(faces.shape[0]):
            print('f', faces[i][0] + 1, faces[i][1] + 1, faces[i][2] + 1, file=f)


def plot(x, save_path, labels=[], title=''):
    #print(cfg.cfg.feature_grad.abs().mean().cpu())
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w', 'tab:brown']
    ave_grads = x
    data_len = ave_grads.shape[1]
    layers = [i for i in range(data_len)]
    # for n, p in named_parameters:
    #     if(p.requires_grad) and ("bias" not in n):
    #         layers.append(n)
    #         if p.grad is None:
    #             ave_grads.append(0)
    #         else:
    #             ave_grads.append(p.grad.abs().mean().cpu())
    plt.figure(num=1, figsize=(15, 8))
    for i in range(x.shape[0]):
        plt.plot(ave_grads[i], alpha=0.3, color=colors[i])
    if labels != []:
        plt.legend(labels=labels)
    plt.hlines(0, 0, data_len+1, linewidth=1, color="k" )
    plt.xticks(range(0,data_len, 1), layers, rotation="vertical")
    plt.xlim(xmin=0, xmax=data_len)
    plt.ylim(-10, 10)
    #plt.xlabel("Layers")
    #plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    #os.makedirs(save_path, exist_ok=True)
    plt.savefig(save_path)
    plt.clf()