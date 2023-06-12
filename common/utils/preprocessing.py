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
#from main.config import cfg
from main import config as cfg
import sys
sys.path.append(cfg.cfg.root_dir + '/common')
from common.utils.transforms import world2cam, cam2pixel
from glob import glob
import random
import math

def load_img(path, order='RGB'):
    
    # load
    img = cv2.imread(path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    if not isinstance(img, np.ndarray):
        raise IOError("Fail to read %s" % path)

    if order=='RGB':
        img = img[:,:,::-1].copy()
    
    img = img.astype(np.float32)
    return img

def load_krt(path):
    # Load KRT file containing intrinsic and extrinsic camera parameters 
    cameras = {}

    with open(path, "r") as f:
        while True:
            name = f.readline()
            if name == "":
                break

            intrin = [[float(x) for x in f.readline().split()] for i in range(3)]
            dist = [float(x) for x in f.readline().split()]
            extrin = [[float(x) for x in f.readline().split()] for i in range(3)]
            f.readline()

            cameras[name[:-1]] = {
                    "intrin": np.array(intrin),
                    "dist": np.array(dist),
                    "extrin": np.array(extrin)}

    return cameras

def get_bbox(joint_world, joint_valid, camrot, campos, focal, princpt):
    
    joint_cam = []
    for i in range(len(joint_world)):
        joint_cam.append(world2cam(joint_world[i], camrot, campos))
    joint_cam = np.array(joint_cam).reshape(-1,3)

    x_img, y_img, z_img = cam2pixel(joint_cam, focal, princpt)
    x_img = x_img[joint_valid==1]; y_img = y_img[joint_valid==1];
    xmin = min(x_img); ymin = min(y_img); xmax = max(x_img); ymax = max(y_img);

    x_center = (xmin+xmax)/2.; width = xmax-xmin;
    xmin = x_center - 0.5*width*1.5
    xmax = x_center + 0.5*width*1.5
    
    y_center = (ymin+ymax)/2.; height = ymax-ymin;
    ymin = y_center - 0.5*height*1.5
    ymax = y_center + 0.5*height*1.5

    # if cfg.cfg.use_s2hand:
    #     a = max(height, width) # make square box
    #     a = a * 9 / 4 # swollen the box to S2HAND typical ratio
    #     # xmax = x_center + a / 2; xmin = x_center - a / 2
    #     # ymax = y_center + a / 2; ymin = y_center - a / 2
    #     bbox = np.array([x_center, y_center, a]).astype(np.float32)
    # else:
    a = max(height, width) # make square box
    # a = a * 9 / 4 # swollen the box to S2HAND typical ratio
    a = a * 1.5
    xmax = x_center + a / 2; xmin = x_center - a / 2
    ymax = y_center + a / 2; ymin = y_center - a / 2
    bbox = np.array([xmin, ymin, a, a]).astype(np.float32)

    return bbox, x_img, y_img

def generate_patch_image(cvimg, bbox, do_flip, scale, rot, out_shape):
    img = cvimg.copy()
    img_height, img_width, img_channels = img.shape

    bb_c_x = float(bbox[0] + 0.5*bbox[2])
    bb_c_y = float(bbox[1] + 0.5*bbox[3])
    bb_width = float(bbox[2])
    bb_height = float(bbox[3])

    if do_flip:
        img = img[:, ::-1, :]
        bb_c_x = img_width - bb_c_x - 1
    
    trans = gen_trans_from_patch_cv(bb_c_x, bb_c_y, bb_width, bb_height, out_shape[1], out_shape[0], scale, rot)
    img_patch = cv2.warpAffine(img, trans, (int(out_shape[1]), int(out_shape[0])), flags=cv2.INTER_LINEAR)
    img_patch = img_patch.astype(np.float32)
    inv_trans = gen_trans_from_patch_cv(bb_c_x, bb_c_y, bb_width, bb_height, out_shape[1], out_shape[0], scale, rot, inv=True)

    return img_patch, trans, inv_trans


def rotate_2d(pt_2d, rot_rad):
    x = pt_2d[0]
    y = pt_2d[1]
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)
    xx = x * cs - y * sn
    yy = x * sn + y * cs
    return np.array([xx, yy], dtype=np.float32)

def gen_trans_from_patch_cv(c_x, c_y, src_width, src_height, dst_width, dst_height, scale, rot, inv=False):
    # augment size with scale
    src_w = src_width * scale
    src_h = src_height * scale
    src_center = np.array([c_x, c_y], dtype=np.float32)

    # augment rotation
    rot_rad = np.pi * rot / 180
    src_downdir = rotate_2d(np.array([0, src_h * 0.5], dtype=np.float32), rot_rad)
    src_rightdir = rotate_2d(np.array([src_w * 0.5, 0], dtype=np.float32), rot_rad)

    dst_w = dst_width
    dst_h = dst_height
    dst_center = np.array([dst_w * 0.5, dst_h * 0.5], dtype=np.float32)
    dst_downdir = np.array([0, dst_h * 0.5], dtype=np.float32)
    dst_rightdir = np.array([dst_w * 0.5, 0], dtype=np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = src_center
    src[1, :] = src_center + src_downdir
    src[2, :] = src_center + src_rightdir

    dst = np.zeros((3, 2), dtype=np.float32)
    dst[0, :] = dst_center
    dst[1, :] = dst_center + dst_downdir
    dst[2, :] = dst_center + dst_rightdir
    
    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans

def trans_point2d(pt_2d, trans):
    src_pt = np.array([pt_2d[0], pt_2d[1], 1.]).T
    dst_pt = np.dot(trans, src_pt)
    return dst_pt[0:2]

def rot_point2d(pt_2d, trans):
    src_pt = np.array([pt_2d[0], pt_2d[1]]).T
    dst_pt = np.dot(trans[:2, :2], src_pt)
    return dst_pt

def get_aug_config():
    trans_factor = 0.15
    scale_factor = 0 #0.15
    rot_factor = 45
    color_factor = 0.2
    
    trans = [np.random.uniform(-trans_factor, trans_factor), np.random.uniform(-trans_factor, trans_factor)]
    scale = np.clip(np.random.randn(), -1.0, 1.0) * scale_factor + 1.0
    rot = np.clip(np.random.randn(), -2.0,
                  2.0) * rot_factor if random.random() <= 0.6 else 0
    #do_flip = random.random() <= 0.5
    c_up = 1.0 + color_factor
    c_low = 1.0 - color_factor
    color_scale = np.array([random.uniform(c_low, c_up), random.uniform(c_low, c_up), random.uniform(c_low, c_up)])

    return trans, scale, rot, color_scale

def augmentation(img, bbox, joint_coord, joint_valid, hand_type, mode, joint_type):
    img = img.copy(); 
    joint_coord = joint_coord.copy(); 
    hand_type = hand_type.copy();

    original_img_shape = img.shape
    joint_num = len(joint_coord)
    
    if mode == 'train' and not cfg.cfg.mesh_refine:
        trans, scale, rot, color_scale = get_aug_config()
    else:
        trans, scale, rot, color_scale = [0,0], 1.0, 0.0, np.array([1,1,1])
    if hand_type[0] == 0 and hand_type[1] == 1:
        do_flip = True
    else:
        do_flip = False
    
    bbox[0] = bbox[0] + bbox[2] * trans[0]
    bbox[1] = bbox[1] + bbox[2] * trans[1]
    img, trans, inv_trans = generate_patch_image(img, bbox, do_flip, scale, rot, cfg.cfg.input_img_shape)
    img = np.clip(img * color_scale[None,None,:], 0, 255)
    if cfg.cfg.crop_projection:
        if do_flip:
            joint_coord[:,0] = original_img_shape[1] - joint_coord[:,0] - 1
            joint_coord[joint_type['right']], joint_coord[joint_type['left']] = joint_coord[joint_type['left']].copy(), joint_coord[joint_type['right']].copy()
            joint_valid[joint_type['right']], joint_valid[joint_type['left']] = joint_valid[joint_type['left']].copy(), joint_valid[joint_type['right']].copy()
            hand_type[0], hand_type[1] = hand_type[1].copy(), hand_type[0].copy()
        for i in range(joint_num):
            joint_coord[i,:2] = trans_point2d(joint_coord[i,:2], trans)
        for i in range(22):
            joint_valid[i] = joint_valid[i] * (joint_coord[i,0] >= 0) * (joint_coord[i,0] < cfg.cfg.input_img_shape[1]) * (joint_coord[i,1] >= 0) * (joint_coord[i,1] < cfg.cfg.input_img_shape[0])
    else:
        if do_flip:
            joint_coord[:,0] = -joint_coord[:,0]
            joint_coord[joint_type['right']], joint_coord[joint_type['left']] = joint_coord[joint_type['left']].copy(), joint_coord[joint_type['right']].copy()
            joint_valid[joint_type['right']], joint_valid[joint_type['left']] = joint_valid[joint_type['left']].copy(), joint_valid[joint_type['right']].copy()
            hand_type[0], hand_type[1] = hand_type[1].copy(), hand_type[0].copy()
        for i in range(joint_num):
            joint_coord[i,:2] = rotate_2d(joint_coord[i,:2], np.pi * rot / 180)
            

    return img, joint_coord, joint_valid, hand_type, trans, inv_trans

def load_skeleton_ih26(path, joint_num):

    # load joint info (name, parent_id)
    skeleton = [{} for _ in range(joint_num)]
    with open(path) as fp:
        for line in fp:
            if line[0] == '#': continue
            splitted = line.split(' ')
            joint_name, joint_id, joint_parent_id = splitted
            joint_id, joint_parent_id = int(joint_id), int(joint_parent_id)
            skeleton[joint_id]['name'] = joint_name
            skeleton[joint_id]['parent_id'] = joint_parent_id
    # save child_id
    for i in range(len(skeleton)):
        joint_child_id = []
        for j in range(len(skeleton)):
            if skeleton[j]['parent_id'] == i:
                joint_child_id.append(j)
        skeleton[i]['child_id'] = joint_child_id
    
    return skeleton

def load_skeleton(path, joint_num):

    # load joint info (name, parent_id)
    skeleton = [{} for _ in range(joint_num)]
    with open(path) as fp:
        for line in fp:
            if line[0] == '#': continue
            splitted = line.split(' ')
            joint_name, joint_id, joint_parent_id = splitted
            joint_id, joint_parent_id = int(joint_id), int(joint_parent_id)
            skeleton[joint_id]['name'] = joint_name
            skeleton[joint_id]['parent_id'] = joint_parent_id
                
            if joint_name.endswith('null'):
                skeleton[joint_id]['DoF'] = np.array([0,0,0],dtype=np.float32)
            elif joint_name.endswith('3'):
                skeleton[joint_id]['DoF'] = np.array([0,0,1],dtype=np.float32)
            elif joint_name.endswith('2'):
                skeleton[joint_id]['DoF'] = np.array([0,0,1],dtype=np.float32)
            elif joint_name.endswith('1'):
                skeleton[joint_id]['DoF'] = np.array([1,1,1],dtype=np.float32)
            elif joint_name.endswith('thumb0'):
                skeleton[joint_id]['DoF'] = np.array([1,1,1],dtype=np.float32)
            else:
                skeleton[joint_id]['DoF'] = np.array([0,0,0],dtype=np.float32)

    # save child_id
    for i in range(len(skeleton)):
        joint_child_id = []
        for j in range(len(skeleton)):
            if skeleton[j]['parent_id'] == i:
                joint_child_id.append(j)
        skeleton[i]['child_id'] = joint_child_id
    
    return skeleton

def transform_input_to_output_space(joint_coord, joint_valid, rel_root_depth, root_valid, root_joint_idx, joint_type):
    # transform to output heatmap space
    joint_coord = joint_coord.copy(); joint_valid = joint_valid.copy()
    
    joint_coord[:,0] = joint_coord[:,0] / cfg.cfg.input_img_shape[1] * cfg.cfg.output_hm_shape[2]
    joint_coord[:,1] = joint_coord[:,1] / cfg.cfg.input_img_shape[0] * cfg.cfg.output_hm_shape[1]
    joint_coord[joint_type['right'],2] = joint_coord[joint_type['right'],2] - joint_coord[root_joint_idx['right'],2]
    joint_coord[joint_type['left'],2] = joint_coord[joint_type['left'],2] - joint_coord[root_joint_idx['left'],2]
  
    joint_coord[:,2] = (joint_coord[:,2] / (cfg.cfg.bbox_3d_size/2) + 1)/2. * cfg.cfg.output_hm_shape[0]
    joint_valid = joint_valid * ((joint_coord[:,2] >= 0) * (joint_coord[:,2] < cfg.cfg.output_hm_shape[0])).astype(np.float32)
    rel_root_depth = (rel_root_depth / (cfg.cfg.bbox_3d_size_root/2) + 1)/2. * cfg.cfg.output_root_hm_shape
    root_valid = root_valid * ((rel_root_depth >= 0) * (rel_root_depth < cfg.cfg.output_root_hm_shape)).astype(np.float32)
    
    return joint_coord, joint_valid, rel_root_depth, root_valid

def process_bbox(bbox): #, original_img_shape):

    # aspect ratio preserving bbox
    w = bbox[2]
    h = bbox[3]
    c_x = bbox[0] + w/2.
    c_y = bbox[1] + h/2.
    aspect_ratio = cfg.cfg.input_img_shape[1]/cfg.cfg.input_img_shape[0]
    if w > aspect_ratio * h:
        h = w / aspect_ratio
    elif w < aspect_ratio * h:
        w = h * aspect_ratio
    bbox[2] = w*1.25
    bbox[3] = h*1.25
    bbox[0] = c_x - bbox[2]/2.
    bbox[1] = c_y - bbox[3]/2.

    return bbox
    
