# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import numpy as np
import torch
import torch.utils.data
import cv2
from glob import glob
import os.path as osp
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
#from main.config import cfg
from main import config as cfg
from common.utils.preprocessing import load_img, load_krt, get_bbox, process_bbox, generate_patch_image, gen_trans_from_patch_cv, augmentation, transform_input_to_output_space, load_skeleton
from common.utils.transforms import world2cam, cam2world, cam2pixel, pixel2cam
from common.utils.vis import *
from main.model_s2hand.utils.fh_utils import dhm2ManoS2HAND

import pickle
from PIL import Image, ImageDraw
import random
#from pytorch3d.datasets.shapenet_base import ShapeNetBase
from pytorch3d.io import load_ply
from pytorch3d.structures import Meshes
from torchvision.transforms.functional import resized_crop
import torchvision.transforms as transfrom
import json



class Deephm(torch.utils.data.Dataset):
    mesh_list = {}
    def __init__(self, transform, mode):
        self.mode = mode
        if cfg.cfg.use_shm:
            self.fast_data_dir = '/dev/shm/tyluan'
        else:
            self.fast_data_dir = cfg.cfg.data_dir
        self.data_dir = cfg.cfg.data_dir
        self.img_path = os.path.join(self.fast_data_dir, 'DeepHandMesh/images/subject_' + cfg.cfg.subject)
        self.annot_path = os.path.join(self.fast_data_dir, 'DeepHandMesh/annotations')
        #self.shm_path = '/dev/shm/tyluan/DeepHandMesh/annotations'
        self.hand_model_path = os.path.join(self.data_dir, 'DeepHandMesh/hand_model')
        split = 'train' if self.mode == 'train' or self.mode == 'train_eval' else 'test'
        self.img_path = osp.join(self.img_path, split)
        
        if cfg.cfg.subject == '1':
            self.sequence_names = glob(osp.join(self.img_path, '00*')) # 00: right single hand, 01: left single hand, 02: double hands
        else:
            self.sequence_names = glob(osp.join(self.img_path, '*RT*')) # RT: right single hand, LT: left single hand, DH: double hands
        self.sequence_names = [name for name in self.sequence_names if osp.isdir(name)]
        self.sequence_names = [name for name in self.sequence_names if 'ROM' not in name] # 'ROM' exclude in training
        self.sequence_names = [x.split('/')[-1] for x in self.sequence_names]

        self.krt_path = osp.join(self.annot_path, 'KRT_512') # camera parameters

        
 
        self.joint_num = 22
        self.joint_num_mano = 21
        self.root_joint_idx = 21
        self.align_joint_idx = [8, 12, 16, 20, 21]
        self.non_rigid_joint_idx = [3,21]

        self.joint_type = {'right': np.arange(0,self.joint_num), 'left': np.arange(self.joint_num,self.joint_num*2)}

        self.skeleton = load_skeleton(osp.join(self.hand_model_path, 'skeleton.txt'), self.joint_num)

        self.original_img_shape = (512, 334) # height, width 
        self.transform = transform
        


        # camera load
        krt = load_krt(self.krt_path)
        self.all_cameras = krt.keys()
        self.selected_cameras = [x for x in self.all_cameras if x[:2] == "40"] # 40xx cameras: color cameras
        
        # compute view directions of each camera
        campos = {}
        camrot = {}
        focal = {}
        princpt = {}
        for cam in self.selected_cameras:
            campos[cam] = -np.dot(krt[cam]['extrin'][:3, :3].T, krt[cam]['extrin'][:3, 3]).astype(np.float32)
            camrot[cam] = np.array(krt[cam]['extrin'][:3, :3]).astype(np.float32)
            focal[cam] = krt[cam]['intrin'][:2, :2]
            focal[cam] = np.array([focal[cam][0][0], focal[cam][1][1]]).astype(np.float32)
            princpt[cam] = np.array(krt[cam]['intrin'][:2, 2]).astype(np.float32) 
        self.campos = campos
        self.camrot = camrot
        self.focal = focal
        self.princpt = princpt
        

        self.exception_list = [700, 718, 894, 888, 11062, 11098, 36, 10540, 10534, 10474, 10048, 3162, 3048, 4134, 4200, 11675, 11615]
      
        # get info for all frames       
        self.framelist = []
        for seq_name in self.sequence_names:
            for cam in self.selected_cameras:
                img_path_list = glob(osp.join(self.img_path, seq_name, 'cam' + cam, '*.jpg'))
                for img_path in img_path_list:
                    frame_idx = int(img_path.split('/')[-1][5:-4])
                    if frame_idx in self.exception_list:
                        continue
                    # if frame_idx % 5 != 0:
                    #     continue
                    # if self.mode == 'eval' and frame_idx % 4 != 0:
                    #     continue
                    # if (self.mode == 'train' or self.mode == 'train_eval') and frame_idx % 4 == 0:
                    #     continue

                    # load joint 3D world coordinates
                    joint_path = osp.join(self.annot_path, 'keypoints', 'subject_' + cfg.cfg.subject, "keypoints{:04d}".format(frame_idx) + '.pts')
                    if osp.isfile(joint_path):
                        joint_world, joint_valid = self.load_joint_coord(joint_path, 'right', self.skeleton) # only use right hand
                    else:
                        continue
                    joint_cam = world2cam(joint_world.transpose(1,0), self.camrot[cam], self.campos[cam].reshape(3,1)).transpose(1,0)
                    joint_img = np.vstack(cam2pixel(joint_cam, self.focal[cam], self.princpt[cam])).transpose(1,0)[:,:2]
                    hand_type = 'right'

                    cam_param = {'focal': self.focal[cam], 'princpt': self.princpt[cam], 'campos': self.campos[cam], 'camrot': self.camrot[cam]}
                    joint = {'world_coord': joint_world, 'cam_coord': joint_cam, 'img_coord': joint_img, 'valid': joint_valid}
                    frame = {'img_path': img_path, 'seq_name': seq_name, 'frame': frame_idx, 'cam_param': cam_param, 'joint': joint, 'hand_type': hand_type}
                    self.framelist.append(frame)
                




    def __len__(self):
        return len(self.framelist)
    
    def __getitem__(self, idx):
        # input data
        frame = self.framelist[idx]
        img_path, joint, hand_type, frame_idx = frame['img_path'], frame['joint'], frame['hand_type'], frame['frame']
        joint_world = joint['world_coord'].copy(); joint_cam = joint['cam_coord'].copy()
        joint_img = joint['img_coord'].copy(); joint_valid = joint['valid'].copy()

        if 0:
            joint_cam2 = dhm2ManoS2HAND(joint_cam)
            joint_cam2 = joint_cam2 - joint_cam2[0]
            joint_valid2 = dhm2ManoS2HAND(joint_valid)
            draw_joints_mano_21('joints1.obj', joint_cam2, joint_valid2)

        focal, princpt, campos, camrot = frame['cam_param']['focal'], frame['cam_param']['princpt'], frame['cam_param']['campos'], frame['cam_param']['camrot']
        hand_type = self.handtype_str2array(hand_type)
        joint_coord = np.concatenate((joint_img, joint_cam[:,2,None]),1)

        if cfg.cfg.mesh_refine:
            verts_cams = []; verts_imgs = []; faces = []
            for i in range(3):
                # if cfg.cfg.pre_load_data:
                #     mesh_path = osp.join(self.shm_path, 'mano_para/mano_para_pin_%d' % i, "{:06d}".format(frame_idx) + '.json')
                # else:
                mesh_path = osp.join(self.annot_path, 'mano_para/mano_para_pin_%d' % i, "{:06d}".format(frame_idx) + '.json')
                with open(mesh_path) as f:
                    mano = json.load(f)
                    verts_world = np.array(mano['vertices'], dtype=np.float32)
                    faces.append(np.array(mano['faces'], dtype=np.int32))
                verts_cams.append(world2cam(verts_world.transpose(1,0), camrot, campos.reshape(3,1)).transpose(1,0))
                verts_imgs.append(np.vstack(cam2pixel(verts_cams[i], focal, princpt)).transpose(1,0)[:,:2])
            

            if 0:
                save_obj('mesh.obj', verts.numpy(), faces.numpy())

        # image read
        img = load_img(img_path)
        # bbox calculate
        bbox, x_img, y_img = get_bbox(joint_world, joint_valid, camrot, campos, focal, princpt)
        #img_width, img_height = img.shape[0]
        bbox = process_bbox(bbox) #, (img_height, img_width))
        joint_coord = np.concatenate((joint_img, joint_cam[:,2,None]),1)
        # if cfg.cfg.crop_projection and cfg.cfg.mesh_refine:
        #     verts_coords = []; joint_coords = []
        #     for i in range(3):
        #         verts_coords.append(np.concatenate((verts_imgs[i], verts_cams[i][:,2,None]),1))
        #         verts_coords.append(np.concatenate((joint_coords[i], verts_coords[i]), 0))


        
        if not cfg.cfg.use_s2hand:
            xmin, ymin, xmax, ymax = bbox
            xmin = max(xmin,0); ymin = max(ymin,0); xmax = min(xmax, self.original_img_shape[1]-1); ymax = min(ymax, self.original_img_shape[0]-1);
            bbox = np.array([xmin, ymin, xmax, ymax])
            xmin, ymin, xmax, ymax = bbox
            xmin, xmax = np.array([xmin, xmax])/self.original_img_shape[1]*img.shape[1]; ymin, ymax = np.array([ymin, ymax])/self.original_img_shape[0]*img.shape[0]
            bbox_img = np.array([xmin, ymin, xmax-xmin+1, ymax-ymin+1])
            # for i in range(x_img.shape[0]):
            #     cv2.circle(img, (int(x_img[i]), int(y_img[i])), 2, (255, 0, 0), 1)
            
        #img, delta_c = generate_patch_image(img, bbox_img, False, 1.0, 0.0, cfg.cfg.input_img_shape)
        img, joint_coord, joint_valid, hand_type, trans, inv_trans = augmentation(img, bbox, joint_coord, joint_valid, hand_type, self.mode, self.joint_type)
        # rel_root_depth = np.array([joint_coord[self.root_joint_idx['left'],2] - joint_coord[self.root_joint_idx['right'],2]],dtype=np.float32).reshape(1)
        # root_valid = np.array([joint_valid[self.root_joint_idx['right']] * joint_valid[self.root_joint_idx['left']]],dtype=np.float32).reshape(1) if hand_type[0]*hand_type[1] == 1 else np.zeros((1),dtype=np.float32)
        # # transform to output heatmap space
        # joint_coord, joint_valid, rel_root_depth, root_valid = transform_input_to_output_space(joint_coord, joint_valid, rel_root_depth, root_valid, self.root_joint_idx, self.joint_type)
        
        if cfg.cfg.crop_projection:
            joint['img_coord'] = joint_coord[:self.joint_num, :2]
            joint['cam_coord'] = np.vstack(pixel2cam(joint_coord, focal, princpt)).transpose(1,0)
            joint['cam_coord'] = joint['cam_coord'] - joint['cam_coord'][self.root_joint_idx]
            if cfg.cfg.mesh_refine:
                gt_mesh = joint['cam_coord'][self.joint_num:]
            joint['cam_coord'] = joint['cam_coord'][:self.joint_num]
        else:
            if cfg.cfg.mesh_refine:
                gt_mesh = []
                for i in range(3):
                    gt_mesh.append(verts_cams[i] - joint['cam_coord'][self.root_joint_idx])
            joint['cam_coord'] = joint['cam_coord'] - joint['cam_coord'][self.root_joint_idx]

        joint['world_coord'] = world2cam(joint['cam_coord'].transpose(1,0), camrot, campos.reshape(3,1)).transpose(1,0)
        joint['valid'] = joint_valid
        

        for key in joint:
            joint[key] = dhm2ManoS2HAND(joint[key])

        


        joint['root_joint_coord'] = joint_cam[self.root_joint_idx] #joint['cam_coord'][0]
        

        if 0:
            draw_joints_mano_21('joints2.obj', joint['cam_coord'], joint['valid'])
            save_obj('mesh_gt.obj', gt_mesh, faces)

        # if cfg.cfg.use_s2hand:
        #     img = cv2.resize(img, dsize=(224, 224))
            # f_scale = 224.0 / int(bbox_img[2])

        if 0:
            draw_joints_mano_21("joint_deephm.obj", joint['cam_coord'], joint_valid)
            cv2.imwrite('image_deephm.jpg', img)

        img = self.transform(img.astype(np.float32))/255.



        if not cfg.cfg.use_s2hand: #deephandmesh depth semi data
            target_depthmaps = []; cam_params = []; affine_transes = [];
            for cam in random.sample(self.selected_cameras, cfg.cfg.render_view_num):
                # bbox calculate
                bbox, _, _ = get_bbox(joint_coord, joint_valid, self.camrot[cam], self.campos[cam], self.focal[cam], self.princpt[cam])
                xmin, ymin, xmax, ymax = bbox
                xmin = max(xmin,0); ymin = max(ymin,0); xmax = min(xmax, self.original_img_shape[1]-1); ymax = min(ymax, self.original_img_shape[0]-1);
                bbox = np.array([xmin, ymin, xmax, ymax])

                if self.mode == 'train':
                    # depthmap read
                    depthmap_path = osp.join(self.annot_path, 'depthmaps', 'subject_' + cfg.cfg.subject, "{:06d}".format(frame_idx), 'depthmap' + cam + '.pkl')
                    with open(depthmap_path,'rb') as f:
                        depthmap = pickle.load(f).astype(np.float32)
                    xmin, ymin, xmax, ymax = bbox
                    xmin, xmax = np.array([xmin, xmax])/self.original_img_shape[1]*depthmap.shape[1]; ymin, ymax = np.array([ymin, ymax])/self.original_img_shape[0]*depthmap.shape[0]
                    bbox_depthmap = np.array([xmin, ymin, xmax-xmin+1, ymax-ymin+1])
                    depthmap, delta_c = generate_patch_image(depthmap[:,:,None], bbox_depthmap, False, 1.0, 0.0, cfg.cfg.rendered_img_shape)
                    target_depthmaps.append(self.transform(depthmap))

                xmin, ymin, xmax, ymax = bbox
                affine_transes.append(gen_trans_from_patch_cv((xmin+xmax+1)/2., (ymin+ymax+1)/2., xmax-xmin+1, ymax-ymin+1, cfg.cfg.rendered_img_shape[1], cfg.cfg.rendered_img_shape[0], 1.0, 0.0).astype(np.float32))
                cam_params.append({'camrot': self.camrot[cam], 'campos': self.campos[cam], 'focal': self.focal[cam], 'princpt': self.princpt[cam]})
            inputs = {'img': img}
            if self.mode == 'train':
                targets = {'depthmap': target_depthmaps, 'joint': joint}
            else:
                targets = {'joint': joint}
            meta_info = {'cam_param': cam_params, 'affine_trans': affine_transes}
        if cfg.cfg.use_s2hand:
            # pesudo intrinsic after resizing and croping
            cam = img_path.split('/')[-2][3:]
            f = np.float32(2 * self.focal[cam][0] * trans[0, 0] / joint['root_joint_coord'][2] * 1000 / cfg.cfg.input_img_shape[0])
            c = 2 * joint['img_coord'][0] / cfg.cfg.input_img_shape[0] - 1
            intrinsic = {'f': f, 'c': c}
            # this is for the network, needed the transform
            #intrinsic = np.array([[f[0], 0, c[0]], [0, f[1], c[1]], [0, 0, 1]], dtype=np.float32)
            # this is for GT joints. No need transform
            cam_params = {'camrot': self.camrot[cam], 'campos': self.campos[cam], \
                'focal': self.focal[cam], 'princpt': self.princpt[cam]}



            # image preprocess
            # rot = 0
            # affinetrans, post_rot_trans = handutils.get_affine_transform(
            #     np.asarray([112, 112]), 224, [224, 224], rot=rot
            # )
            # s_img = torch.shape(input_img)
            # img_size = min(s_img[0], s_img[1])
            # img_crop = resized_crop(input_img, int((s_img[0] - img_size)/2), int((s_img[1] - img_size)/2), 224, 224, [224, 224])
            
            inputs = {'img': img, 'intrinsic': intrinsic}
            if 1: # not cfg.cfg.mesh_register:
                targets = {'joint': joint, 'cam_param': cam_params, 'img_id': frame_idx}
                if cfg.cfg.mesh_refine:
                    if 0: #frame_idx not in self.mesh_list:
                        if 0:
                            mesh_path = osp.join(self.annot_path, '3D_scans_decimated', 'subject_' + cfg.cfg.subject, "{:06d}".format(frame_idx) + '.ply')
                            verts, faces = load_ply(mesh_path) 
                            #targets['gt_mesh0'] = Meshes(verts=[verts], faces=[faces])
                        if 0:
                            mesh_path = osp.join(self.annot_path, 'mano_para/mano_para_pin_0', "{:06d}".format(frame_idx) + '.json')
                            with open(mesh_path) as f:
                                mano = json.load(f)
                                verts = torch.from_numpy(np.array(mano['vertices'], dtype=np.float32))
                                faces = torch.from_numpy(np.array(mano['faces'], dtype=np.int32))
                            if 0:
                                save_obj('mesh.obj', verts.numpy(), faces.numpy())
                                
                        self.mesh_list[frame_idx] = Meshes(verts=[verts], faces=[faces])
                    targets['gt_mesh'] = [Meshes(verts=[torch.from_numpy(gt_mesh[i])], faces=[torch.from_numpy(faces[i])]) for i in range(len(gt_mesh))]
                    # targets['vertice'] = vertices_list
                    # targets['sff'] = sff_list
                    # targets['normals'] = normals_list
                    # targets['laplacian'] = laplacian_list
            else:
                # mesh pseudo gt
                mesh_path = osp.join(self.annot_path, '3D_scans_decimated', 'subject_' + cfg.cfg.subject, "{:06d}".format(frame_idx) + '.ply')
                verts, faces = load_ply(mesh_path)
                mesh = Meshes(verts=[verts], faces=[faces])
                targets = {'joint': joint, 'mesh': mesh, 'cam_param': cam_params, 'img_id': frame_idx}
            
            meta_info = {}
      
        return inputs, targets, meta_info

    def handtype_str2array(self, hand_type):
        if hand_type == 'right':
            return np.array([1,0], dtype=np.float32)
        elif hand_type == 'left':
            return np.array([0,1], dtype=np.float32)
        elif hand_type == 'interacting':
            return np.array([1,1], dtype=np.float32)
        else:
            assert 0, print('Not supported hand type: ' + hand_type)

    def load_joint_coord(self, joint_path, hand_type, skeleton):

        # create link between (joint_index in file, joint_name)
        # all the codes use joint index and name of 'skeleton.txt'
        db_joint_name = ['b_r_thumb_null', 'b_r_thumb3', 'b_r_thumb2', 'b_r_thumb1', 'b_r_index_null', 'b_r_index3', 'b_r_index2', 'b_r_index1', 'b_r_middle_null', 'b_r_middle3', 'b_r_middle2', 'b_r_middle1', 'b_r_ring_null', 'b_r_ring3', 'b_r_ring2', 'b_r_ring1', 'b_r_pinky_null', 'b_r_pinky3', 'b_r_pinky2', 'b_r_pinky1', 'b_r_wrist'] # joint names of 'keypointst****.pts'

        # load 3D world coordinates of joints
        joint_world = np.ones((len(skeleton),3),dtype=np.float32)
        joint_valid = np.zeros((len(skeleton)),dtype=np.float32)
        with open(joint_path) as f:
            for line in f:
                parsed_line = line.split()
                parsed_line = [float(x) for x in parsed_line]
                joint_idx, x_world, y_world, z_world, score_sum, num_view = parsed_line
                joint_idx = int(joint_idx) # joint_idx of the file

                if hand_type == 'right' and joint_idx > 20: # 00: right hand, 21~41: left hand
                    continue
                if hand_type == 'left' and joint_idx < 21: # 01: left hand, 0~20: right hand
                    continue
     
                joint_name = db_joint_name[joint_idx]
                joint_idx = [i for i,_ in enumerate(skeleton) if _['name'] == joint_name][0] # joint_idx which follows 'skeleton.txt'
               
                joint_world[joint_idx] = np.array([x_world, y_world, z_world], dtype=np.float32)
                joint_valid[joint_idx] = 1

        return joint_world, joint_valid

if __name__ == "__main__":
    vis_dir = os.path.join('output', 'vis', 'deephm_data')
    os.makedirs(vis_dir, exist_ok=True)
    dataset = Deephm(transfrom.Compose([]), "train")
    print(len(dataset))
    #inputs, targets, meta_info = dataset[0]
    #cv2.imwrite('test_dhm.jpg', inputs['img'])
    from torch.utils.data import DataLoader
    dataloader = DataLoader(dataset=dataset,batch_size=2, shuffle=False)
    for itr, (inputs, targets, meta_info) in enumerate(dataloader):
        if itr % 20 == 0:
            cv2.imwrite(os.path.join(vis_dir, 'train_%06d.jpg' % itr), inputs['img'][0].numpy())
