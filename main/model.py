import imp
import os
import math
import time
import sys

from functools import partial
import json
import numpy as np
from scipy import sparse
import cv2
import random
import pymeshlab

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.modules.pooling import AvgPool2d
from torch.nn.parameter import Parameter

from pytorch3d.loss import chamfer_distance, point_mesh_face_distance
from pytorch3d.renderer import MeshRasterizer, RasterizationSettings, PerspectiveCameras
from pytorch3d.structures import Meshes
from pytorch3d.ops import cot_laplacian
from pytorch3d.ops.laplacian_matrices import laplacian
from pytorch3d.loss import mesh_laplacian_smoothing
from pytorch3d.renderer.mesh.rasterize_meshes import rasterize_meshes
from pytorch3d.io import load_obj

from common.nets.loss import DepthmapLoss, JointLoss
from common.utils.transforms import world2cam, world2cam_torch
from common.mano_webuser.smpl_handpca_wrapper_HAND_only import ready_arguments
from common.utils.vis import *



from net_prepare.util_regi.chamfer import chamfer_distance, chamfer_distance_test
from net_prepare.util_regi.laplacian import mesh_laplacian_smoothing
from net_prepare.util_regi.sff_distance import sffd

# sys.path.append(os.path.realpath('.'))
#from main.config import cfg
from main import config as cfg
from main.model_s2hand.utils.freihandnet import MyPoseHand, HM2Mano, normal_init#, mesh2poseNet
from main.model_s2hand.utils.net_hg import Net_HM_HG
from main.model_s2hand.utils.hand_3d_model import rot_pose_beta_to_mesh
from main.model_s2hand.util import face_vertices, json_load, compute_uv_from_integral
from main.model_s2hand.efficientnet_pt.model import EfficientNet
from main.model_s2hand.gcn.graph_cnn import GraphCNN



class Args_s2hand():
    def __init__(self):
        self.train_requires = ['images', 'Ks', 'joints', 'open_2dj']
        self.test_requires = ['heatmaps', 'joints', 'verts'] #, 'textures', 'lights']
        self.regress_mode = 'mano'
        self.use_mean_shape = cfg.cfg.use_mean_shape
        self.use_2d_as_attention = False
        self.renderer_mode = None #'NR'
        self.texture_mode = 'surf'
        self.image_size = 224
        self.train_datasets = ['FreiHand']
        self.use_pose_regressor = False
        self.pretrain_segmnet = None
        self.pretrain_model = cfg.cfg.pretrain_model
        self.pretrain_texture_model = None
        self.pretrain_rgb2hm = None

args_s2hand = Args_s2hand()



#from utils.fh_utils import AverageMeter

# encoder efficientnet
class Encoder(nn.Module):
    def __init__(self,version='b3'):
        super(Encoder, self).__init__()
        self.version = version
        if self.version == 'b3':
            if cfg.cfg.use_pretrained_backbone:
                self.encoder = EfficientNet.from_pretrained('efficientnet-b3')
            else:
                self.encoder = EfficientNet.from_name('efficientnet-b3')
            # b3 [1536,7,7]
            self.pool = nn.AvgPool2d(7, stride=1)
        '''
        elif self.version == 'b5':
            self.encoder = EfficientNet.from_pretrained('efficientnet-b5')
            # b5 [2048,7,7]
            self.pool = nn.AvgPool2d(7, stride=1)
        '''
    def forward(self, x):
        features, low_features, low, mid, high = self.encoder.extract_features(x)#[B,1536,7,7] [B,32,56,56]
        features = self.pool(features)
        features = features.view(features.shape[0],-1)##[B,1536]
        return features, low_features, low, mid, high

'''
class Percep_Encoder(nn.Module):
    def __init__(self):
        super(Percep_Encoder, self).__init__()
        self.percep_encoder = EfficientNet.from_pretrained('efficientnet-b0')
    def forward(self, x):
        y = self.percep_encoder(x)
        return y

''' 
def normalize_image(im):
    """
    byte -> float, / pixel_max, - 0.5
    :param im: torch byte tensor, B x C x H x W, 0 ~ 255
    :return:   torch float tensor, B x C x H x W, -0.5 ~ 0.5
    """
    #return ((im.float() / 255.0) - 0.5)
    '''
    :param im: torch byte tensor, B x C x H x W, 0 ~ 1
    :return:   torch float tensor, B x C x H x W, -0.5 ~ 0.5
    '''
    return (im - 0.5)

class RGB2HM(nn.Module):
    def __init__(self):
        super(RGB2HM, self).__init__()
        num_joints = 21
        self.net_hm = Net_HM_HG(num_joints,
                                num_stages=2,
                                num_modules=2,
                                num_feats=256)
    def forward(self, images):
        images = normalize_image(images)
        # 1. Heat-map estimation
        est_hm_list, encoding = self.net_hm(images)
        return est_hm_list, encoding

# FreiHand Decoder
class MyHandDecoder(nn.Module):
    def __init__(self,inp_neurons=1536,use_mean_shape=False):
        super(MyHandDecoder, self).__init__()
        self.hand_decode = MyPoseHand(inp_neurons=inp_neurons,use_mean_shape = use_mean_shape)
        if use_mean_shape:
            print("use mean MANO shape")
        else:
            print("do not use mean MANO shape")
        #self.hand_faces = self.hand_decode.mano_branch.faces

    def forward(self, features):
        #sides = torch.zeros(features.shape[0],1)
        #verts, faces, joints = self.hand_decode(features, Ks)
        '''
        joints, verts, faces, theta, beta = self.hand_decode(features)
        return joints, verts, faces, theta, beta
        '''
        joints, verts, faces, theta, beta, scale, trans, rot, tsa_poses = self.hand_decode(features)
        return joints, verts, faces, theta, beta, scale, trans, rot, tsa_poses

class light_estimator(nn.Module):
    def __init__(self, dim_in=1536):
        super(light_estimator, self).__init__()
        self.fc1 = nn.Linear(dim_in, 256)
        self.fc2 = nn.Linear(256, 11)
    def forward(self, x):
        x = F.relu(self.fc1(x), inplace=True)
        lights = torch.sigmoid(self.fc2(x))
        return lights

class texture_light_estimator(nn.Module):
    def __init__(self, num_channel=32, dim_in=56,mode='surf'):
        super(texture_light_estimator, self).__init__()
        self.base_layers = nn.Sequential(
            nn.Conv2d(32, 48, kernel_size=10, stride=4, padding=1),#[48,13,13]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),#[48,6,6]
            nn.Conv2d(48, 64, kernel_size=3),#[64,4,4]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),#[64,2,2]
        )
        self.texture_reg = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1538*3),
        )
        self.light_reg = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 11),
            #nn.Sigmoid()
        )
        self.mode = mode
        #self.texture_mean = torch.tensor([0.5, 0.5, 0.5])
        self.texture_mean = torch.tensor([200/256, 150/256, 150/256]).float()
        normal_init(self.texture_reg[0],std=0.001)
        normal_init(self.texture_reg[2],std=0.001)
        normal_init(self.light_reg[0],std=0.001)
        normal_init(self.light_reg[2],mean=1,std=0.001)

    def forward(self, low_features):
        base_features = self.base_layers(low_features)#[b,64,2,2]
        base_features = base_features.view(base_features.shape[0],-1)##[B,256]
        # texture
        bias = self.texture_reg(base_features)
        mean_t = self.texture_mean.to(device=bias.device)
        if self.mode == 'surf':
            bias = bias.view(-1, 1538, 3)#[b, 778, 3]
            mean_t = mean_t.unsqueeze(0).unsqueeze(0).repeat(1,bias.shape[1],1)#[1, 778, 3]
        #import pdb; pdb.set_trace()
        textures = mean_t + bias#[b,778,3]
        #import pdb; pdb.set_trace()
        # lighting
        lights = self.light_reg(base_features)#[b,11]
        #import pdb; pdb.set_trace()
        #textures = torch.clamp(textures,0,1)
        #textures = torch.clamp(textures,min=0)
        #lights = torch.clamp(lights,min=0)
        #import pdb; pdb.set_trace()
        return textures, lights

class heatmap_attention(nn.Module):
    def __init__(self, num_channel=256, dim_in=64, out_len=1536, mode='surf'):
        super(heatmap_attention, self).__init__()
        self.base_layers = nn.Sequential(
            nn.BatchNorm2d(num_channel),
            nn.Conv2d(num_channel, 64, kernel_size=10, stride=7, padding=1),#[64,9,9]
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=5, stride=3, padding=1),#[64,3,3]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3),#[64,1,1]
        )
        self.reg = nn.Sequential(
            nn.Linear(64, out_len),
        )
        #import pdb; pdb.set_trace()
    def forward(self, x):
        x0 = self.base_layers(x)
        x0 = x0.view(x.shape[0],-1)#[b,64]
        return self.reg(x0)


class Model_refine(nn.Module):
    def __init__(self):
        super(Model_refine, self).__init__()
        C_feature_map = [384, 96, 24]
        C_gcn_ucon = [61, 29, 29]
        C_gcn_out = [32, 32, 32]
        C_gcn_in = []
        for i,_ in enumerate(C_gcn_ucon):
            if i == 0:
                C_gcn_in.append(C_gcn_ucon[i] + 3)
            else:
                C_gcn_in.append(C_gcn_out[i-1] + C_gcn_ucon[i] + 3)
        mano_path = cfg.cfg.mano_dir
        #self.faces = []; self.A = []; self.up_weight = []

        for i in range(3):
            self.add_module('conv%d' % i, torch.nn.Conv2d(in_channels=C_feature_map[i], out_channels=C_gcn_ucon[i], kernel_size=1, stride=1, padding=0))
            torch.nn.init.kaiming_uniform_(self.get_submodule('conv%d' % i).weight)
            #torch.nn.init.normal_(self.get_submodule('conv%d' % i).weight, mean=0, std=1e-3)
            dd = ready_arguments(os.path.join(mano_path, 'subdivision_model_%d' % (i), 'MANO_RIGHT.pkl'))
            faces = torch.LongTensor(np.array(dd['f'], dtype=np.int32))
            vertices = torch.FloatTensor(np.array(dd['v_template'], dtype=np.int32))
            data = []; row = []; col = []
            for _, triangle in enumerate(faces):
                for j in range(3):
                    data.append(1.0); row.append(triangle[j]); col.append(triangle[(j + 1) % 3]);
                    data.append(1.0); col.append(triangle[j]); row.append(triangle[(j + 1) % 3]);
            # A = sparse.coo_matrix((data, (row, col)), shape=(), dtype=float)
            # values = A.data
            # indices = np.vstack((A.row, A.col))
            # i = torch.LongTensor(indices)
            # v = torch.FloatTensor(values)
            # shape = A.shape
            A = torch.sparse_coo_tensor([row, col], data, (vertices.shape[0], vertices.shape[0])) #.to_dense()

            self.add_module('gcn%d' % i, GraphCNN(A, C_gcn_in[i], C_gcn_out[i]))
            self.avg_pooling = nn.AdaptiveAvgPool2d(1)
            self.max_pooling = nn.AdaptiveMaxPool2d(1)
            
            if i != 0:
                data = []; col = []; row = [];
                # faces_d = self.get_buffer('faces%d' % (i-1))
                for r in range(vertices.shape[0]):
                    for j,vs in enumerate(dd['relative_vertex'][r]):
                        row.append(r)
                        col.append(vs)
                        data.append(dd['relative_vertex_weight'][r][j])
                up_weight = torch.sparse_coo_tensor([row, col], data, (vertices.shape[0], last_n_vertices))
                #self.up_weight = sparse.coo_matrix(dd['upsample_weight'])
                self.register_buffer('up_weight%d' % (i-1), up_weight)
            last_n_vertices = vertices.shape[0]
            self.register_buffer('faces%d' % i, faces)
            #self.register_buffer('A%d' % i, self.A)
            # self.faces.append(faces)
            # self.A.append(A)
            
            #     index = []
            #     for i,vs in enumerate(dd['vertex_list']):

            #     self.up_weight = sparse.csc_matrix(dd['vertex_list'])

            # self.faces.append(faces)
            # self.A.append(A)

        # Projection: perspective cameras
        # Fix f and c, render, and then transfer the (x, y) into origin image coordinate, and then indexing
        self.f0 = 1280; self.r_img_shape = 512; self.z0 = 1;
        self.register_buffer('c0', torch.FloatTensor(np.array([self.r_img_shape // 2, self.r_img_shape // 2])))
        # cameras = PerspectiveCameras(focal_length=self.f0, principal_point=self.c0)
        # raster_settings = RasterizationSettings(image_size=self.r_img_shape)
        #self.rasterizer = MeshRasterizer(cameras=cameras, raster_settings=raster_settings).to(device='cuda:0')




    def indexing(self, feature_map, vertices, faces, fz, cz):
        """
        Inputs:
            feature_map: size = (B, C, H, W)
            vertices: size = (B, V, 3)
            faces: size = (B, F, 3)
            intrisic: size = (B, 3, 3)
            root_joint: size = (B, 3)
        Returns:
            feature_gcn: size = (B, V, C_out_gcn)
        """
        eps = 5e-3
        vertices[:, :, 2] = vertices[:, :, 2] + 100
        vertices = fz.unsqueeze(-1).unsqueeze(-1) * vertices
        vertices[...,0] = vertices[...,0] + cz[:, 0].unsqueeze(-1)
        vertices[...,1] = vertices[...,1] + cz[:, 0].unsqueeze(-1)
        meshes = Meshes(verts=vertices, faces=faces.repeat(vertices.shape[0], 1, 1))

        _, depth_map, _, _ = rasterize_meshes(meshes, faces_per_pixel=1)
        depth_map = torch.flip(depth_map.permute([0,3,1,2]), dims=[2, 3]) #(N, 1, H, W)
        if 0:
            depth_map_vis = depth_map[0, 0].detach().cpu().numpy()
            depth_map_vis = ((depth_map_vis - np.min(depth_map_vis)) / (np.max(depth_map_vis) - np.min(depth_map_vis)) * 255).astype(np.uint8)
            import cv2
            cv2.imwrite('depth_test.jpg', depth_map_vis)
        #depth_map = fragments.zbuf.squeeze()
        # img_coord = (self.f0 * vertices[:, :, :2] + self.c0).div(vertices[:, :, 2:3]) #(N, V, 2)
        # img_coord = (img_coord / self.r_img_shape).unsqueeze(1) # (N, 1, V, 2)
        img_coord = vertices[:, :, :2].unsqueeze(1) # (N, 1, V, 2)
        sampled_depth = F.grid_sample(depth_map, img_coord) # (N, 1, 1, V)
        visible = torch.abs(sampled_depth.squeeze() - vertices[:, :, 2]) < eps  # (N, V)
        # img_coord = fz[:, None, None, :] * img_coord  / self.f0 + c[:, None, None, :] - self.c0[None, None, None, :] # (N, 1, V, 2)
        # if 0:
        #     trans = np.array([[fz/self.f0, 0, (c - self.c)[0]], [0, fz/self.f0, (c - self.c)[1]]] ,dtype=float)
        #     trans = torch.from_numpy(trans).cuda()
        #     grid = F.affine_grid(trans.unsqueeze(0), sampled_depth.unsqueeze(0).size())
        #     output = F.grid_sample(sampled_depth.unsqueeze(0), grid)
        #     cv2.imwrite('test.jpg', output)
        sampled_feature = F.grid_sample(feature_map, img_coord) #(N, C, 1, V)
        sampled_feature = sampled_feature.squeeze(2).multiply(visible.unsqueeze(dim=1)) #(N, C, V)
        return sampled_feature



    def forward(self, feature_maps, vertices, fz, cz):
        """Forward pass
        Inputs:
            vertices: size = (B, V, 3)
            feature_maps: size = [(B, C0, H0, W0), (B, C1, H1, W1), (B, C2, H2, W2)]
        Returns:
            feature_gcn: size = (B, V, C_out_gcn)
        """
        assert len(feature_maps) == 3
        x = None
        # vertices_list: mm
        vertices_list = []; dv_list = []; vertices_ori_list = []
        vertices1 = [torch.matmul(self.get_buffer('up_weight0'), vertices[b]).detach() * 1000 for b in range(vertices.shape[0])]
        vertices1 = torch.stack(vertices1, dim=0)
        vertices2 = [torch.matmul(self.get_buffer('up_weight1'), vertices1[b]).detach() for b in range(vertices1.shape[0])]
        vertices2 = torch.stack(vertices2, dim=0)

        vertices_ori_list.append(vertices2)

        # vertices_ori_list.append(vertices.detach() * 1000)
        # vertices1 = [torch.matmul(self.up_weight0, vertices_ori_list[0][b].detach()) for b in range(vertices.shape[0])]
        # vertices_ori_list.append(torch.stack(vertices1, dim=0))
        # vertices1 = [torch.matmul(self.up_weight1, vertices_ori_list[1][b]) for b in range(vertices.shape[0])]
        # vertices_ori_list.append(torch.stack(vertices1, dim=0))

        for i in range(3):
            #face = self.get_buffer('faces%d' % i)
            if cfg.cfg.feature_sample == 'index':
                feature_maps[i] = self.indexing(feature_maps[i], vertices.clone(), self.get_buffer('faces%d' % i), fz, cz)
            elif cfg.cfg.feature_sample == 'avg_pool':
                feature_maps[i] = self.avg_pooling(feature_maps[i]).squeeze(-1).squeeze(-1).unsqueeze(-1).repeat(1, 1, vertices.shape[1])
            elif cfg.cfg.feature_sample == 'max_pool':
                feature_maps[i] = self.max_pooling(feature_maps[i]).squeeze(-1).squeeze(-1).unsqueeze(-1).repeat(1, 1, vertices.shape[1])
            else:
                feature_maps[i] = self.max_pooling(feature_maps[i]).squeeze(-1).squeeze(-1).unsqueeze(-1).repeat(1, 1, vertices.shape[1])
                feature_maps[i][:,:,:] = 0
            feature_maps[i] = self.get_submodule('conv%d' % i)(feature_maps[i].unsqueeze(-1)).squeeze(-1)
            x, vertices, dv = self.get_submodule('gcn%d' % i, )(feature_maps[i], x, vertices.permute(0, 2, 1))
            vertices_list.append(vertices.transpose(1,2) * 1000)
            # if i == 0:
            #     vertices1 = [torch.matmul(self.get_buffer('up_weight1'), vertices[b].transpose(0, 1)).transpose(0,1).detach() for b in range(vertices.shape[0])]
            #     vertices_ori_list.append(torch.stack(vertices1, dim=0))
            dv_list.append(dv * 1000)
            if i != 2:
                x_b = []; vertices_b = []
                for b in range(x.shape[0]):
                    x_b.append(torch.matmul(self.get_buffer('up_weight%d' % i), x[b].transpose(0, 1)).transpose(0,1))
                    vertices_b.append(torch.matmul(self.get_buffer('up_weight%d' % i), vertices[b].transpose(0, 1)).transpose(0,1))
                    #x = torch.matmul(x.transpose(1,2), self.up_weight).transpose(1,2)
                x = torch.stack(x_b, dim=0)
                vertices = torch.stack(vertices_b, dim=0).transpose(1, 2)
                if i == 0:
                    vertices1 = [torch.matmul(self.get_buffer('up_weight1'), vertices[b]).detach() * 1000 for b in range(vertices.shape[0])]
                    vertices_ori_list.append(torch.stack(vertices1, dim=0))
                if i == 1:
                    vertices_ori_list.append(vertices * 1000)
        return vertices_list, dv_list, vertices_ori_list


if __name__ == "__main__":
    model = Model_refine()


def smoothing(verts, mask_face):
    ms = pymeshlab.MeshSet()
    mesh = pymeshlab.Mesh(vertex_matrix=verts, face_matrix=mask_face)
    ms.add_mesh(mesh)
    ms.apply_coord_taubin_smoothing(lambda_=0.4, mu=-0.42, stepsmoothnum=12, selected=False)
    new_verts = ms.current_mesh().vertex_matrix()
    return new_verts

def subdivison(verts, faces, subnum):
    if subnum == 0:
        return verts, faces
    ms = pymeshlab.MeshSet()
    mesh = pymeshlab.Mesh(vertex_matrix=verts, face_matrix=faces)
    ms.add_mesh(mesh)
    ms.meshing_surface_subdivision_loop(iterations=subnum, threshold=pymeshlab.Percentage(0))
    new_verts = ms.current_mesh().vertex_matrix()
    new_faces = ms.current_mesh().face_matrix()
    return new_verts, new_faces


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.loss_joint_train = torch.nn.MSELoss(reduction='none')
        self.loss_joint_test = torch.nn.MSELoss(reduction='none')
        self.l2loss = torch.nn.MSELoss(reduction='none')

        self.mode = "train"

        if cfg.cfg.mesh_refine:
            self.model_refine = Model_refine()
            for i in range(3):
                 U = np.load("assets/U%d.npy" % i)
                 U = torch.from_numpy(U).float()
                 self.register_buffer('U%d' % i, U)
            # [0].numpy()
            if cfg.cfg.post_smooth:
                _, self.mask_face, _ = load_obj('assets/mask.obj')
            


        # 2D hand estimation
        if 'heatmaps' in args_s2hand.train_requires or 'heatmaps' in args_s2hand.test_requires:
            self.rgb2hm = RGB2HM()

        # 3D hand estimation
        if "joints" in args_s2hand.train_requires or "verts" in args_s2hand.train_requires or "joints" in args_s2hand.test_requires or "verts" in args_s2hand.test_requires:
            self.regress_mode = args_s2hand.regress_mode#options: 'mano' 'hm2mano'
            self.use_mean_shape = args_s2hand.use_mean_shape
            self.use_2d_as_attention = args_s2hand.use_2d_as_attention
            if self.regress_mode == 'mano':# efficient-b3
                self.encoder = Encoder()
                self.dim_in = 1536
                self.hand_decoder = MyHandDecoder(inp_neurons=self.dim_in, use_mean_shape = self.use_mean_shape)
                if self.use_2d_as_attention:
                    self.heatmap_attention = heatmap_attention(out_len=self.dim_in)

            self.render_choice = args_s2hand.renderer_mode
            self.texture_choice = args_s2hand.texture_mode

            # Renderer & Texture Estimation & Light Estimation
            if self.render_choice == 'NR':
                # Define a neural renderer
                import neural_renderer as nr
                if 'lights' in args_s2hand.train_requires or 'lights' in args_s2hand.test_requires:
                    renderer_NR = nr.Renderer(image_size=args_s2hand.image_size,background_color=[1,1,1],camera_mode='projection',orig_size=224,light_intensity_ambient=None, light_intensity_directional=None,light_color_ambient=None, light_color_directional=None,light_direction=None)#light_intensity_ambient=0.9
                else:
                    renderer_NR = nr.Renderer(image_size=args_s2hand.image_size,camera_mode='projection',orig_size=224)
                #import pdb;pdb.set_trace()
                self.renderer_NR = renderer_NR

                '''
                if self.texture_choice == 'surf':
                    self.texture_estimator = TextureEstimator(dim_in=self.dim_in,mode='surfaces')
                elif self.texture_choice == 'nn_same':
                    self.color_estimator = ColorEstimator(dim_in=self.dim_in)
                self.light_estimator = light_estimator(dim_in=self.dim_in)
                '''

                self.texture_light_from_low = texture_light_estimator(mode='surf')
                #[print(aa.requires_grad) for aa in self.encoder.parameters()]
            # Pose adapter
            self.use_pose_regressor = args_s2hand.use_pose_regressor
            if (args_s2hand.train_datasets)[0] == 'FreiHand':
                self.get_gt_depth = True
                self.dataset = 'FreiHand'
            elif (args_s2hand.train_datasets)[0] == 'RHD':
                self.get_gt_depth = False
                self.dataset = 'RHD'
                if self.use_pose_regressor:
                    pass
                    #self.mesh2pose = mesh2poseNet()
            elif (args_s2hand.train_datasets)[0] == 'Obman':
                self.get_gt_depth = False
                self.dataset = 'Obman'
            elif (args_s2hand.train_datasets)[0] == 'HO3D':
                self.get_gt_depth = True
                self.dataset = 'HO3D'
                #Check
            else:
                self.get_gt_depth = False
            
        else:
            self.regress_mode = None
        #import pdb; pdb.set_trace()
        #import numpy as np
        #np.sum([p.numel() for p in model.parameters()]).item()
  

    def predict_singleview(self, images, mask_images, Ks, task, requires, gt_verts, bgimgs):
        vertices, faces, joints, shape, pose, trans, segm_out, textures, lights = None, None, None, None, None, None, None, None, None
        re_images, re_sil, re_img, re_depth, gt_depth = None, None, None, None, None
        pca_text, face_textures = None, None
        output = {}
        # 1. Heat-map estimation
        #end = time.time()
        if self.regress_mode == 'hm2mano' or task == 'hm_train' or 'heatmaps' in requires:
            images_this = images
            if images_this.shape[3] != 256:
                pad = nn.ZeroPad2d(padding=(0,32,0,32))
                #import pdb; pdb.set_trace()
                images_this = pad(images_this)#[b,3,256,256]
            est_hm_list, encoding = self.rgb2hm(images_this)
            
            # est_hm_list: len() 2; [b, 21, 64, 64]
            # this is not well differentiable
            #est_pose_uv = util.compute_uv_from_heatmaps(est_hm_list[-1], images_this.shape[2:4])#images.shape[2:4] torch.Size([224, 224]))  # B x K x 3
            est_pose_uv_list = []
            for est_hm in est_hm_list:
                est_pose_uv = compute_uv_from_integral(est_hm, images_this.shape[2:4])#check
                est_pose_uv_list.append(est_pose_uv)
            
            output['hm_list'] = est_hm_list
            #output['hm_pose_uv'] = est_pose_uv#[b,21,3]
            output['hm_pose_uv_list'] = est_pose_uv_list
            output['hm_j2d_list'] = [hm_pose_uv[:,:,:2] for hm_pose_uv in est_pose_uv_list]
        if task == 'hm_train': 
            #return est_pose_uv, est_hm_list
            return output
        else:
            feature_maps = [0, 0, 0]
            if self.regress_mode == 'hm2mano':
                # 2. Hand shape and pose estimate
                joints, vertices, faces, pose, shape, features = self.hm2hand(est_hm_list, encoding)
                # joints: [b,21,3]; vertices: [b,778,3]; faces: [b,1538,3]; 
                # pose: [b,6]; shape: [b,10]; features: [b,4096]; 
            elif self.regress_mode == 'mano' or self.regress_mode == 'mano1':
                
                features, low_features, feature_maps[0], feature_maps[1], feature_maps[2] = self.encoder(images)#[b,1536]
                if 0:
                    images_vis = images[0].detach().cpu().numpy()
                    # images_vis = ((images_vis - np.min(images_vis)) / (np.max(images_vis) - np.min(images_vis)) * 255).astype(np.uint8)
                    import cv2
                    cv2.imwrite('img_test.jpg', images_vis)
                
                if self.use_2d_as_attention:
                    attention_2d = self.heatmap_attention(encoding[-1])
                    features = torch.mul(features, attention_2d)
                #import pdb; pdb.set_trace()

                #import pdb;pdb.set_trace()
                if 'joints' in requires or 'verts' in requires:
                    #joints, vertices, faces, pose, shape = self.hand_decoder(features)
                    joints, vertices, faces, pose, shape, scale, trans, rot, tsa_poses  = self.hand_decoder(features)
                    out_root_joint = joints[:, 0:1, :]
                    joints = joints - out_root_joint; vertices = vertices - out_root_joint
                    if 0:
                        vertices_np = vertices[0].detach().cpu().numpy()
                        faces_np = faces[0].detach().cpu().numpy()
                        save_obj('mesh.obj', vertices_np, faces_np)
                    if cfg.cfg.mesh_refine:
                        vertices_list, dv_list, vertices_ori_list = self.model_refine(feature_maps, vertices.clone(), Ks['f'], Ks['c'])
                    if self.dataset == 'RHD' and self.use_pose_regressor:
                        joints_res = self.mesh2pose(vertices)
                        #import pdb; pdb.set_trace()
                        joints = joints + joints_res
            #print(time.time() - end)
            #print('Time {batch_time.val:.0f}\t'.format(batch_time))   
            output['joints'] = joints
            output['vertices'] = vertices
            output['pose'] = pose
            output['shape'] = shape
            output['scale'] = scale
            output['trans'] = trans
            output['rot'] = rot
            output['tsa_poses'] = tsa_poses
            if cfg.cfg.mesh_refine:
                output['vertices_list'] = vertices_list
                output['vertices_ori_list'] = vertices_ori_list
                output['dv_list'] = dv_list
            
            #import pdb; pdb.set_trace()
            # 3. Texture & Lighting Estimation
            if 'textures' in requires or 'lights' in requires:
                #low_features.requires_grad = True
                #end = time.time()
                textures, lights = self.texture_light_from_low(low_features)
                #print(time.time() - end)
                #import pdb; pdb.set_trace()
                if 'lights' in requires:                     
                    self.renderer_NR.light_intensity_ambient = lights[:,0].to(vertices.device)
                    self.renderer_NR.light_intensity_directional = lights[:,1].to(vertices.device)
                    self.renderer_NR.light_color_ambient = lights[:,2:5].to(vertices.device)
                    self.renderer_NR.light_color_directional = lights[:,5:8].to(vertices.device)
                    self.renderer_NR.light_direction = lights[:,8:].to(vertices.device)
                    '''
                    self.renderer_NR.light_intensity_ambient = lights[:,0].to(vertices.device)
                    self.renderer_NR.light_color_ambient = torch.ones_like(lights[:,2:5]).to(vertices.device)
                    self.renderer_NR.light_intensity_directional = lights[:,1].to(vertices.device)
                    self.renderer_NR.light_color_directional = torch.ones_like(lights[:,5:8]).to(vertices.device)
                    self.renderer_NR.light_direction = lights[:,8:].to(vertices.device)
                    '''
                output['textures'] = textures
                output['lights'] = lights
                #import pdb;pdb.set_trace()

            #del features
            #import pdb; pdb.set_trace()
            # 4. Render image
            faces = faces.type(torch.int32)
            if self.render_choice == 'NR':
                # use neural renderer
                #I = torch.tensor([[1,0,0,0],[0,1,0,0],[0,0,1,0]]).float()
                #Is = torch.unsqueeze(I,0).repeat(Ks.shape[0],1,1).to(Ks.device)
                # create textures
                if textures is None:
                    texture_size = 1
                    textures = torch.ones(faces.shape[0], faces.shape[1], texture_size, texture_size, texture_size, 3, dtype=torch.float32).to(vertices.device)
                
                self.renderer_NR.R = torch.unsqueeze(torch.tensor([[1,0,0],[0,1,0],[0,0,1]]).float(),0).repeat(Ks.shape[0],1,1).to(vertices.device)
                self.renderer_NR.t = torch.unsqueeze(torch.tensor([[0,0,0]]).float(),0).repeat(Ks.shape[0],1,1).to(vertices.device)
                self.renderer_NR.K = Ks[:,:,:3].to(vertices.device)
                self.renderer_NR.dist_coeffs = self.renderer_NR.dist_coeffs.to(vertices.device)
                #import pdb; pdb.set_trace()
                
                face_textures = textures.view(textures.shape[0],textures.shape[1],1,1,1,3)
                
                re_img,re_depth,re_sil = self.renderer_NR(vertices, faces, torch.tanh(face_textures), mode=None)

                re_depth = re_depth * (re_depth < 1).float()#set 100 into 0

                #import pdb; pdb.set_trace()
                if self.get_gt_depth and gt_verts is not None:
                    gt_depth = self.renderer_NR(gt_verts, faces, mode='depth')
                    gt_depth = gt_depth * (gt_depth < 1).float()#set 100 into 0
                #import pdb; pdb.set_trace()
            
            output['faces'] = faces
            output['re_sil'] = re_sil
            output['re_img'] = re_img
            output['re_depth'] = re_depth
            output['gt_depth'] = gt_depth
            if re_sil is not None:
                output['maskRGBs'] = images.mul((re_sil>0).float().unsqueeze(1).repeat(1,3,1,1))
            output['face_textures'] = face_textures
            #output['render'] = self.renderer_NR
            #output[''] = 
            # Perceptual calculation
            if 'percep_feat' in requires and re_img is not None:
                # only use foreground part
                #import pdb; pdb.set_trace()
                
                #perc_loss = self.perc_crit(torch.mul(images,re_sil.detach().unsqueeze(1)),re_img)
                #perc_features = self.perc_crit.extract_features(torch.mul(images,re_sil.detach().unsqueeze(1)),re_img)
                #in_percep, in_percep_low = self.percep_encoder(torch.mul(images,re_sil.detach().unsqueeze(1)))
                #out_percep, out_percep_low = self.percep_encoder(re_img)
                #perc_features = self.perc_crit(images,re_img)
                perc_loss = self.perc_crit(images,re_img)
                # [a for a in self.perc_crit.model.model1_0.parameters()][0][0]
                #import pdb; pdb.set_trace()
                '''
                iii = 0
                for name,parameters in self.percep_encoder.named_parameters():
                    if iii == 0:
                        print(name,':',parameters.size())
                        print(parameters[0])
                    iii += 1
                #import pdb; pdb.set_trace()
                '''
                #output['in_percep'] = in_percep
                #output['out_percep'] = out_percep
                output['perc_loss'] = perc_loss
                #output['perc_features'] = perc_features
            # Network stacked
            if 'stacked' in requires:
                
                new_img = torch.where(re_img>0,re_img,bgimgs).detach()
                #import pdb;pdb.set_trace()
                
                if self.regress_mode == 'mano':
                    features_s, low_features_s = self.encoder(new_img)#[b,1536]
                    textures_s, lights_s = self.texture_light_from_low(low_features_s)
                    crit = nn.L1Loss()
                    light_s_loss = crit(lights_s, lights.detach())
                    textures_s_loss = crit(textures_s, textures.detach())
                    output['light_s_loss'] = light_s_loss
                    output['textures_s_loss'] = textures_s_loss
                    joints_s, vertices_s, faces_s, pose_s, shape_s, scale_s, trans_s, rot_s, tsa_poses_s  = self.hand_decoder(features_s)
                    crit_mse = nn.MSELoss()
                    joints_s_loss = crit_mse(joints_s, joints.detach())
                    vertices_s_loss = crit_mse(vertices_s, vertices.detach())
                    output['joints_s_loss'] = joints_s_loss
                    output['vertices_s_loss'] = vertices_s_loss
                    output['tsa_poses_s'] = tsa_poses_s
                    faces_s = faces_s.type(torch.int32)
                    face_textures_s = textures_s.view(textures_s.shape[0],textures_s.shape[1],1,1,1,3)
                    re_img_s,re_depth_s,re_sil_s = self.renderer_NR(vertices_s, faces_s, torch.tanh(face_textures_s), mode=None)
                    re_depth_s = re_depth_s * (re_depth_s < 1).float()#set 100 into 0
                    photo_s_loss = crit(re_img_s,re_img.detach())
                    depth_s_loss = crit(re_depth_s,re_depth.detach())
                    sil_s_loss = crit(re_sil_s,re_sil.detach())
                    # ss photometric
                    mask_img_s = images.mul(re_sil_s.unsqueeze(1)).detach()
                    photo_ss_loss = crit(re_img_s,mask_img_s)
                    output['photo_s_loss'] = photo_s_loss
                    output['depth_s_loss'] = depth_s_loss
                    output['sil_s_loss'] = sil_s_loss
                    output['photo_ss_loss'] = photo_ss_loss

                    # feature consistency
                    feature_perceptual_loss = crit(features, features_s)#check detach??
                    output['feature_perceptual_loss'] = feature_perceptual_loss
                    #import pdb; pdb.set_trace()
                    output['vertices_s'] = vertices_s
                    output['joints_s'] = joints_s
                    output['faces_s'] = faces_s
                    output['re_img_s'] = re_img_s
                    output['re_depth_s'] = re_depth_s
                    output['re_sil_s'] = re_sil_s
                    output['new_img'] = new_img
                    output['new_maskimg'] = mask_img_s
            
            '''
            if 'feats' in requires and re_img is not None:
                feats_mask = self.encoder(mask_images)
            #import pdb; pdb.set_trace()
            if 'percep_feat' in requires and re_img is not None:
                #import pdb; pdb.set_trace()
                #images = images.permute(0,2,3,1)*255
                cv_images = torch.zeros_like(images)
                cv_images[:,0,:,:]=images[:,2,:,:]
                cv_images[:,1,:,:]=images[:,1,:,:]
                cv_images[:,2,:,:]=images[:,0,:,:]
                
                #images = re_img.permute(0,2,3,1)*255
                #import pdb; pdb.set_trace()
                re_img_mix = torch.where(re_img>0,re_img,images)
                #re_img_mix = re_img
                cv_re_img = torch.zeros_like(re_img)
                cv_re_img[:,0,:,:]=re_img_mix[:,2,:,:]
                cv_re_img[:,1,:,:]=re_img_mix[:,1,:,:]
                cv_re_img[:,2,:,:]=re_img_mix[:,0,:,:]
                
                in_percep = self.hand_det(cv_images)# cv_images: GBR 0-255 [b,3,224,224]
                #
                out_percep = self.hand_det(cv_re_img)# [b,22,28,28]
                # mse of in-percep and out_percep is e-17!
                #import pdb; pdb.set_trace()
                del cv_images, cv_re_img, images
            '''

            '''
            if task == 'stacked_train':
                return vertices, faces, joints, shape, pose, trans, segm_out, re_sil, re_img, re_depth, gt_depth, features
            elif 'feats' in requires:
                return vertices, faces, joints, shape, pose, trans, segm_out, re_sil, re_img, re_depth, gt_depth, features, feats_mask
            elif 'percep_feat' in requires:
                return vertices, faces, joints, shape, pose, trans, segm_out, re_sil, re_img, re_depth, gt_depth, textures, in_percep, out_percep
            elif 'scalerot' in requires:
                return vertices, faces, joints, shape, pose, trans, segm_out, re_sil, re_img, re_depth, gt_depth, textures, pca_text, scale, rot, tsa_poses
            else:
                return vertices, faces, joints, shape, pose, trans, segm_out, re_sil, re_img, re_depth, gt_depth, textures, pca_text, tsa_poses#, scale, rot
            '''
            return output
    
    

    #def forward(self, images=None, mask_images = None, viewpoints=None, P=None, voxels=None, mano_para = None, task='train', requires=['joints'], gt_verts=None, gt_2d_joints=None, bgimgs=None):  
    def forward(self, inputs, targets, meta_info, mode=None):
        images = inputs['img']
        P = inputs['intrinsic']
        
        mask_images = None
        task, requires, gt_verts, bgimgs, voxels = 'train', args_s2hand.test_requires, None, None, None
        #print(targets['img_id'])
        if 0:
            print(targets['img_id'])
            img = (images[0].permute(1, 2, 0).detach().cpu().numpy() * 255).astype(np.uint8)
            import cv2
            cv2.imwrite('ori_img.jpg', img)

        if task == 'train' or task == 'hm_train':
            res = self.predict_singleview(images, mask_images, P, task, requires, gt_verts, bgimgs)
        elif task == 'stacked_train':
            res = self.stacked_predict_singleview(images, mask_images, P, task, requires, gt_verts, bgimgs)
        elif task == 'test':
            res = self.evaluate_iou(images, voxels)

        #if mode == 'test':
        out = {}
        out['joint_out'] = res['joints'] * 1000
        out_root_joint = out['joint_out'][:, 0:1, :]
        out['joint_out'] = out['joint_out'] - out_root_joint
        out['mesh_out'] = res['vertices'] * 1000 - out_root_joint
        out['faces'] = res['faces']
        #out['scale'] = res['scale']
        

        #targets['joint']['world_coord']
        # joint_cam_gt = torch.matmul(targets['cam_param']['camrot'], \
        #     (targets['joint']['world_coord'] - \
        #         torch.unsqueeze(targets['cam_param']['campos'], dim=1)).permute(0,2,1)).permute(0,2,1)

        joint_cam_gt = targets['joint']['cam_coord']
        joint_valid = torch.unsqueeze(targets['joint']['valid'], dim=2)
        train_loss = {}; test_loss = {}
        train_loss['joint'] = torch.mean(self.loss_joint_train(out['joint_out'], joint_cam_gt) * joint_valid) * 3  # L2 loss
        #train_loss['joint'] = self.loss_joint_train(out['joint_out'], joint_cam_gt) # * joint_valid
        #train_loss['scale'] = ((1 - out['scale']).mul(1 - out['scale']) + (1 - 1/out['scale']).mul(1 - 1/out['scale'])) * cfg.cfg.loss_scale_weight
        if cfg.cfg.mesh_refine:
            out['vertices_list'] = res['vertices_list']
            out['vertices_ori_list'] = res['vertices_ori_list']
            out['dv_list'] = res['dv_list']
            # meshes_ori = Meshes(out['mesh_out'], out['faces'])

            # targets['gt_mesh'][0] = world2cam_torch(targets['gt_mesh'][0].transpose(1,2), 
            #   targets['cam_param']['camrot'], targets['cam_param']['campos'].unsqueeze(-1)).transpose(1,2) - targets['joint']['root_joint_coord'].unsqueeze(1)
            #targets['gt_mesh'][0] -= targets['joint']['root_joint_coord'].unsqueeze(1)
            if 0:
                vertices_np = out['vertices_list'][0][0].detach().cpu().numpy()
                gtv_np = targets['gt_mesh'][0][0].detach().cpu().numpy()
                faces_gt_np = targets['gt_mesh'][1][0].detach().cpu().numpy()
                joint_np = joint_cam_gt[0].detach().cpu().numpy()
                joint_out_np = out['joint_out'][0].detach().cpu().numpy()
                joint_valid_np = joint_valid[0].detach().cpu().numpy()
                #save_obj('mesh_out.obj', vertices_np, faces_gt_np)
                save_obj('mesh_gt.obj', gtv_np, faces_gt_np)
                draw_joints_mano_21('joints_gt.obj', joint_np, joint_valid_np)
                #save_joints('joints_out.obj', joint_out_np, joint_valid_np)

            if cfg.cfg.post_smooth and self.mode == 'test':
                out2 = out['vertices_list'][2].clone()
                new_verts = []
                for i in range(out2.shape[0]):
                    new_verts.append(smoothing(out2[i].detach().cpu().numpy(), self.mask_face[0].detach().cpu().numpy()))
                new_verts = np.stack(new_verts, axis=0)
                out2 = torch.from_numpy(new_verts).float().to(out2.device)
            else:
                out2 = out['vertices_list'][2]
            
            if cfg.cfg.sota:
                freq_gt_sub = torch.matmul(targets['gt_mesh'][2][0].transpose(1,2), self.U2).transpose(1,2)

            # if cfg.cfg.post_smooth:
            #         new_vert = smoothing(out['vertices_list'][2][0].detach().cpu().numpy(), mask_face[0].numpy())
            #         test_loss['mpve_%d' % i] = torch.mean(torch.sqrt(torch.sum((out['vertices_list'][i]- gt_mesh.verts_padded())**2, dim=2)), dim=1)
            for i in range(3):
                gt_mesh = Meshes(targets['gt_mesh'][i][0], targets['gt_mesh'][i][1])
                out['vertices_list'][i] = out['vertices_list'][i] - out_root_joint
                if cfg.cfg.loss_mpve:
                    train_loss['mpve_%d' % i] = torch.mean(self.loss_joint_train(out['vertices_list'][i], gt_mesh.verts_padded()))
                U = self.get_buffer('U%d' % i)
                freq_pred = torch.matmul(out['vertices_list'][i].transpose(1,2), U).transpose(1,2)
                
                if i == 2:
                    freq_pred_test = torch.matmul(out2.transpose(1,2), U).transpose(1,2)
                else:
                    freq_pred_test = freq_pred
                freq_gt = torch.matmul(gt_mesh.verts_padded().transpose(1,2), U).transpose(1,2)
                if cfg.cfg.loss_mpfe_type == 'div_predxgt':
                    train_loss['mpfe_%d' % i] = torch.mean(torch.log2(torch.norm((freq_pred - freq_gt), dim=2)**2
                        / ((torch.norm(freq_pred, dim=2) * torch.norm(freq_gt, dim=2)) + 1e-10) + 1))
                    test_loss['sub_MSNR_%d' % i] =  torch.mean(torch.log((torch.norm(freq_pred, dim=2)
                        / (torch.norm((freq_pred - freq_gt), dim=2) + 1e-8))), dim=1)
                elif cfg.cfg.loss_mpfe_type == 'nolog':
                    train_loss['mpfe_%d' % i] = torch.mean(torch.norm((freq_pred - freq_gt), dim=2)**2
                        / ((torch.norm(freq_pred, dim=2) * torch.norm(freq_gt, dim=2)) + 1e-10))
                    test_loss['sub_MSNR_%d' % i] =  torch.mean(torch.log((torch.norm(freq_pred, dim=2)
                        / (torch.norm((freq_pred - freq_gt), dim=2) + 1e-8))), dim=1)
                elif cfg.cfg.loss_mpfe_type == 'sqr_nolog':
                    train_loss['mpfe_%d' % i] = torch.mean((torch.norm((freq_pred - freq_gt), dim=2)**2
                        / ((torch.norm(freq_pred, dim=2) * torch.norm(freq_gt, dim=2)) + 1e-10))**2)
                    test_loss['sub_MSNR_%d' % i] =  torch.mean(torch.log((torch.norm(freq_pred, dim=2)
                        / (torch.norm((freq_pred - freq_gt), dim=2) + 1e-8))), dim=1)
                elif cfg.cfg.loss_mpfe_type == 'div_pred':
                    train_loss['mpfe_%d' % i] = torch.mean(torch.log2(torch.norm((freq_pred - freq_gt), dim=2) / (torch.norm(freq_pred, dim=2) + 1e-10) + 1))
                    test_loss['sub_MSNR_%d' % i] =  torch.mean(torch.log((torch.norm(freq_pred, dim=2)
                        / (torch.norm((freq_pred - freq_gt), dim=2) + 1e-8))), dim=1)
                elif cfg.cfg.loss_mpfe_type == 'div_gt':
                    train_loss['mpfe_%d' % i] = torch.mean(torch.log2(torch.norm((freq_pred - freq_gt), dim=2) / (torch.norm(freq_gt, dim=2) + 1e-10) + 1))
                    test_loss['sub_MSNR_%d' % i] =  torch.mean(torch.log((torch.norm(freq_pred, dim=2)
                        / (torch.norm((freq_pred - freq_gt), dim=2) + 1e-8))), dim=1)
                elif cfg.cfg.loss_mpfe_type == 'no':
                    pass
                else:
                    meshes = Meshes(out['vertices_list'][i], self.model_refine.get_buffer('faces%d' % i).repeat(out['vertices_list'][i].shape[0], 1, 1))
                    train_loss['laplacian_%d' % i] = torch.sum(torch.sum(torch.sum(self.l2loss(mesh_laplacian_smoothing(meshes), mesh_laplacian_smoothing(gt_mesh)), dim=2), dim=1))
                    test_loss['laplacian_%d' % i] = torch.sum(torch.sum(self.l2loss(mesh_laplacian_smoothing(meshes), mesh_laplacian_smoothing(gt_mesh)), dim=2), dim=1)
                if i==2:
                    test_loss['cd_%d' % i], _, _, _ = chamfer_distance_test(out2, targets['gt_mesh'][2][0], batch_reduction=None)
                else:
                    test_loss['cd_%d' % i], _, _, _ = chamfer_distance_test(out['vertices_list'][i], targets['gt_mesh'][2][0], batch_reduction=None)
                if cfg.cfg.sota:
                    # if i == 0:
                    #     self.up_weight0
                    # if i != 2:
                    #     subnum = 1 if i == 1 else 2
                    #     sub_verts = []
                    #     for _ in range(out['vertices_list'][i].shape[0]):
                    #         self.get_buffer('up_weight%d' % i)
                    #         #sub_vert, sub_faces = subdivison(out['vertices_list'][i][_].detach().cpu().numpy(), targets['gt_mesh'][i][1][_].detach().cpu().numpy(), subnum)
                    #         sub_verts.append(sub_vert)
                    #     sub_verts = np.stack(sub_verts, axis=0)
                    #sub_verts = torch.from_numpy(sub_verts).float().to(device=out['vertices_list'][i].device)
                    freq_pred_sub = torch.matmul(out['vertices_ori_list'][i].transpose(1,2), self.U2).transpose(1,2)

                    test_loss['sub_MSNR_%d' % i] =  torch.mean(torch.log((torch.norm(freq_pred_sub, dim=2) / (torch.norm((freq_pred_sub - freq_gt_sub), dim=2) + 1e-8))), dim=1)
            # if cfg.cfg.sota:
            #     sub_verts = []
            #     for _ in range(out['mesh_out'].shape[0]):
            #         sub_vert, sub_faces = subdivison(out['mesh_out'][_].detach().cpu().numpy(), targets['gt_mesh'][0][1][_].detach().cpu().numpy(), 2)
            #         sub_verts.append(sub_vert)
            #     sub_verts = np.stack(sub_verts, axis=0)
            #     sub_verts = torch.from_numpy(sub_verts).float().to(device=out['mesh_out'].device)
            #     freq_pred_sub = torch.matmul(sub_verts.transpose(1,2), self.U2).transpose(1,2)
            #     test_loss['sub0_MSNR'] =  torch.mean(torch.log2((torch.norm(freq_pred_sub, dim=2) / (torch.norm((freq_pred_sub - freq_gt_sub), dim=2) + 1e-8))), dim=1)
                #print('test')


                #train_loss['joint_%d' % i] = torch.mean(self.loss_joint_train(out['joint_out'], joint_cam_gt) * joint_valid) * 3
                #train_loss['vertices_%d' % i] = torch.mean(self.loss_joint_train(out['vertices_list'][i], gt_mesh.verts_padded()[i])) * 3
                #mesh_laplacian = out['vertices_list'][i] - torch.sparse.mm(self.model_refine.A[i], out['vertices_list'][i])
                #meshes_ori = Meshes(out['vertices_ori_list'][i], self.model_refine.get_buffer('faces%d' % i).repeat(out['vertices_ori_list'][i].shape[0], 1, 1))
                
                
                #train_loss['cd_%d' % i], _, _, _ = chamfer_distance(out['vertices_list'][i], targets['gt_mesh'][0])
                #test_loss['cd_%d' % i], _, _, _ = chamfer_distance_test(out['vertices_list'][i], targets['gt_mesh'][0], batch_reduction=None)
                #out['vertices_list'][i].register_hook(extract)
                
                # sff_distance, curve_distance = sffd(meshes, gt_mesh)
                # train_loss['sffd_%d' % i] = torch.sum(torch.mean(curve_distance, dim=1))
                # test_loss['sffd_%d' % i] = torch.mean(torch.sqrt(sff_distance), dim=1)
                # laplacian = mesh_laplacian_smoothing(meshes) - mesh_laplacian_smoothing(meshes_ori)
                # train_loss['tangent_%d' % i] = torch.mean(self.l2loss(torch.cross(out['dv_list'][i], targets['normals'][i])))
                # train_loss['regular_%d' % i] = torch.mean(out['dv_list'][i].mul(out['dv_list'][i]))
            #train_loss['freq'] = 0 # if an upper layer's frequency 
        if 0:
            draw_joints_mano_21('out.obj', out['joint_out'][0].detach().cpu().numpy())
            draw_joints_mano_21('out.obj', joint_cam_gt[0].detach().cpu().numpy())
        
        #test_loss['joint'] = self.loss_joint_test(out['joint_out']-out['joint_out'][:,:1,:], dhm2Mano(joint_cam_gt-joint_cam_gt[:,-1:,:]))
        test_loss['joint'] = (out['joint_out'] - joint_cam_gt)**2 * joint_valid
        test_loss['joint'] = torch.mean(torch.sqrt(torch.sum(test_loss['joint'], dim=2)), dim=1) # mean of distances

        # test_loss['scale'] = out['scale']
        # if cfg.cfg.mesh_refine:
        #     for i in range(3):
        #         test_loss['cd_%d' % i], _, _, _ = chamfer_distance_test(meshes.verts_padded(), gt_mesh.verts_padded())
        #         sff_distance = sffd(meshes, gt_mesh)
        #         test_loss['sffd_%d' % i] = torch.mean(torch.sqrt(sff_distance))

        

        if cfg.cfg.mesh_register:

            out_dir = os.path.join(cfg.cfg.data_dir, 'annotations', 'mano_para', 'mano_para_raw')
            os.makedirs(out_dir, exist_ok=True)

            for i in range(res['pose'].shape[0]):
                out_file = "{:06d}".format(targets['img_id'][i:i+1].item())
                out_file_json = os.path.join(out_dir, out_file + '.json')
                out_file_loss = os.path.join(out_dir, out_file + '.txt')
                if os.path.exists(out_file_loss):
                    with open(out_file_loss, 'r') as f:
                        error = float(f.readlines()[0].rstrip('\n'))
                        if error <= test_loss['joint'][i:i+1].item():
                            continue

                theta = res['pose'][i:i+1]
                beta = res['shape'][i:i+1]
                trans = res['trans'][i:i+1]
                rot = res['rot'][i:i+1]
                scale = res['scale'][i:i+1]
                camrot = targets['cam_param']['camrot'][i:i+1]

                out_dict = {
                    'theta': np.squeeze(theta.detach().cpu().numpy(), axis=0).tolist(),
                    'beta':  np.squeeze(beta.detach().cpu().numpy(), axis=0).tolist(),
                    'trans': np.squeeze(trans.detach().cpu().numpy(), axis=0).tolist(),
                    'rot':  np.squeeze(rot.detach().cpu().numpy(), axis=0).tolist(),
                    'scale':  np.squeeze(scale.detach().cpu().numpy(), axis=0).tolist(),
                    'camrot':  np.squeeze(camrot.detach().cpu().numpy(), axis=0).tolist(),
                }

                with open(out_file_json, 'w+') as f:
                    json.dump(out_dict, f)
                with open(out_file_loss, 'w+') as f:
                    print(test_loss['joint'][i:i+1].item(), file=f)


        if 0:
            res['pose'] = res['pose'].detach()
            res['shape'] = res['shape'].detach()
            res['trans'] = res['trans'].detach()
            res['rot'] = res['rot'].detach()
            res['scale'] = res['scale'].detach()
            targets['mesh'] = targets['mesh'].detach()
            for i in range(res['pose'].shape[0]):
                theta = Variable(res['pose'][i:i+1], requires_grad=True)
                beta = Variable(res['shape'][i:i+1], requires_grad=True)
                trans = Variable(res['trans'][i:i+1], requires_grad=True)
                rot = Variable(res['rot'][i:i+1], requires_grad=True)
                scale = Variable(res['scale'][i:i+1], requires_grad=True)

                optimizer = torch.optim.Adam([
                    {'params': theta, 'lr': cfg.cfg.lr_register_1 * 1},
                    {'params': beta,  'lr': cfg.cfg.lr_register_1 * 0},
                    {'params': trans, 'lr': cfg.cfg.lr_register_1 * 1},
                    {'params': rot,   'lr': cfg.cfg.lr_register_1 * 1},
                    {'params': scale, 'lr': cfg.cfg.lr_register_1 * 0},
                ])
                optimizer.zero_grad()
                
                # only pose, joint error
                for step in range(500):
                    jv, faces, tsa_poses = rot_pose_beta_to_mesh(rot, theta, beta)#rotation pose shape
                    jv_ts = trans.unsqueeze(1) + torch.abs(scale.unsqueeze(2)) * jv[:,:,:]
                    joints = jv_ts[:,0:21] - jv_ts[:,0:1]
                    verts = jv_ts[:,21:] - jv_ts[:,0:1]
                    loss = torch.mean(self.loss_joint_test(joints * 1000, joint_cam_gt[i:i+1]) * joint_valid[i:i+1])
                    if step % 10 == 0:
                        losses = torch.mean(torch.sqrt(torch.sum(self.loss_joint_test(joints * 1000, joint_cam_gt[i:i+1]) * joint_valid[i:i+1], dim=2)), dim=1)
                        print('test error %d : %f, %f' % (step, losses, scale))
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    #print('error=%f' % (loss))

                for k in range(len(optimizer.param_groups)):
                    optimizer.param_groups[k]['lr'] = cfg.cfg.lr_register_1

                for step in range(5000):
                    jv, faces, tsa_poses = rot_pose_beta_to_mesh(rot, theta, beta)#rotation pose shape
                    jv_ts = trans.unsqueeze(1) + torch.abs(scale.unsqueeze(2)) * jv[:,:,:]
                    joints = jv_ts[:,0:21] - jv_ts[:,0:1]
                    verts = jv_ts[:,21:] - jv_ts[:,0:1]
                    loss = torch.mean(self.loss_joint_test(joints * 1000, joint_cam_gt[i:i+1]) * joint_valid[i:i+1])
                    if step % 10 == 0:
                        losses = torch.mean(torch.sqrt(torch.sum(self.loss_joint_test(joints * 1000, joint_cam_gt[i:i+1]) * joint_valid[i:i+1], dim=2)), dim=1)
                        print('test error %d : %f, %f' % (step, losses, scale))
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                # pose, shape, scale
                for i in range(len(optimizer.param_groups)):
                    optimizer.param_groups[i]['lr'] = cfg.cfg.lr_register_2

                
                for step in range(100):
                    jv, faces, tsa_poses = rot_pose_beta_to_mesh(rot, theta, beta)#rotation pose shape
                    jv_ts = trans.unsqueeze(1) + torch.abs(scale.unsqueeze(2)) * jv[:,:,:]
                    joints = jv_ts[:,0:21]
                    verts = jv_ts[:,21:]
                    loss1 = chamfer_distance(verts, targets['mesh'][i:i+1])
                    loss2 = chamfer_distance(targets['mesh'][i:i+1], verts)
                    loss3 = point_mesh_face_distance(targets['mesh'][i:i+1], verts)
                    print("error=%f, %f" % (loss1, loss2))


        out['joint_gt'] = joint_cam_gt
        out['joint_valid'] = joint_valid
        if cfg.cfg.post_smooth and self.mode == 'test':
            out['vertices_list'][2] = out2

        return out, train_loss, test_loss


def load_model(model):
    current_epoch = 0

    #import pdb; pdb.set_trace()
    if args_s2hand.pretrain_segmnet is not None:
        state_dict = torch.load(args_s2hand.pretrain_segmnet)
        model.seghandnet.load_state_dict(state_dict['seghandnet'])
        #current_epoch = 0
        current_epoch = state_dict['epoch']
        print('loading the model from:', args_s2hand.pretrain_segmnet)
        # logging.info('pretrain_segmentation_model: %s' %args_s2hand.pretrain_segmnet)
    if args_s2hand.pretrain_model is not None:
        state_dict = torch.load(args_s2hand.pretrain_model)
        #import pdb; pdb.set_trace()
        # dir(model)
        if 'encoder' in state_dict.keys() and hasattr(model,'encoder'):
            model.encoder.load_state_dict(state_dict['encoder'])
            print('load encoder')
        if 'decoder' in state_dict.keys() and hasattr(model,'hand_decoder'):
            model.hand_decoder.load_state_dict(state_dict['decoder'])
            print('load hand_decoder')
        if 'heatmap_attention' in state_dict.keys() and hasattr(model,'heatmap_attention'):
            model.heatmap_attention.load_state_dict(state_dict['heatmap_attention'])
            print('load heatmap_attention')
        if 'rgb2hm' in state_dict.keys() and hasattr(model,'rgb2hm'):
            model.rgb2hm.load_state_dict(state_dict['rgb2hm'])
            print('load rgb2hm')
        if 'hm2hand' in state_dict.keys() and hasattr(model,'hm2hand'):
            model.hm2hand.load_state_dict(state_dict['hm2hand'])
        if 'mesh2pose' in state_dict.keys() and hasattr(model,'mesh2pose'):
            model.mesh2pose.load_state_dict(state_dict['mesh2pose'])
            print('load mesh2pose')
        
        if 'percep_encoder' in state_dict.keys() and hasattr(model,'percep_encoder'):
            model.percep_encoder.load_state_dict(state_dict['percep_encoder'])
        
        if 'texture_light_from_low' in state_dict.keys() and hasattr(model,'texture_light_from_low'):
            model.texture_light_from_low.load_state_dict(state_dict['texture_light_from_low'])
            print('load texture_light_from_low')
        if 'textures' in args_s2hand.train_requires and 'texture_estimator' in state_dict.keys():
            if hasattr(model,'renderer'):
                model.renderer.load_state_dict(state_dict['renderer'])
                print('load renderer')
            if hasattr(model,'texture_estimator'):
                model.texture_estimator.load_state_dict(state_dict['texture_estimator'])
                print('load texture_estimator')
            if hasattr(model,'pca_texture_estimator'):
                model.pca_texture_estimator.load_state_dict(state_dict['pca_texture_estimator'])
                print('load pca_texture_estimator')
        if 'lights' in args_s2hand.train_requires and 'light_estimator' in state_dict.keys():
            if hasattr(model,'light_estimator'):
                model.light_estimator.load_state_dict(state_dict['light_estimator'])
                print('load light_estimator')
        print('loading the model from:', args_s2hand.pretrain_model)
        # logging.info('pretrain_model: %s' %args_s2hand.pretrain_model)
        current_epoch = state_dict['epoch']

        if hasattr(model,'texture_light_from_low') and args_s2hand.pretrain_texture_model is not None:
            texture_state_dict = torch.load(args_s2hand.pretrain_texture_model)
            model.texture_light_from_low.load_state_dict(texture_state_dict['texture_light_from_low'])
            print('loading the texture module from:', args_s2hand.pretrain_texture_model)
    # load the pre-trained heat-map estimation model
    if hasattr(model,'rgb2hm') and args_s2hand.pretrain_rgb2hm is not None:
        #util.load_net_model(args_s2hand.pretrain_rgb2hm, model.rgb2hm.net_hm)
        #import pdb; pdb.set_trace()
        hm_state_dict = torch.load(args_s2hand.pretrain_rgb2hm)
        model.rgb2hm.load_state_dict(hm_state_dict['rgb2hm'])
        print('load rgb2hm')
        print('loading the rgb2hm model from:', args_s2hand.pretrain_rgb2hm)
    #import pdb; pdb.set_trace()
    return model, current_epoch