import json
import numpy as np
from numpy.lib.function_base import diff
from pytorch3d.ops.laplacian_matrices import laplacian
import torch
from torch.nn.parallel.data_parallel import DataParallel
import torch.utils.data
import cv2
from glob import glob
import os.path as osp
import sys
import os

from torch.utils.data.dataloader import DataLoader
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import pickle
from PIL import Image, ImageDraw
import random
#from pytorch3d.datasets.shapenet_base import ShapeNetBase
from pytorch3d.io import load_ply
from pytorch3d.structures import Meshes
from pytorch3d.loss import mesh_laplacian_smoothing
from pytorch3d.datasets import collate_batched_meshes_tyluan
from net_prepare.util_regi.chamfer import chamfer_distance, chamfer_distance_test, _handle_pointcloud_input, _validate_chamfer_reduction_inputs
from net_prepare.util_regi.laplacian import mesh_laplacian_smoothing
from net_prepare.util_regi.sff_distance import sffd
from torchvision.transforms.functional import resized_crop
import torchvision.transforms as transfrom
from main.model_s2hand.utils.hand_3d_model import rodrigues, rot_pose_beta_to_mesh
from main.model_s2hand.utils.fh_utils import dhm2ManoS2HAND
from common.utils.preprocessing import load_skeleton
from common.utils.vis import *
from torch.autograd import Variable
from torch.nn import Parameter
from torch.nn.functional import pad
from tqdm import tqdm


from manopth.manolayer import ManoLayer
from pytorch3d.ops.knn import knn_gather, knn_points

batch_size = 64
scale = 0
mano_path = 'assets/mano/subdivision_model_%d' % scale
test_path = 'data/DeepHandMesh/annotations/mano_para/mano_para_raw_obj'


class Dataset(torch.utils.data.Dataset):
    def __init__(self, input_path, stage):
        self.mano_path = input_path
        self.gt_mesh_path = 'data/DeepHandMesh/annotations/3D_scans_decimated/subject_4'
        self.joint_path ='data/DeepHandMesh/annotations/keypoints/subject_4'
        self.data_list = []
        self.stage = stage
        for mano_file in os.listdir(self.mano_path):
            if mano_file.endswith('.json'):
                file_num = int(mano_file.split('/')[-1][:-5])
                gt_mesh_file = "{:06d}.ply".format(file_num)
                joint_file = "keypoints{:04d}".format(file_num) + '.pts'
                skeleton = load_skeleton(osp.join('data/DeepHandMesh/hand_model', 'skeleton.txt'), 22)
                joint_world, joint_valid = self.load_joint_coord(osp.join(self.joint_path, joint_file), 'right', skeleton)
                joint_world = dhm2ManoS2HAND(joint_world)
                joint_valid = dhm2ManoS2HAND(joint_valid)
                self.data_list.append({'mano':mano_file, 'gt': gt_mesh_file, 'num': file_num, 'joint_world': joint_world, 'joint_valid': joint_valid})

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

    def __len__(self):
        return len(self.data_list)
    def __getitem__(self, index):
        data = self.data_list[index]
        mano_path = data['mano']
        if self.stage >= 3:
            gt_mesh_path = data['gt']
            gt_mesh_path = osp.join(self.gt_mesh_path, gt_mesh_path)
            verts, faces = load_ply(gt_mesh_path)
            gt_mesh = Meshes(verts=[verts], faces=[faces])

        with open(osp.join(self.mano_path, mano_path)) as f:
            mano = json.load(f)
            theta = np.array(mano['theta'], dtype=np.float32)
            beta = np.array(mano['beta'], dtype=np.float32)
            trans = np.array(mano['trans'], dtype=np.float32)
            rot = np.array(mano['rot'], dtype=np.float32)
            scale = np.array(mano['scale'], dtype=np.float32)
            camrot = np.array(mano['camrot'], dtype=np.float32)
            if 'diff_v' in mano:
                diff_v = np.array(mano['diff_v'], dtype=np.float32)
            else:
                diff_v = []
            if 'diff_v0' in mano:
                diff_v0 = np.array(mano['diff_v0'], dtype=np.float32)
            else:
                diff_v0 = []

        if self.stage >= 3:
            return theta, beta, trans, rot, scale, camrot, data['num'], \
                data['joint_world'], data['joint_valid'], gt_mesh, diff_v, diff_v0
        else:
            return theta, beta, trans, rot, scale, camrot, data['num'], \
                data['joint_world'], data['joint_valid']

class Model(torch.nn.Module):
    def __init__(self, stage, theta, beta, trans, rot, scale, mano_path):
        super(Model, self).__init__()
        self.trans = Parameter(torch.zeros(trans.shape, dtype=torch.float), requires_grad=True)
        self.theta = Parameter(torch.zeros(theta.shape, dtype=torch.float), requires_grad=True)
        self.beta = Parameter(torch.zeros(beta.shape, dtype=torch.float), requires_grad=True)
        self.rot = Parameter(torch.zeros(rot.shape, dtype=torch.float), requires_grad=True)
        self.scale = Parameter(torch.zeros(scale.shape, dtype=torch.float), requires_grad=True)
        self.stage = stage
        self.mano = ManoLayer(mano_root=mano_path, ncomps=30,  flat_hand_mean=False)
        if self.stage >= 3:
            self.diff_v = Parameter(torch.zeros((self.trans.shape[0], self.mano.th_v_template.shape[1], 3), dtype=torch.float), requires_grad=True)
            self.diff_v0 = Parameter(torch.zeros((self.trans.shape[0], self.mano.th_v_template.shape[1], 3), dtype=torch.float), requires_grad=False)
        self.mano_file = os.path.join(mano_path, 'MANO_RIGHT.pkl')


    def load(self, theta, beta, trans, rot, scale, diff_v=None, diff_v0=None):
        self.theta.data = theta.to(self.theta.device)
        if self.stage > 2:
            self.beta.data = beta.to(self.beta.device)
        if self.stage > 0:
            self.trans.data = trans.to(self.trans.device)
        self.rot.data = rot.to(self.rot.device)
        self.scale.data = scale.to(self.scale.device)
        if diff_v is not None and diff_v != []:
            self.diff_v.data = diff_v.to(self.diff_v.device)
        if diff_v0 is not None and diff_v0 != []:
            self.diff_v0.data = diff_v0.to(self.diff_v0.device)
                #self.diff_v0.data = diff_v.to(self.diff_v0.device)
            #self.diff_v.data += torch.rand(*self.diff_v.shape).to(self.diff_v.device) * 1e-4

    def forward(self, camrot, num, diff0=False):
        if 1: 
            if self.stage >= 3:
                jv, faces, tsa_poses = rot_pose_beta_to_mesh(self.rot, self.theta, self.beta, self.diff_v, mano_file=self.mano_file)#rotation pose shape
            else:
                jv, faces, tsa_poses = rot_pose_beta_to_mesh(self.rot, self.theta, self.beta, mano_file=self.mano_file)#rotation pose shape
            jv_ts = self.trans.unsqueeze(1) + torch.abs(self.scale.unsqueeze(2)) * jv[:,:,:]
            joints = jv_ts[:,0:21] * 1000
            verts = jv_ts[:,21:] * 1000

                # jv_ts_refine = self.trans.unsqueeze(1) + torch.abs(self.scale.unsqueeze(2)) * jv_refine[:,:,:]
                # joints_refine = jv_ts_refine[:,0:21] * 1000
                # verts_refine = jv_ts_refine[:,21:] * 1000
                # joints_refine = torch.matmul(camrot.transpose(1, 2), joints_refine.transpose(1, 2)).transpose(1, 2)
                # verts_refine = torch.matmul(camrot.transpose(1, 2), verts_refine.transpose(1, 2)).transpose(1, 2)

        if 0: #else:
            root_rot = torch.zeros(self.rot.shape, dtype=torch.float).to(self.rot.device)
            th_pose_coeffs = torch.cat([root_rot, self.theta.clone()], dim=1)
            verts, joints = self.mano(th_pose_coeffs, th_betas=self.beta, th_trans=self.trans)
            Rots = rodrigues(self.rot)[0]
            verts = torch.matmul(Rots, verts.permute(0,2,1)).permute(0,2,1) #.contiguous().view(batch_size,-1)
            joints = torch.matmul(Rots, joints.permute(0,2,1)).permute(0,2,1) #.contiguous().view(batch_size,-1)
            faces = self.mano.th_faces

        joints = torch.matmul(camrot.transpose(1, 2), joints.transpose(1, 2)).transpose(1, 2)
        verts = torch.matmul(camrot.transpose(1, 2), verts.transpose(1, 2)).transpose(1, 2)

        if self.stage >= 3:
            verts += self.diff_v0

        # if self.stage >= 3:
        #     return joints, verts, joints_refine, verts_refine
        # else:
        return joints, verts


def train(stage, input_path, output_path, lr, iter_num, loss_weights):
    basic_lr = 2e-4
    os.makedirs(test_path, exist_ok=True)
    os.makedirs(output_path, exist_ok=True)
    dataset = Dataset(input_path, stage)
    print(dataset.__len__())
    data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False, num_workers=8, collate_fn=collate_batched_meshes_tyluan)
    model = None
    optimizer = None
    l2loss = torch.nn.MSELoss(reduction='none')
    
    for _, data in enumerate(data_loader):
        # for i, _ in enumerate(data):
        #     data[i] = data[i].cuda()
        if stage < 3:
            theta, beta, trans, rot, scale, camrot, num, joint_world, joint_valid = data
        else:
            theta, beta, trans, rot, scale, camrot, num, joint_world, joint_valid, gt_mesh, diff_v, diff_v0 = data
        # trans = torch.unsqueeze(trans, dim=1)
        # scale = torch.unsqueeze(scale, dim=2)
        if 0:
            if 3930 not in num.numpy().tolist():
                continue
        camrot = camrot.cuda()
        joint_world = joint_world.cuda()
        joint_valid = joint_valid.cuda()
        joint_valid = torch.unsqueeze(joint_valid, dim=2)
        if stage >= 3:
            if diff_v != []:
                diff_v = diff_v.cuda()
            gt_mesh = gt_mesh.cuda()
            
        #if model is None:
        if stage < 3:
            model = Model(stage, theta, beta, trans, rot, scale, mano_path)
            optimizer = torch.optim.Adam([
                {'params': model.theta, 'lr': 0},
                {'params': model.beta,  'lr': 0},
                {'params': model.trans, 'lr': 0},
                {'params': model.rot,   'lr': 0},
                {'params': model.scale, 'lr': 0},
            ])
        else:
            model = Model(stage, theta, beta, trans, rot, scale, mano_path)
            optimizer = torch.optim.Adam([
                {'params': model.theta, 'lr': 0},
                {'params': model.beta,  'lr': 0},
                {'params': model.trans, 'lr': 0},
                {'params': model.rot,   'lr': 0},
                {'params': model.scale, 'lr': 0},
                {'params': model.diff_v,'lr': 0, 'weight_decay': 1e-3},
            ])
            # a = model.parameters()
            # print('')
            #optimizer = torch.optim.Adam(model.parameters(), lr=1)
        if stage <= 3:
            model.load(theta, beta, trans, rot, scale)
        else:
            model.load(theta, beta, trans, rot, scale, diff_v, diff_v0)
        model = model.cuda()
        model.train()

        faces = model.mano.th_faces
        for k in range(len(optimizer.param_groups)):
            optimizer.param_groups[k]['lr'] = lr[k] * basic_lr
        for step in range(iter_num):
            joints, verts = model(camrot, num)
            if 0:
                num = num.detach().cpu().numpy()
                verts = verts.detach().cpu().numpy()
                faces = faces.detach().cpu().numpy()
                for i in range(joints.shape[0]):
                    save_obj('mesh_mano_%d.obj' % num[i], verts[i], faces)
            if stage < 3:
                loss = torch.sum(l2loss(joint_world, joints) * joint_valid)
                test_losses = {}
                test_losses['joint'] = torch.mean(torch.sqrt(torch.sum(l2loss(joint_world, joints) * joint_valid, dim=2)), dim=1)
            else:
                if step == 0:
                    joints_ori, verts_ori = model(camrot, num, True)
                    joints_ori, verts_ori = joints_ori.detach(), verts_ori.detach()
                    meshes_ori = Meshes(verts_ori, faces.repeat(verts_ori.shape[0], 1, 1))
                    normals_ori = meshes_ori.verts_normals_packed().reshape(verts.shape[0], -1, 3)
                losses = {}; test_losses = {}
                meshes = Meshes(verts, faces.repeat(verts.shape[0], 1, 1))

                sff_distance = sffd(meshes, gt_mesh)
                losses['sffd'] = torch.sum(torch.mean(sff_distance, dim=1))
                test_losses['sffd'] = torch.mean(torch.sqrt(sff_distance), dim=1)


                losses['cd'], _, _, _ = chamfer_distance(meshes.verts_padded(), gt_mesh.verts_padded())
                test_losses['cd'], _, _, _ = chamfer_distance_test(meshes.verts_padded(), gt_mesh.verts_padded(), batch_reduction=None)
                losses['joint'] = torch.sum(torch.mean(l2loss(joint_world, joints) * joint_valid, dim=1))
                test_losses['joint'] = torch.mean(torch.sqrt(torch.sum(l2loss(joint_world, joints) * joint_valid, dim=2)), dim=1)
                losses['laplacian'] = torch.sum(torch.sum(torch.sum(l2loss(mesh_laplacian_smoothing(meshes), mesh_laplacian_smoothing(meshes_ori)), dim=2), dim=1))  # should be big at step 3 and small at step 4
                test_losses['laplacian'] = torch.sum(torch.sum(l2loss(mesh_laplacian_smoothing(meshes), mesh_laplacian_smoothing(meshes_ori)), dim=2), dim=1)
                laplacian = mesh_laplacian_smoothing(meshes) - mesh_laplacian_smoothing(meshes_ori)
                normal_laplacian =  torch.square(torch.sum(normals_ori * laplacian, dim=2))
                losses['normal_laplacian'] = torch.sum(torch.sum(normal_laplacian, dim=1))
                test_losses['normal_laplacian'] = torch.sum(normal_laplacian, dim=1)
                tangent_laplacian = torch.sum(torch.square(torch.cross(normals_ori, laplacian, dim=2)), dim=2)
                losses['tangent_laplacian'] = torch.sum(torch.sum(tangent_laplacian, dim=1))
                test_losses['tangent_laplacian'] = torch.sum(tangent_laplacian, dim=1)

                
                
                losses['tangent'] = torch.sum(torch.mean(torch.norm(torch.cross(normals_ori, verts - verts_ori, dim=2), dim=2), dim=1)) # should be small at stage 3, large at stage 4
                test_losses['tangent'] = torch.mean(torch.norm(torch.cross(normals_ori, verts - verts_ori, dim=2), dim=2), dim=1)
                #losses['tan_lap'] = torch.cross(normals, mesh_laplacian_smoothing(meshes_refine))

                loss = sum(losses[key] * loss_weights[key] for key in losses)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # if step == iter_num - 1 or step % 2 == 0:
            #     for i in range(joints.shape[0]):
            #         print('test error %d, case %d. ' % (step, num[i]), end=' ')
            #         for key in test_losses:
            #             print('%s:%f' % (key, test_losses[key][i] * loss_weights[key]), end=' ')
            #         print('')
            # if step % 20 == 0:
            #     print(step)

        if 0:
            verts = verts.detach().cpu().numpy()
            faces = model.mano.th_faces.detach().cpu().numpy()
            joint_world = joint_world.detach().cpu().numpy()
            joints = joints.detach().cpu().numpy()
            joint_valid = joint_valid.detach().cpu().numpy()
            num = num.detach().cpu().numpy()
            for i in range(verts.shape[0]):
                mesh_out = osp.join(test_path, '{:06d}_mesh.obj'.format(num[i]))
                gt_joints_out = osp.join(test_path, '{:06d}_gt_joints.obj'.format(num[i]))
                joints_out = osp.join(test_path, '{:06d}_joints.obj'.format(num[i]))
                save_obj(mesh_out, verts[i], faces)
                draw_joints_mano_21(gt_joints_out, joint_world[i], joint_valid[i])
                draw_joints_mano_21(joints_out, joints[i], joint_valid[i])
            print('')
    
        for i in range(joints.shape[0]):
            out_dict = {
                'theta': np.squeeze(model.theta[i:i+1].detach().cpu().numpy(), axis=0).tolist(),
                'beta':  np.squeeze(model.beta[i:i+1].detach().cpu().numpy(), axis=0).tolist(),
                'trans': np.squeeze(model.trans[i:i+1].detach().cpu().numpy(), axis=0).tolist(),
                'rot':  np.squeeze(model.rot[i:i+1].detach().cpu().numpy(), axis=0).tolist(),
                'scale':  np.squeeze(model.scale[i:i+1].detach().cpu().numpy(), axis=0).tolist(),
                'camrot':  np.squeeze(camrot[i:i+1].detach().cpu().numpy(), axis=0).tolist(),
            }
            if stage >= 3:
                out_dict['diff_v'] = np.squeeze(model.diff_v[i:i+1].detach().cpu().numpy(), axis=0).tolist()
            out_file = "{:06d}".format(num[i:i+1].item())
            out_file_json = os.path.join(output_path, out_file + '.json')
            out_file_loss = os.path.join(output_path, out_file + '.txt')
            if 0: #os.path.exists(out_file_loss):
                with open(out_file_loss, 'r') as f:
                    error = float(f.readlines()[0].rstrip('\n'))
                    if error <= test_losses['joint'][i].item():
                        continue
            with open(out_file_json, 'w+') as f:
                json.dump(out_dict, f)
            with open(out_file_loss, 'w+') as f:
                print(test_losses['joint'][i].item(), file=f)
                print(test_losses['cd'][i].item(), file=f)
        
        break
        #print('')

def pin(input_path, output_path):

    os.makedirs(test_path, exist_ok=True)
    os.makedirs(output_path, exist_ok=True)
    dataset = Dataset(input_path, 3)
    print(dataset.__len__())
    data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False, num_workers=8, collate_fn=collate_batched_meshes_tyluan)
    with torch.no_grad():
        for _, data in tqdm(enumerate(data_loader)):
            theta, beta, trans, rot, scale, camrot, num, joint_world, joint_valid, gt_mesh, diff_v, diff_v0 = data
            camrot = camrot.cuda()
            joint_world = joint_world.cuda()
            joint_valid = joint_valid.cuda()
            joint_valid = torch.unsqueeze(joint_valid, dim=2)
            gt_mesh_verts = gt_mesh[0].cuda()
            gt_mesh = Meshes(verts=gt_mesh[0], faces=gt_mesh[1])

            model = Model(3, theta, beta, trans, rot, scale, mano_path)
                # a = model.parameters()
                # print('')
                #optimizer = torch.optim.Adam(model.parameters(), lr=1)
            model.load(theta, beta, trans, rot, scale)
            model = model.cuda()
            model.eval()

            faces = model.mano.th_faces
            joints, verts = model(camrot, num)
            if 0:
                num = num.detach().cpu().numpy()
                verts = verts.detach().cpu().numpy()
                faces = faces.detach().cpu().numpy()
                for i in range(joints.shape[0]):
                    #if num[i] in [9082, 1595, 3524, 1099, 3541, 2573, 11153, 2192]:
                    save_obj(os.path.join(output_path, 'mesh_mano_%d.obj' % num[i]), verts[i], faces)
                continue

            meshes = Meshes(verts, faces.repeat(verts.shape[0], 1, 1))
            normals = meshes.verts_normals_padded()

            if 0:
                x, x_lengths, x_normals = _handle_pointcloud_input(verts, None, None)
                y, y_lengths, y_normals = _handle_pointcloud_input(gt_mesh_verts, None, None)

                N, P1, D = x.shape
                P2 = y.shape[1]

                if y.shape[0] != N or y.shape[2] != D:
                    raise ValueError("y does not have the correct shape.")
                x_nn = knn_points(x, y, lengths1=x_lengths, lengths2=y_lengths, K=1)
                idx = x_nn.idx.repeat(1, 1, 3)

            segment_size = 25
            padding_size = int(np.ceil(normals.shape[1] / segment_size) * segment_size - normals.shape[1])
            normals_padded = pad(normals, (0, 0, 0, padding_size, 0, 0))
            normals_padded = normals_padded.reshape(normals.shape[0], segment_size, -1, 3)
            verts_padded = pad(verts, (0, 0, 0, padding_size, 0, 0))
            verts_padded = verts_padded.reshape(verts.shape[0], segment_size, -1, 3)
            idx = []
            for b in range(0, verts.shape[0]):
                idx_b = []
                #gt_mesh_verts_b = torch.FloatTensor(gt_mesh.verts_list()[b]).cuda()
                for i in range(0, segment_size):
                    dv_all = verts_padded[b, i].unsqueeze(1) - gt_mesh_verts[b].unsqueeze(0)
                    cos2_theta = dv_all.mul(normals_padded[b, i].unsqueeze(1))
                    cos2_theta = torch.sum(cos2_theta, dim=-1)
                    cos2_theta = cos2_theta.div(normals_padded[b, i].norm(dim=-1).unsqueeze(-1) + 1e-10)
                    cos2_theta = cos2_theta.div(dv_all.norm(dim=-1) + 1e-10)
                    cos2_theta = cos2_theta.mul(cos2_theta)
                    #sin2_theta = 1 - cos2_theta
                    dv_all = dv_all.norm(dim=-1)
                    dv_all = torch.abs(dv_all.mul(1 - cos2_theta)) + 0.3 * torch.abs(dv_all.mul(cos2_theta))
                    #err = dv_all - dv_all2
                    idx_i = dv_all.argmin(dim=-1)
                    idx_b.append(idx_i)
                idx_b = torch.cat(idx_b, dim=0)
                idx.append(idx_b)
            idx = torch.stack(idx, dim=0)
            idx = idx[:, :verts.shape[1]].unsqueeze(-1).repeat(1, 1, 3)

            nearest_pts = torch.gather(gt_mesh_verts, 1, idx)
            d = nearest_pts - verts

            # model.diff_v0.data = d
            # model.diff_v.data = torch.zeros(d.shape).float().to(d.device)
            # joints, verts2 = model(camrot, num)
            # err = nearest_pts - verts2
            # print('')

            if 0:
                mpve = np.mean(np.linalg.norm(d.detach().cpu().numpy(), axis=1))
                nearest_pts = verts.detach().cpu().numpy()
                faces = model.mano.th_faces.detach().cpu().numpy()
                joint_world = joint_world.detach().cpu().numpy()
                joints = joints.detach().cpu().numpy()
                joint_valid = joint_valid.detach().cpu().numpy()
                num = num.detach().cpu().numpy()
                for i in range(nearest_pts.shape[0]):
                    mesh_out = osp.join(test_path, '{:06d}_mesh.obj'.format(num[i]))
                    gt_joints_out = osp.join(test_path, '{:06d}_gt_joints.obj'.format(num[i]))
                    joints_out = osp.join(test_path, '{:06d}_joints.obj'.format(num[i]))
                    if 1:
                        from scipy import sparse
                        data = []; row = []; col = []
                        for _, triangle in enumerate(faces):
                            for j in range(3):
                                data.append(1.0); row.append(triangle[j]); col.append(triangle[(j + 1) % 3]);
                                data.append(1.0); col.append(triangle[j]); row.append(triangle[(j + 1) % 3]);
                        A = sparse.coo_matrix((data, (row, col)), dtype=float)
                        print('')

                    save_obj(mesh_out, nearest_pts[i], faces)
                    draw_joints_mano_21(gt_joints_out, joint_world[i], joint_valid[i])
                    draw_joints_mano_21(joints_out, joints[i], joint_valid[i])
                print('')
            for i in range(joints.shape[0]):
                out_dict = {
                    'theta': np.squeeze(model.theta[i:i+1].detach().cpu().numpy(), axis=0).tolist(),
                    'beta':  np.squeeze(model.beta[i:i+1].detach().cpu().numpy(), axis=0).tolist(),
                    'trans': np.squeeze(model.trans[i:i+1].detach().cpu().numpy(), axis=0).tolist(),
                    'rot':  np.squeeze(model.rot[i:i+1].detach().cpu().numpy(), axis=0).tolist(),
                    'scale':  np.squeeze(model.scale[i:i+1].detach().cpu().numpy(), axis=0).tolist(),
                    'camrot':  np.squeeze(camrot[i:i+1].detach().cpu().numpy(), axis=0).tolist(),
                    'diff_v0': np.squeeze(d[i:i+1].detach().cpu().numpy(), axis=0).tolist(),
                    'diff_v': np.squeeze(np.zeros(d[i:i+1].shape, dtype=np.float32), axis=0).tolist(),
                    'vertices': np.squeeze(nearest_pts[i:i+1].detach().cpu().numpy(), axis=0).tolist(),
                    'faces': np.squeeze(faces).tolist(),
                }
                out_file = "{:06d}".format(num[i:i+1].item())
                out_file_json = os.path.join(output_path, out_file + '.json')
                with open(out_file_json, 'w+') as f:
                    json.dump(out_dict, f)
            #print('')


def adjust(input_path, output_path):
    os.makedirs(test_path, exist_ok=True)
    os.makedirs(output_path, exist_ok=True)
    dataset = Dataset(input_path, 4)
    print(dataset.__len__())
    data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False, num_workers=8, collate_fn=collate_batched_meshes_tyluan)
    with torch.no_grad():
        for _, data in tqdm(enumerate(data_loader)):
            theta, beta, trans, rot, scale, camrot, num, joint_world, joint_valid, gt_mesh, diff_v, diff_v0 = data
            camrot = camrot.cuda()
            joint_world = joint_world.cuda()
            joint_valid = joint_valid.cuda()
            joint_valid = torch.unsqueeze(joint_valid, dim=2)
            diff_v = diff_v.cuda()
            diff_v0 = diff_v0.cuda()
            gt_mesh_verts = gt_mesh[0].cuda()
            gt_mesh = Meshes(verts=gt_mesh[0], faces=gt_mesh[1])

            model = Model(4, theta, beta, trans, rot, scale, mano_path)
            model.load(theta, beta, trans, rot, scale, diff_v, torch.zeros(diff_v0.shape).float().to(diff_v0.device))
            model = model.cuda()
            model.eval()

            faces = model.mano.th_faces
            joints, verts_ori = model(camrot, num) 
            meshes_ori = Meshes(verts_ori, faces.repeat(verts_ori.shape[0], 1, 1))
            #normals = meshes.verts_normals_padded()
            LV_ori = mesh_laplacian_smoothing(meshes_ori).norm(dim=-1)

            model.diff_v0.data = diff_v0
            joints, verts = model(camrot, num)
            if 1:
                num = num.detach().cpu().numpy()
                verts = verts.detach().cpu().numpy()
                faces = faces.detach().cpu().numpy()
                for i in range(joints.shape[0]):
                    save_obj(os.path.join(output_path, 'mesh_detailed_%d.obj' % num[i]), verts[i], faces)
                continue
            meshes = Meshes(verts, faces.repeat(verts.shape[0], 1, 1))
            #normals = meshes.verts_normals_padded()
            LV = mesh_laplacian_smoothing(meshes).norm(dim=-1)

            DV = model.diff_v.norm(dim=-1)

            LV_diff = LV- LV_ori
            if 1:
                LV_diff = LV_diff.detach().cpu().numpy()
                import matplotlib.pyplot as plt
                plt.figure(figsize=(100, 50))
                plt.plot(LV_diff[0], alpha=0.3, color="b")
                plt.hlines(0, 0, len(LV_diff[0])+1, linewidth=1, color="k" )
                plt.xlim(xmin=0, xmax=len(LV_diff[0]))
                plt.xlabel("Layers")
                plt.ylabel("average gradient")
                plt.title("Gradient flow")
                plt.grid(True)
                plt.savefig('LV_diff.png')

            if 1:
                verts = verts.detach().cpu().numpy()
                faces = model.mano.th_faces.detach().cpu().numpy()
                joint_world = joint_world.detach().cpu().numpy()
                joints = joints.detach().cpu().numpy()
                joint_valid = joint_valid.detach().cpu().numpy()
                num = num.detach().cpu().numpy()
                LV = LV.detach().cpu().numpy()
                DV = DV.detach().cpu().numpy()
                for i in range(verts.shape[0]):
                    
                    gt_joints_out = osp.join(test_path, '{:06d}_gt_joints.obj'.format(num[i]))
                    joints_out = osp.join(test_path, '{:06d}_joints.obj'.format(num[i]))

                    mesh_out_lv = osp.join(test_path, '{:06d}_mesh_lv.obj'.format(num[i]))
                    LV_color = np.abs(LV[i])
                    maxmc = max(LV_color)
                    LV_color = LV_color / maxmc * 255
                    for j in range(LV_color.shape[0]):
                        if LV_color[j] > 255:
                            LV_color[j] = 255
                    LV_color = LV_color.astype(np.uint8)
                    LV_color = np.squeeze(cv2.applyColorMap(LV_color, cv2.COLORMAP_JET))
                    save_obj(mesh_out_lv, verts[i], faces, LV_color)
                    
                    mesh_out_dv = osp.join(test_path, '{:06d}_mesh_dv.obj'.format(num[i]))
                    DV_color = np.abs(DV[i])
                    maxmc = max(DV_color)
                    DV_color = DV_color / maxmc * 255
                    for j in range(DV_color.shape[0]):
                        if DV_color[j] > 255:
                            DV_color[j] = 255
                    DV_color = DV_color.astype(np.uint8)
                    DV_color = np.squeeze(cv2.applyColorMap(DV_color, cv2.COLORMAP_JET))
                    save_obj(mesh_out_dv, verts[i], faces, DV_color)

                    mesh_out = osp.join(test_path, '{:06d}_mesh.obj'.format(num[i]))
                    save_obj(mesh_out, verts[i], faces)
                    draw_joints_mano_21(gt_joints_out, joint_world[i], joint_valid[i])
                    draw_joints_mano_21(joints_out, joints[i], joint_valid[i])
                print('')
            for i in range(joints.shape[0]):
                out_dict = {
                    'theta': np.squeeze(model.theta[i:i+1].detach().cpu().numpy(), axis=0).tolist(),
                    'beta':  np.squeeze(model.beta[i:i+1].detach().cpu().numpy(), axis=0).tolist(),
                    'trans': np.squeeze(model.trans[i:i+1].detach().cpu().numpy(), axis=0).tolist(),
                    'rot':  np.squeeze(model.rot[i:i+1].detach().cpu().numpy(), axis=0).tolist(),
                    'scale':  np.squeeze(model.scale[i:i+1].detach().cpu().numpy(), axis=0).tolist(),
                    'camrot':  np.squeeze(camrot[i:i+1].detach().cpu().numpy(), axis=0).tolist(),
                    'diff_v0': np.squeeze(d[i:i+1].detach().cpu().numpy(), axis=0).tolist()
                }
                out_file = "{:06d}".format(num[i:i+1].item())
                out_file_json = os.path.join(output_path, out_file + '.json')
                with open(out_file_json, 'w+') as f:
                    json.dump(out_dict, f)
            print('')


if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '3'
    path = []
    path.append('data/DeepHandMesh/annotations/mano_para/mano_para_raw') # 0
    path.append('data/DeepHandMesh/annotations/mano_para/mano_para_trans') # 1
    path.append('data/DeepHandMesh/annotations/mano_para/mano_para_pose') # 2
    path.append('data/DeepHandMesh/annotations/mano_para/mano_para_shape') # 3
    path.append(['data/DeepHandMesh/annotations/mano_para/mano_para_pin_0', # 4
    'data/DeepHandMesh/annotations/mano_para/mano_para_pin_1',
    'data/DeepHandMesh/annotations/mano_para/mano_para_pin_2'])
    path.append(['data/DeepHandMesh/annotations/mano_para/mesh_mano_0', # 5
    'data/DeepHandMesh/annotations/mano_para/mesh_mano_1',
    'data/DeepHandMesh/annotations/mano_para/mesh_mano_2'])
    path.append('data/DeepHandMesh/annotations/mano_para/mano_para_adj') # 6
    path.append(['data/DeepHandMesh/annotations/mano_para/mesh_detailed_0', # 7
    'data/DeepHandMesh/annotations/mano_para/mesh_detailed_1',
    'data/DeepHandMesh/annotations/mano_para/mesh_detailed_2'])
    # path.append('data/DeepHandMesh/annotations/mano_para/mano_para_vert_test')
    # path.append('data/DeepHandMesh/annotations/mano_para/mano_para_detail_test')
    # para_list = ['theta', 'beta', 'trans', 'rot', 'scale']
    lr = [
        [0, 0, 10, 0, 0],
        [5, 0, 5, 5, 0],
        [3, 3, 3, 3, 3],
        [0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 1],
    ]
    # loss_list = ['cd', 'joint', 'laplacian', 'tangent']

    loss_weights0 = [
        {},
        {},
        {},
        {'cd': 10, 'joint': 10, 'laplacian': 2e5, 'normal_laplacian': 1e6, 'tangent_laplacian': 0, 'tangent': 0},
        {'cd': 1, 'joint': 1, 'laplacian': 2e4, 'normal_laplacian': 0, 'tangent_laplacian': 1e5, 'tangent': 0},
    ]
    loss_weights1 = [
        {},
        {},
        {},
        {'cd': 10, 'joint': 10, 'laplacian': 5e6, 'normal_laplacian': 1e7, 'tangent_laplacian': 0, 'tangent': 0},
        {'cd': 1, 'joint': 1, 'laplacian': 5e5, 'normal_laplacian': 0, 'tangent_laplacian': 1e6, 'tangent': 0},
    ]

    loss_weights2 = [
        {},
        {},
        {},
        {'cd': 1000, 'joint': 1, 'laplacian': 0, 'normal_laplacian': 1e8, 'tangent_laplacian': 1e10, 'tangent': 0, 'regular': 0, 'sffd': 0},
        {'cd': 1000, 'joint': 1, 'laplacian': 1e9, 'normal_laplacian': 0, 'tangent_laplacian': 0, 'tangent': 0, 'regular': 0, 'sffd': 1000},
    ]
    iter_num = [500, 500, 1, 2000, 200]
    for i in range(3, 3):
        train(i, path[i], path[i+1], lr[i], iter_num[i], loss_weights2[i])

    pin(path[3], path[5][scale])
    #adjust(path[4][scale], path[7][scale])
    
    # i = 4
    # train(i, path[i], path[i+1], lr[i], iter_num[i], loss_weights[i])