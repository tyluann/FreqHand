#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
Copyright (c) 2021 Tianyu Luan. All Rights Reserved.
About this file:
================
This file is to test if curve calculation is invariable to sample rate changes.

'''

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "subdivision"))

# import mano
# from parsingObj_mano import _read_mano
# from subdiv_mano import subdivide
from curve import curve, sff, sff_torch
import numpy as np
import cv2
import torch

def save_obj(file, vertices, faces, curve=None):
    with open(file, 'w+') as f:
        for i in range(vertices.shape[0]):
            if curve is None:
                print('v', vertices[i][0], vertices[i][1], vertices[i][2], file=f)
            else:
                print('v', vertices[i][0], vertices[i][1], vertices[i][2], curve[i][2], curve[i][1], curve[i][0], file=f)
        for i in range(faces.shape[0]):
            print('f', faces[i][0] + 1, faces[i][1] + 1, faces[i][2] + 1, file=f)

def read_obj(file):
    vertices = []
    faces = []
    with open(file, 'r') as f:
        for line in f:
            strs = line.split(' ')
            if strs[0] == 'v':
                vertices.append([float(strs[1]), float(strs[2]), float(strs[3])])
            if strs[0] == 'f':
                faces.append([int(strs[1]) - 1, int(strs[2]) - 1, int(strs[3]) - 1])
    vertices = np.array(vertices)
    faces = np.array(faces)
    return vertices, faces

def read_manos(mano_dir, i):
    file = os.path.join(mano_dir, 'subdivision_model_%d' % i, 'MANO_RIGHT.obj')
    return read_obj(file)

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '3'
    import os
    subdNum = 4
    #mesh = _read_obj_file('subdivision/SampleFiles/bigguy.obj')
    #mano_model = 'mano/models/MANO_LEFT.pkl'
    #mano_model = 'MANO_RIGHT.pkl'
    mano_path = 'asset/mano'
    # mesh, dd = _read_mano(mano_model)
    # mesh_refine = subdivide(mesh, subdNum)

    out_dir = os.path.join(os.path.dirname(__file__), 'vis')
    os.makedirs(out_dir, exist_ok=True)

    maxgc, mingc, maxmc, minmc = 0, 0, 0, 0

    for k in range(subdNum + 1):
        # vertices = np.stack(mesh_refine[k][1], axis=0)
        # faces = np.array(mesh_refine[k][2]).reshape(-1, 3)
        vertices, faces = read_manos(mano_path, k)
        if 0:
            verticesc, facesc = vertices.copy(), faces.copy()
            VertexSFM,GC,MC,Normals=sff(verticesc,facesc)
        else:
            verticesc, facesc = torch.from_numpy(vertices).double().cuda(), torch.from_numpy(faces).cuda()
            verticesc = verticesc.repeat(2, 1, 1); facesc = facesc.repeat(2, 1, 1)
            VertexSFM,GC,MC,Normals=sff_torch(verticesc,facesc)
            #VertexSFM,GC,MC,Normals = VertexSFM.cpu().numpy(),GC.cpu().numpy(),MC.cpu().numpy(),Normals.cpu().numpy()
            VertexSFM,GC,MC,Normals = VertexSFM[1].cpu().numpy(),GC[1].cpu().numpy(),MC[1].cpu().numpy(),Normals[1].cpu().numpy()

        GC = np.sqrt(np.abs(GC))
        if k == 0:
            maxgc = max(GC)
        GC = GC / maxgc * 255
        for i in range(GC.shape[0]):
            if GC[i] > 255:
                GC[i] = 255
        GC = GC.astype(np.uint8)
        GC_color = np.squeeze(cv2.applyColorMap(GC, cv2.COLORMAP_JET))

        MC = np.abs(MC)
        if k == 0:
            maxmc = max(MC)
        MC = MC / maxmc * 255
        for i in range(MC.shape[0]):
            if MC[i] > 255:
                MC[i] = 255
        MC = MC.astype(np.uint8)
        MC_color = np.squeeze(cv2.applyColorMap(MC, cv2.COLORMAP_JET))

        save_obj(os.path.join(out_dir, "sub%d_GC.obj" % k), vertices, faces, GC_color)
        save_obj(os.path.join(out_dir, "sub%d_MC.obj" % k), vertices, faces, MC_color)
        print('')
