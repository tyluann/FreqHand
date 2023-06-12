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

import mano
from parsingObj_mano import _read_mano
from subdiv_mano import subdivide
from curve import curve
import numpy as np


if __name__ == '__main__':
    import os

    subdNum = 4
    #mesh = _read_obj_file('subdivision/SampleFiles/bigguy.obj')
    mano_model = 'mano/models/MANO_LEFT.pkl'
    #mano_model = 'MANO_RIGHT.pkl'
    mesh, dd = _read_mano(mano_model)

    mesh_refine = subdivide(mesh, subdNum)

    tmc = []
    for k in range(subdNum + 1):
        vertices = np.stack(mesh_refine[k][1], axis=0)
        faces = np.array(mesh_refine[k][2]).reshape(-1, 3)
        GC,MC,Normals,dS=curve(vertices,faces)
        tmc.append(np.multiply(MC,MC).dot(dS))
        print(tmc[k])
    print(tmc)


    # for k in range(subdNum):
    #     dir = 'mano/subdivision_models_%d' % (k + 1)
    #     os.makedirs(dir, exist_ok=True)
    #     _write_mano(os.path.join(dir, mano_model), mesh_refine[k+1], dd)
    #     _write_mano_obj(os.path.join(dir, 'test.obj'), mesh_refine[k+1])
    # print('')
