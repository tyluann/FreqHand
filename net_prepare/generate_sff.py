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

from curve import curve, sff
import numpy as np
import cv2
import json
from tqdm import tqdm
import trimesh 
from multiprocessing.dummy import Pool as ThreadPool

in_dir = 'data/annotations/3D_scans_decimated/subject_4'
in_dir2 = 'data/annotations/mano_para/mano_para_raw'
out_dir = 'data/annotations/gt_sff'
os.makedirs(out_dir, exist_ok=True)
count = 0

def process_mesh(file):
    print(file, 'begin!')
    mesh_path = os.path.join(in_dir, file)
    mesh = trimesh.load(mesh_path)

    VertexSFM,_,_,Normals=sff(mesh.vertices, mesh.faces, file)
    VertexSFM = [item.tolist() for item in VertexSFM]
    out_dic = {}
    out_dic['sff'] = VertexSFM
    out_dic['normals'] = Normals.tolist()
    out_file = os.path.join(out_dir, file[:-4] + '.json')
    with open(out_file, 'w+') as f:
        json.dump(out_dic, f)
    #print(file)



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='thread')
    parser.add_argument('--tt', type=int, default=22, help='')
    parser.add_argument('--nt', type=int, default=0, help='')
    args = parser.parse_args()

    file_list = os.listdir(in_dir)
    file_needed = os.listdir(in_dir2)
    file_process = []
    for file in file_list:
        if file[:-4] + '.json' in file_needed and file[:-4] + '.json' not in out_dir:
            file_process.append(file)
    for i, file in enumerate(sorted(file_process)):
        if i % args.tt == args.nt:
            #process_mesh(file)
            try:
                process_mesh(file)
            except Exception as e:
                print('Reason:', e)
    # pool.map(process_mesh, file_process)
    # pool.close()
    # pool.join()
