import trimesh
import pymeshlab

import numpy as np
from pytorch3d.io import load_obj
from pytorch3d.structures import Meshes

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from common.utils.vis import *

def load_mask(path):
    verts, faces, aux = load_obj(path)
    flag= np.zeros(verts.shape[0])
    for face in faces[0]:
        for i in range(3):
            flag[face[i]] = 1
    #print(np.nonzero(flag)[0].shape[0])
    return flag, faces[0].numpy()


def smoothing(verts, mask_face):
    ms = pymeshlab.MeshSet()
    mesh = pymeshlab.Mesh(vertex_matrix=verts.numpy(), face_matrix=mask_face)
    ms.add_mesh(mesh)
    ms.apply_coord_taubin_smoothing(lambda_=0.4, mu=-0.42, stepsmoothnum=12, selected=False)
    new_verts = ms.current_mesh().vertex_matrix()
    return new_verts


if __name__ == "__main__":
    # mask, mask_face = load_mask('assets/mask.obj')
    # mask = np.nonzero(mask)[0].tolist()
    _, mask_face, _ = load_obj('assets/mask.obj')

    # verts, faces, aux = load_obj('output_new/0302_082056_loss_predxgt/vis/3801eval_out_2.obj')
    # mesh = Meshes(verts=verts, faces=mask_face)
    mesh_file = 'output_new/0302_082056_loss_predxgt/vis/3801eval_out_2.obj'
    old_verts, total_faces, aux = load_obj(mesh_file)
    new_verts = smoothing(old_verts, mask_face[0].numpy())
    #ms.load_new_mesh(mesh_file)
    #ms.load_new_mesh('assets/mask.obj')
    # for i in mask:
    #     ms.compute_selection_by_condition_per_vertex(condselect='(vi==i)')
    # ms.apply_coord_laplacian_smoothing(stepsmoothnum=2, boundry = True, cotangentweight = True, selected=True)
    # old_verts = ms.current_mesh().vertex_matrix()
    # total_faces = ms.current_mesh().face_matrix()
    # faces = 
    # mesh = pymeshlab.Mesh(vertex_matrix=old_verts.numpy(), face_matrix=total_faces.numpy())
    # ms.ad
    save_obj('smooth.obj', new_verts, total_faces[0].numpy())
    #ms.save_current_mesh("smooth.obj")
    #mesh = trimesh.load_mesh('output_new/0302_082056_loss_predxgt/vis/3801eval_out_2.obj')
    
    pass
