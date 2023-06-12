import re
import array
#from geometry import *
import sys
import os
#sys.path.append(os.path.realpath('.'))
sys.path.append(os.path.dirname(os.path.realpath('.')))
from shapes_mano import TriangleMesh
import numpy as np
from common.mano_webuser.smpl_handpca_wrapper_HAND_only import ready_arguments

#from PyQt4 import QtCore

_float_regex = re.compile(r'(?:(?:^|\s))(?P<fnum>[-+]?(\d+(\.\d+(e[+-]\d+)?)?))(?=($|\s))',re.U)
_face_regex = re.compile(r'(?:(?:^|\s))(?P<v>\d+)(/((?P<vt>\d*)/)?(?P<vn>\d+))?(?=($|\s))',re.U)


def _read_mano(filename):
    dd = ready_arguments(filename)
    return TriangleMesh(dd), dd

    

# def _parse_vertex_data(s):
#     matches = _float_regex.finditer(s)
#     coords = map(lambda x:  float(x.group('fnum')), matches)
#     return Vector(*coords)
    

def _parse_face_data(s):
    matches = _face_regex.finditer(s)
    vi = array.array('I')
    vni = array.array('I')
    vti = array.array('I')
    
    for x in matches:
        vi.append(int(x.group('v')) - 1)
            
        if x.group('vt'):
            vti.append(int(x.group('vt')) - 1)
        if x.group('vn'):
            vni.append(int(x.group('vn')) - 1)
    if len(vi) < 3:
        return None
    return _triangulate_face(vi, vni, vti)

def _triangulate_face(vi, vni, vti):
    info = vi.buffer_info()
    if info[1] > 3:
        v1 = vi[0]
        v = array.array('I')
        vn1 = vni[0] if len(vni) else None
        vn = array.array('I')
        vt1 = vti[0] if len(vti) else None
        vt = array.array('I')
        for i in range(1,info[1] - 1):
            v.extend( (v1, vi[i], vi[i+1]) )
            if vn1 != None:
                vn.extend( (vn1, vni[i], vni[i+1]) )
            if vt1!= None:
                vt.extend( (vt1, vti[i], vti[i+1]) )
        return (v, vn, vt)
    return (vi, vni, vti)

def _write_mano(filename, refined_mesh, dd):
    import pickle
    import chumpy as ch
    import numpy as np
    from scipy import sparse
    viNum, v, vi, pd, sd, w, rv, rvw, regs = refined_mesh.viNum, refined_mesh.v, refined_mesh.vi, \
        refined_mesh.posedirs, refined_mesh.shapedirs, refined_mesh.weights, refined_mesh.relative_vertex, \
            refined_mesh.relative_vertex_weight, refined_mesh.regressors

    verts = []
    pds = []
    sds = []
    ws = []
    #regs = []
    for i in range(len(v)):
        verts.append(v[i])
        pds.append(pd[i])
        sds.append(sd[i])
        ws.append(w[i])
        #regs.append(reg[i])
    faces = []
    for i in range(int(viNum / 3)):
        face = np.zeros(3, dtype=np.int)
        face[0] = vi[i * 3]
        face[1] = vi[i * 3 + 1]
        face[2] = vi[i * 3 + 2]
        faces.append(face)
    dd['v_template'] = np.array(verts)
    dd['f'] = np.array(faces)
    dd['posedirs'] = np.array(pds)
    dd['shapedirs'] = np.array(sds)
    dd['weights'] = np.array(ws)
    dd['relative_vertex'] = rv
    dd['relative_vertex_weight'] = rvw
    dd['J_regressor'] = sparse.csr_matrix(regs.T)
    pickle.dump(dd, open(filename, 'wb'))

# def _write_mano_obj(filename, refined_mesh):
#     (viNum, v, vi, pd, sd, w, rv, reg) = refined_mesh
#     with open(filename, "w+") as file:
#         for pos in v:
#             file.write('v %f %f %f\n' % (pos[0], pos[1], pos[2]))
#         for i in range(int(viNum / 3)):
#             file.write('f %d %d %d\n' % (vi[i * 3] + 1, vi[i * 3 + 1] + 1, vi[i * 3 + 2] + 1))

def _write_mano_obj(filename, refined_mesh):
    color = [
        (0, 0, 0), 
        (50, 0, 0), (100, 0, 0), (150, 0, 0), (200, 0, 0), #(250, 0, 0), 
        (0, 60, 0), (0, 120, 0), (0, 180, 0), (0, 240, 0),
        (0, 0, 60), (0, 0, 120), (0, 0, 180), (0, 0, 240),
        (60, 60, 0), (120, 120, 0), (180, 180, 0), (240, 240, 0),
        (0, 60, 60), (0, 120, 120), (0, 180, 180), (0, 240, 240)
    ]
    v, vi, viNum, regs = refined_mesh.v, refined_mesh.vi, refined_mesh.viNum, refined_mesh.regressors
    J = np.matmul(regs.transpose(), v)
    with open(filename, "w+") as file:
        for pos in v:
            file.write('v %f %f %f\n' % (pos[0], pos[1], pos[2]))
        if 0:
            for i in range(J.shape[0]):
                file.write('v %f %f %f %d %d %d\n' % (J[i][0], J[i][1], J[i][2], color[i][0], color[i][1], color[i][2]))
        for i in range(int(viNum / 3)):
            file.write('f %d %d %d\n' % (vi[i * 3] + 1, vi[i * 3 + 1] + 1, vi[i * 3 + 2] + 1))
        

def _write_network_obj(filename, filename_lines, refined_meshes):
    bias = 0.07
    vertex_count = 0
    vertex_count_former_layer = 0
    color = [[1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 1, 1], [0, 0, 1], [1, 0, 1]]
    with open(filename, "w+") as file:
        for j in range(len(refined_meshes)):
            (viNum, v, vi, pd, sd, w, rv, rvw, reg) = refined_meshes[j]
            for pos in v:
                #file.write('v %f %f %f\n' % (pos[0], pos[1] + bias * i, pos[2]))
                file.write('v %f %f %f %d %d %d\n' % (pos[0], pos[1] + bias * j, pos[2], color[j][0], color[j][1], color[j][2]))
            for i in range(int(viNum / 3)):
                file.write('f %d %d %d\n' % (vi[i * 3] + 1 + vertex_count, vi[i * 3 + 1] + 1 + vertex_count, vi[i * 3 + 2] + 1 + vertex_count))

            vertex_count_former_layer = vertex_count
            vertex_count += len(v)

    # line
    sample_number = 15
    vertex_count = 0
    vertex_count_former_layer = 0
    import random
    with open(filename_lines, "w+") as file:
        for j in range(len(refined_meshes)):
            (viNum, v, vi, pd, sd, w, rv, rvw, reg) = refined_meshes[j]
            for pos in v:
                #file.write('v %f %f %f\n' % (pos[0], pos[1] + bias * i, pos[2]))
                file.write('v %f %f %f %d %d %d\n' % (pos[0], pos[1] + bias * j, pos[2], color[j][0], color[j][1], color[j][2]))
            draw_list = list(range(len(rv)))
            random.shuffle(draw_list)
            draw_list = draw_list[:sample_number]
            for i in draw_list:
                for k in rv[i]:
                    file.write('l %d %d\n' % (i + 1 + vertex_count, k + 1 + vertex_count_former_layer))
            vertex_count_former_layer = vertex_count
            vertex_count += len(v)

            count_connection = 0
            for i in range(len(rv)):
                count_connection += len(rv[i])
            print('subdivision time:', j, '  vertex number:', len(v), '  connection number:', count_connection)