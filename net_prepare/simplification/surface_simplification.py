import time
import os
import copy
import heapq
import sys
from tqdm import tqdm
import numpy as np
import pylab as plt
import trimesh
from scipy import sparse
sys.path.append(os.path.dirname(__file__))
import io_off_model
import mesh_calc
sys.path.append(os.path.realpath('.'))
from shapes_mano import TriangleMesh

# Some flags / constants to define the simplification
CALC_OPTIMUM_NEW_POINT = False # True / False
ENABLE_NON_EDGE_CONTRACTION = False # True / False
CLOSE_DIST_TH = 0.1
SAME_V_TH_FOR_PREPROCESS = 0.001
PRINT_COST = False
SELF_CHECKING = False # True / False
np.set_printoptions(linewidth=200)
ALLOW_CONTRACT_NON_MANIFOLD_VERTICES = False
MINIMUM_NUMBER_OF_FACES = 200

def calc_Q_for_vertex(mesh, v_idx):
  # Calculate K & Q according to eq. (2)
  Q = np.zeros((4, 4))
  for f_idx in np.where(mesh['vf_adjacency_matrix'][v_idx])[0]:
    plane_params = mesh['face_plane_parameters'][f_idx][:, None]
    Kp = plane_params * plane_params.T
    Q += Kp
  return Q

def calc_Q_for_each_vertex(mesh):
  # Prepare some mesh paramenters and run on all vertices to call Q calculation
  mesh['all_v_in_same_plane'] = np.abs(np.diff(mesh['face_plane_parameters'], axis=0)).sum() == 0
  mesh['Qs'] = []
  for v_idx in range(mesh['n_vertices']):
    Q = calc_Q_for_vertex(mesh, v_idx)
    mesh['Qs'].append(Q)

def add_pair(mesh, v1, v2, edge_connection):
  # Do not use vertices on bound or non-manifold ones
  if not ALLOW_CONTRACT_NON_MANIFOLD_VERTICES:
    if not mesh['is_watertight'] and (v1 in mesh['non_maniford_vertices'] or v2 in mesh['non_maniford_vertices']):
      return

  # Add pair of indices to the heap, keys by the cost
  Q = mesh['Qs'][v1] + mesh['Qs'][v2]
  new_v1_ = calc_new_vertex_position(mesh, v1, v2, Q)
  if mesh['all_v_in_same_plane']:
    cost = np.linalg.norm(mesh['vertices'][v1] - mesh['vertices'][v2])
  else:
    new_v1 = np.vstack((new_v1_[:, None], np.ones((1, 1))))
    cost = np.dot(np.dot(new_v1.T, Q), new_v1)[0, 0]
  if PRINT_COST:
    print('For pair: ', v1, ',', v2, ' ; the cost is: ', cost)
  heapq.heappush(mesh['pair_heap'], (cost, v1, v2, edge_connection, new_v1_))

def select_vertex_pairs(mesh):
  print('Calculating pairs cost and add to heap')
  tb = time.time()
  for v1 in tqdm(range(mesh['n_vertices'])):
    for v2 in range(v1 + 1, mesh['n_vertices']):
      edge_connection = mesh['v_adjacency_matrix'][v1, v2]
      vertices_are_very_close = ENABLE_NON_EDGE_CONTRACTION and np.linalg.norm(mesh['vertices'][v2] - mesh['vertices'][v1]) < CLOSE_DIST_TH
      if edge_connection or vertices_are_very_close:
        add_pair(mesh, v1, v2, edge_connection)
  print('time:', time.time() - tb)

def look_for_minimum_cost_on_connected_line():        # TODO
  return None

def calc_new_vertex_position(mesh, v1, v2, Q):
  # Calculating the new vetrex position, given 2 vertices (paragraph 4.):
  # 1. If A (to be defined below) can be inverted, use it
  # 2. If this matrix is not invertible, attempt to find the optimal vertex along the segment V1 and V2
  # 3. The new vertex will be at the midpoint
  A = Q.copy()
  A[3] = [0, 0, 0, 1]                                 # Defined by eq. (1)
  if CALC_OPTIMUM_NEW_POINT:
    A_can_be_ineverted = np.linalg.matrix_rank(A) == 4  # TODO: bug fix!
  else:
    A_can_be_ineverted = False
  if A_can_be_ineverted:
    A_inv = np.linalg.inv(A)
    new_v1 = np.dot(A_inv, np.array([[0, 0, 0, 1]]).T)[:3]
    new_v1 = np.squeeze(new_v1)
  else:
    if CALC_OPTIMUM_NEW_POINT:
      new_v1 = look_for_minimum_cost_on_connected_line()
    else:
      new_v1 = None
    if new_v1 is None:
      new_v1 = (mesh['vertices'][v1] + mesh['vertices'][v2]) / 2

  return new_v1

def contract_best_pair(mesh):
  # Get the best pair of indices from heap, and contract them to a single vertex

  # get pair from heap
  if len(mesh['pair_heap']) == 0:
    return
  cost, v1, v2, is_edge, new_v1 = heapq.heappop(mesh['pair_heap'])

  # no matter how we get new_v1, we represent new_v1 as linear combination of v1, v2, and mean of all v1/v2's neigibers.
  v_neighbors = np.hstack((np.where(mesh['v_adjacency_matrix'][v1])[0], np.where(mesh['v_adjacency_matrix'][v2])[0]))
  position_v3 = np.mean(np.array(mesh['vertices'][v_neighbors]), axis=0)
  weight = np.linalg.inv(np.vstack((mesh['vertices'][v1], mesh['vertices'][v2], position_v3)).T).dot(new_v1)
  # test if the weight is right
  if 0:
    position_v1 = mesh['vertices'][v1]
    position_v2 = mesh['vertices'][v2]
    error = new_v1 - (weight[0] * mesh['vertices'][v1] + weight[1] * mesh['vertices'][v2] + weight[2] * position_v3)
  
  # calculate mano parameters of new vertex using that weight
  mesh['posedirs'][v1] = weight[0] * mesh['posedirs'][v1] + weight[1] * mesh['posedirs'][v2] + \
    weight[2] * np.mean(np.array(mesh['posedirs'][v_neighbors]), axis=0)
  mesh['shapedirs'][v1] = weight[0] * mesh['shapedirs'][v1] + weight[1] * mesh['shapedirs'][v2] + \
    weight[2] * np.mean(np.array(mesh['shapedirs'][v_neighbors]), axis=0)
  mesh['weights'][v1] = weight[0] * mesh['weights'][v1] + weight[1] * mesh['weights'][v2] + \
    weight[2] * np.mean(np.array(mesh['weights'][v_neighbors]), axis=0)
  mesh['J_regressor'][v1] = mesh['J_regressor'][v1] + mesh['J_regressor'][v2]
  mesh['relative_vertex'][v1] = list(set(mesh['relative_vertex'][v1] + mesh['relative_vertex'][v2]))
  mesh['relative_vertex_weight'][v1] = list(set(mesh['relative_vertex_weight'][v1] + mesh['relative_vertex_weight'][v2]))
  mesh['relative_vertex_weight'][v1] = [w/2 for w in mesh['relative_vertex_weight'][v1]]

  # update v1 - position
  mesh['vertices'][v1] = new_v1

  # remove v2:
  mesh['vertices'][v2] = [-1, -1, -1]                 # "remove" vertex from mesh (will be finally removed at function: clean_mesh_from_removed_items)
  mesh['v_adjacency_matrix'][v2, :] = False
  mesh['v_adjacency_matrix'][:, v2] = False
  if is_edge:
    all_v2_faces = np.where(mesh['vf_adjacency_matrix'][v2])[0]
    mesh['vf_adjacency_matrix'][v2, :] = False
    for f in all_v2_faces:
      if v1 in mesh['faces'][f]:                      # If the face contains v2 also share vertex with v1:
        mesh['faces'][f] = [-1, -1, -1]               #  "remove" face from mesh.
        mesh['vf_adjacency_matrix'][:, f] = False
      else:                                           # else:
        v2_idx = np.where(mesh['faces'][f] == v2)[0]  #  replace v2 with v1
        new_v1_nbrs = mesh['faces'][f][mesh['faces'][f] != v2]
        mesh['faces'][f, v2_idx] = v1
        mesh['vf_adjacency_matrix'][v1, f] = True
        mesh['v_adjacency_matrix'][v1, new_v1_nbrs] = True
        mesh['v_adjacency_matrix'][new_v1_nbrs, v1] = True
  else:
    mesh['faces'][mesh['faces'] == v2] = v1
    idxs = np.where(np.sum(mesh['faces'] == v1, axis=1) > 1)[0]
    mesh['faces'][idxs, :] = -1

  mesh['n_faces'] = (mesh['faces'][:, 0] != -1).sum()

  if SELF_CHECKING:
    check_mesh(mesh)

  # remove all v1, v2 pairs from heap (forbidden_vertices can be than removed)
  for pair in mesh['pair_heap'][:]:
    if pair[1] in [v1, v2] or pair[2] in [v1, v2]:
      mesh['pair_heap'].remove(pair)

  # Update Q of vertex v1
  if CALC_OPTIMUM_NEW_POINT:
    update_planes_parameters_near_vertex(mesh, v1)
    calc_Q_for_vertex(mesh, v1)

  # add new pairs of the new vertex
  v2 = None
  for v2_ in range(mesh['n_vertices']):
    if v1 == v2:
      continue
    edge_connection = mesh['v_adjacency_matrix'][v1, v2_]
    vertices_are_very_close = ENABLE_NON_EDGE_CONTRACTION and np.linalg.norm(mesh['vertices'][v2] - mesh['vertices'][v1]) < CLOSE_DIST_TH
    if edge_connection or vertices_are_very_close:
      add_pair(mesh, v1, v2_, edge_connection)

def update_planes_parameters_near_vertex(mesh, v):
  # Get faces near v and recalculate their plane parameters
  mesh_calc.calc_face_plane_parameters(mesh, must_recalc=True)

def check_mesh(mesh):
  # Check that there is no duplicated face
  f_idx_no_erased = np.where(mesh['faces'][:, 1] != -1)[0]
  real_faces = mesh['faces'][f_idx_no_erased]
  if np.unique(real_faces, axis=0).shape != real_faces.shape:
    raise Exception('Duplicated face')
  # Check that adjacent matrices coherent to faces list
  tmp_v_adjacency_matrix = mesh['v_adjacency_matrix'].copy()
  tmp_vf_adjacency_matrix = mesh['vf_adjacency_matrix'].copy()
  for f_idx, f in enumerate(mesh['faces']):
    if f[0] == -1:
      continue
    for v1_, v2_ in [(f[0], f[1]), (f[0], f[2]), (f[1], f[2])]:
      tmp_v_adjacency_matrix[v1_, v2_] = False
      tmp_v_adjacency_matrix[v2_, v1_] = False
      if mesh['v_adjacency_matrix'][v1_, v2_] == False or mesh['v_adjacency_matrix'][v2_, v1_] == False:
        raise Exception('Bad v_adjacency_matrix')
    for v_ in f:
      tmp_vf_adjacency_matrix[v_, f_idx] = False
      if mesh['vf_adjacency_matrix'][v_, f_idx] == False:
        raise Exception('Bad vf_adjacency_matrix')
  if np.any(tmp_vf_adjacency_matrix):
    raise Exception('vf_adjacency_matrix has wrong True elements')
  if np.any(tmp_v_adjacency_matrix):
    raise Exception('v_adjacency_matrix has wrong True elements')

  # Check if a face have 2 same vertex indices
  idxs = np.where(mesh['faces'][:, 0] != -1)[0]
  to_check = mesh['faces'][idxs]
  if np.any(np.diff(np.sort(to_check, axis=1), axis=1) == 0):
    raise Exception('Bug: face found with 2 idintical vertex indices!')


def clean_mesh_from_removed_items(mesh):
  # Remove Faces
  faces2delete = np.where(np.all(mesh['faces'] == -1, axis=1))[0]
  mesh['faces'] = np.delete(mesh['faces'], faces2delete, 0)

  # Check if a face have 2 same vertex indices
  idxs = np.where(mesh['faces'][:, 0] != -1)[0]
  to_check = mesh['faces'][idxs]
  differ = np.diff(np.sort(to_check, axis=1), axis=1) == 0
  faces2delete = np.where(differ)[0]
  if np.size(faces2delete) != 0:
    mesh['faces'] = np.delete(mesh['faces'], faces2delete, 0)
    print('same vertex face removed.')
    print(faces2delete)

  # Remove vertices and fix face indices
  is_to_remove = (mesh['vertices'][:, 0] == -1) + np.isnan(mesh['vertices'][:, 0])
  v_to_remove = np.where(is_to_remove)[0]
  v_to_keep   = np.where(is_to_remove == 0)[0]
  mesh['vertices'] = mesh['vertices'][v_to_keep, :]
  # remove corresponding mano parameters
  mesh['posedirs'] = mesh['posedirs'][v_to_keep, :]
  mesh['shapedirs'] = mesh['shapedirs'][v_to_keep, :]
  mesh['weights'] = mesh['weights'][v_to_keep, :]
  mesh['J_regressor'] = mesh['J_regressor'][v_to_keep, :]
  mesh['relative_vertex'] = [mesh['relative_vertex'][i] for i in v_to_keep]
  mesh['relative_vertex_weight'] = [mesh['relative_vertex_weight'][i] for i in v_to_keep]

  for v_idx in v_to_remove[::-1]:
    f_to_update = np.where(mesh['faces'] > v_idx)
    mesh['faces'][f_to_update] -= 1


def mesh_preprocess(mesh):
  if 0:
    # Unite all "same" vertices - ones that are very close
    for v_idx, v in enumerate(mesh['vertices']):
      d = np.linalg.norm(mesh['vertices'] - v, axis=1)
      idxs0 = np.where(d < SAME_V_TH_FOR_PREPROCESS)[0][1:]
      for v_idx_to_update in idxs0:
        mesh['vertices'][v_idx_to_update] = [np.nan, np.nan, np.nan]
        idxs1 = np.where(mesh['faces'] == v_idx_to_update)
        mesh['faces'][idxs1] = v_idx

    # Remove duplicated faces
    for f in mesh['faces']:
      if f[0] == -1:
        continue
      dup_faces = np.where(np.all(mesh['faces'] == f, axis=1))[0][1:]
      mesh['faces'][dup_faces, :] = -1

  # Check if model is watertight
  mesh_calc.add_edges_to_mesh(mesh)
  print('is water-tight:', mesh['is_watertight'])

  # Prepare mesh
  mesh_calc.calc_v_adjacency_matrix(mesh)
  mesh_calc.calc_vf_adjacency_matrix(mesh)
  mesh_calc.calc_face_plane_parameters(mesh)

  # Make sure the mesh is good now
  if SELF_CHECKING:
    check_mesh(mesh)

def simplify_mesh(mesh_orig, n_vertices_to_merge):
  mesh = copy.deepcopy(mesh_orig)
  tb = time.time()
  mesh_preprocess(mesh)

  mesh['pair_heap'] = []

  # Calc Q matrix for eack vertex
  calc_Q_for_each_vertex(mesh)

  # Select pairs and add them to a heap
  select_vertex_pairs(mesh)

  print('Init time:', time.time() - tb)

  # Take and contract pairs
  tb = time.time()
  print('Simplifing Mesh')
  for _ in tqdm(range(n_vertices_to_merge)):
    contract_best_pair(mesh)
    if mesh['n_faces'] <= MINIMUM_NUMBER_OF_FACES:
      break
  print('Iteration time:', time.time() - tb)

  # Remove old unused faces
  clean_mesh_from_removed_items(mesh)

  return mesh

def simplify(mesh, n_vertices_to_merge):
    mesh_orig = {}
    mesh_orig['vertices'] = mesh.v
    mesh_orig['faces'] = mesh.vi.reshape(-1 ,3)

    mesh_orig['posedirs'] = mesh.posedirs
    mesh_orig['shapedirs'] = mesh.shapedirs
    mesh_orig['weights'] = mesh.weights
    mesh_orig['J_regressor'] = mesh.regressors
    mesh_orig['relative_vertex'] = mesh.relative_vertex
    mesh_orig['relative_vertex_weight'] = mesh.relative_vertex_weight

    mesh_orig['n_faces'] = mesh_orig['faces'].shape[0]
    mesh_orig['n_vertices'] = mesh_orig['vertices'].shape[0]

    mesh_out = simplify_mesh(mesh_orig, n_vertices_to_merge)

    mesh_out['v_template'] = mesh_out['vertices']
    mesh_out['f'] = mesh_out['faces']
    mesh_out.pop('vertices')
    mesh_out.pop('faces')
    mesh_out['J_regressor'] = sparse.csc_matrix(mesh_out['J_regressor'].T)
    mesh_out = TriangleMesh(mesh_out)
    return mesh_out
    

def get_mesh(idx=0):
  global CLOSE_DIST_TH

  if idx == -1:
    mesh = io_off_model.get_simple_mesh('for_mesh_simplification_1')
    mesh['name'] = 'simple_2d_mesh_1'
    n_vertices_to_merge = 1
  elif idx == -2:
    mesh = io_off_model.get_simple_mesh('for_mesh_simplification_2')
    mesh['name'] = 'simple_2d_mesh_2'
    n_vertices_to_merge = 1
    CLOSE_DIST_TH = 0.5
  else:
    mesh_fns = [['meshes/bottle_0320.off',    30],    # 50
                ['meshes/person_0067.off',    600],
                ['meshes/airplane_0359.off',  1000],
                ['meshes/person_0004.off',    1000],
                ['meshes/bunny2.off',         4000],
                ['meshes/cat.off',            6000],
                ['meshes/phands.off',         2000],
                ]
    n_vertices_to_merge = mesh_fns[idx][1]
    mesh = io_off_model.read_off(mesh_fns[idx][0], verbose=True)
    mesh['name'] = os.path.split(mesh_fns[idx][0])[-1]

  return mesh, n_vertices_to_merge

def run_one(mesh_id=0, n_vertices_to_merge=None):
  mesh, n_vertices_to_merge_ = get_mesh(mesh_id)
  tb = time.time()
  if n_vertices_to_merge is None:
    n_vertices_to_merge = n_vertices_to_merge_
  mesh_simplified = simplify_mesh(mesh, n_vertices_to_merge)
  print('Number of faces in: ', mesh['n_faces'])
  print('Number of faces after simplification: ', mesh_simplified['n_faces'])
  print('Time: ', time.time() - tb)
  if not os.path.isdir('output_meshes'):
    os.makedirs('output_meshes')
  fn = 'output_meshes/' + mesh['name'].split('.')[0] + '_simplified_' + str(n_vertices_to_merge) + '.off'
  io_off_model.write_off_mesh(fn, mesh_simplified)
  #fn = 'output_meshes/' + mesh['name'].split('.')[0] + '_simplified.obj'
  #io_off_model.write_off_mesh(fn, mesh_simplified)
  fn = 'output_meshes/' + mesh['name'].split('.')[0] + '.obj'
  io_off_model.write_mesh(fn, mesh)

def run_bunny_many():
  for n_vertices_to_merge in [4000, 6000, 6100, 6200]:
    run_one(mesh_id=4, n_vertices_to_merge=n_vertices_to_merge)

def run_all():
  simple_models = [-2, -1]
  watertight_models = [4, 5, 6]
  non_watertight_models = [0, 1, 2, 3]
  for mesh_id in non_watertight_models:
    run_one(mesh_id)

if __name__ == '__main__':
  #run_all()
  #run_bunny_many()
  run_one(4)