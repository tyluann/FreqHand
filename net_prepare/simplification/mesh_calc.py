import collections

import pylab as plt
import numpy as np
import scipy.linalg

import io_off_model

def calc_v_adjacency_matrix(mesh, verbose=False):
  if 'v_adjacency_matrix' in mesh.keys():
    return
  if verbose:
    print('calc_v_adjacency_matrix')
  n_vertices = mesh['vertices'].shape[0]
  mesh['v_adjacency_matrix'] = np.zeros((n_vertices, n_vertices), dtype=np.bool)
  for face in mesh['faces']:
    for i, j in zip([0, 1, 1, 2, 2, 0], [1, 0, 2, 1, 0, 2]):
      mesh['v_adjacency_matrix'][face[i], face[j]] = True

def calc_vf_adjacency_matrix(mesh, verbose=False):
  # Usage:
  #   vf_adjacency_matrix[vertex_index, face_index],
  #   if True, vertex_index is in mesh['faces'][face_index]
  if 'vf_adjacency_matrix' in mesh.keys():
    return
  if verbose:
    print('calc_vf_adjacency_matrix')
  n_vertices = mesh['vertices'].shape[0]
  n_faces = mesh['faces'].shape[0]
  mesh['vf_adjacency_matrix'] = np.zeros((n_vertices, n_faces), dtype=np.bool)
  for f_ind in range(mesh['faces'].shape[0]):
    face = mesh['faces'][f_ind]
    mesh['vf_adjacency_matrix'][face, f_ind] = True

def calc_face_normals(mesh, verbose=False, must_recalc=False):
  if 'face_normals' in mesh.keys() and must_recalc == False:
    return
  if verbose:
    print('calc_face_normals')
  v0 = mesh['vertices'][mesh['faces'][:, 0]]
  v1 = mesh['vertices'][mesh['faces'][:, 1]]
  v2 = mesh['vertices'][mesh['faces'][:, 2]]
  tmp = np.cross(v0 - v1, v1 - v2)
  mesh['face_normals'] = tmp / np.linalg.norm(tmp, axis=1, keepdims=True)

def calc_face_plane_parameters(mesh, verbose=False, must_recalc=False):
  if 'face_plane_parameters' in mesh.keys() and must_recalc == False:
    return
  if verbose:
    print('calc_face_plane_parameters')
  calc_face_normals(mesh, must_recalc=must_recalc)
  normals = mesh['face_normals']
  p0s = mesh['vertices'][mesh['faces'][:, 0]]
  ds = np.sum(-normals * p0s, axis=1)[:, None]
  mesh['face_plane_parameters'] = np.hstack((normals, ds))

  # Check
  if 0:
    # All a^2 + b^2 + c^2 ~= 1 :
    norma_ = np.linalg.norm(mesh['face_plane_parameters'][:, :3], axis=1)
    print('Are all normas ~1? ', np.all(np.abs(norma_ - 1) < 1e-6))
    # All points of faces lay in the planes :
    p1s = mesh['vertices'][mesh['faces'][:, 1]] # p1 is taken
    p1s_with1ns = np.hstack((p1s, np.ones((p1s.shape[0], 1))))
    d = np.sum(mesh['face_plane_parameters'] * p1s_with1ns, axis=1)
    print('Are all 2nd points of all faces lay in the plane defined?', np.all(np.abs(d) < 1e-6))

def calc_triangles_area(mesh, verbose=False):
  if 'faces_area' in mesh.keys():
    return
  if verbose:
    print('calc_triangles_area')
  all_triangles = mesh['vertices'][mesh['faces']]               # get all triangles, matrix of : [n-faces , 3 , 3]
  diff_each_2 = np.diff(all_triangles, axis=1)                  # get two edges for each triangle
  cross = np.cross(diff_each_2[:, 0], diff_each_2[:, 1])        # the magnitude result of the cross product equals to the "makbilit" area
  mesh['faces_area'] = (np.sum(cross ** 2, axis=1) ** .5) / 2   # get the magnitue for each face and devide it by 2

  return mesh['faces_area']

def calc_vertices_area(mesh, verbose=False):
  if verbose:
    print('calc_vertices_area')
  if 'faces_area' not in mesh.keys():
    calc_triangles_area(mesh)
  if 'vf_adjacency_matrix' not in mesh.keys():
    calc_vf_adjacency_matrix(mesh)
  mesh['vertices_area'] = np.zeros((mesh['vertices'].shape[0]))
  for i in range(mesh['vertices_area'].shape[0]):
    areas = mesh['faces_area'][mesh['vf_adjacency_matrix'][i]]
    mesh['vertices_area'][i] = np.sum(areas) / 3
  return mesh['vertices_area']

def add_edges_to_mesh(mesh):
  if 'edges' in mesh.keys():
    return
  edges = {}
  edges2face = {}
  for f_index, f in enumerate(mesh['faces']):
    for e in [(f[0], f[1]), (f[1], f[2]), (f[2], f[0])]:
      e_ = (min(e), max(e))
      if e_ not in edges.keys():
        edges2face[e_] = []
        edges[e_] = 0
      edges[e_] += 1
      edges2face[e_].append(f_index)
  mesh['edges'] = np.array(list(edges.keys()))
  n_faces_for_edge = np.array(list(edges.values()))
  mesh['n_boundary_edges'] = np.sum(n_faces_for_edge == 1)
  mesh['is_watertight'] = np.all(n_faces_for_edge == 2)
  non_maniford_edges_idxs = np.where(n_faces_for_edge != 2)[0]
  non_maniford_edges = mesh['edges'][non_maniford_edges_idxs]
  mesh['non_maniford_vertices'] = np.unique(non_maniford_edges)

  mesh['faces_graph'] = {f:[] for f in range(mesh['n_faces'])}
  for adj_faces in edges2face.values():
    for f in adj_faces:
      mesh['faces_graph'][f] += [f_ for f_ in adj_faces if f_ is not f]

def topological_measures(mesh, verbose=False):
  V = mesh['vertices'].shape[0]
  F = mesh['faces'].shape[0]
  if 'edges' not in mesh.keys():
    add_edges_to_mesh(mesh)
  E = mesh['edges'].shape[0]
  chi = V + F - E
  genus = 1 - chi / 2

  if verbose:
    print('Number of faces / vertices / edges : ', F, V, E)
    print('Chi : ', chi)
    print('Genus : ', genus)
    print('Number of boundary edges : ', mesh['n_boundary_edges'])

  return genus, mesh['n_boundary_edges']

def calc_valences(mesh):
  if 'valences' in mesh.keys():
    return

  mesh['valences'] = np.bincount(mesh['faces'].flatten())

def calc_v2f_one_ring_matrix(mesh):
  if 'v2f_one_ring_matrix' in mesh.keys():
    return

  mesh['v2f_one_ring_matrix'] = np.zeros((mesh['faces'].shape[0], mesh['vertices'].shape[0]))
  for v in range(mesh['vertices'].shape[0]):
    idxs = np.where(mesh['faces'] == v)[0]
    mesh['v2f_one_ring_matrix'][idxs, v] = True

def calc_interpolation_matrices(mesh, verbose=False):
  if 'interp_matrix_v2f' in mesh.keys():
    return
  if verbose:
    print('calc_interpolation_matrices')
  # Make sure area is calculated and also the adjacency matrix
  calc_triangles_area(mesh)
  calc_vertices_area(mesh)
  calc_v2f_one_ring_matrix(mesh)

  # Resape to matrices (for "brodcasting") and multiply to get the results
  Af = mesh['faces_area'].reshape(1, -1)
  Av = mesh['vertices_area'].reshape(-1, 1)
  invAv = (1 / (3 * Av))

  mesh['interp_matrix_f2v'] = mesh['v2f_one_ring_matrix'].T * (Af * invAv / 3)
  mesh['interp_matrix_v2f'] = (1 / Af.T) * mesh['interp_matrix_f2v'].T * Av.T

def calc_face_centers(mesh):
  if 'face_centers' in mesh.keys():
    return
  mesh['face_centers'] = mesh['vertices'][mesh['faces']].mean(axis=1)

def bfsdfs(graph, root, max_size=np.inf, bfs_flag=True):
  res = []
  seen, queue = set([root]), collections.deque([root])
  while queue:
    if bfs_flag:
      vertex = queue.popleft()   # Change to pop for DFS
    else:
      vertex = queue.pop()  # Change to pop for DFS
    res.append(vertex)
    if len(res) >= max_size:
      break
    for node in graph[vertex]:
      if node not in seen:
        seen.add(node)
        queue.append(node)
  return res

def calc_dist_from_plane(mesh, f0_mean, a, f):
  a = a.reshape((1, 3))
  x = np.mean(mesh['vertices'][mesh['faces'][f]], axis=0).reshape((3, 1)) - f0_mean
  dist = np.dot(a, x) / np.linalg.norm(a)

  return abs(dist[0][0])

def cut_mesh(mesh, f0, a, max_length=np.inf):
  if 'faces_graph' not in mesh.keys():
    add_edges_to_mesh(mesh)

  f0_mean = np.mean(mesh['vertices'][mesh['faces'][f0]], axis=0).reshape((3, 1))
  faces = [f0]
  not_allowed = [f0]

  while len(faces) < max_length:
    f_to_add = -1
    if f0 in mesh['faces_graph'][faces[-1]] and len(faces) > 3: # if we've returned to the same face we've started, its time to finish..
      break
    min_d = np.inf
    for f in mesh['faces_graph'][faces[-1]]:
      d = calc_dist_from_plane(mesh, f0_mean, a, f)
      if d < min_d and f not in not_allowed:
        min_d = d
        f_to_add = f
    if f_to_add == -1:
      break
    print(f_to_add)
    not_allowed += mesh['faces_graph'][faces[-1]]
    faces.append(f_to_add)

  return faces

if __name__ == '__main__':
  from visualization import visualize_mesh

  mesh_fn = r"C:\Users\alon\Downloads\ModelNet40\ModelNet40\car\train\car_0016.off"
  mesh_fn = 'hw2_data/phands.off' # hw2_data/
                                # torus_fat_r2 / cat / sphere_s0 / vase / phands / disk
                                # .off
  mesh = io_off_model.read_off(mesh_fn, verbose=True)

  if 0: # HW2, Ex. 2, part I
    calc_vf_adjacency_matrix(mesh)
  if 0: # HW2, Ex. 4, part I
    t_area = calc_triangles_area(mesh)
    print('Surface area: ', t_area.sum())
    v_area = calc_vertices_area(mesh)
    print('Total vertices area: ', v_area.sum())
  if 0: # HW2, Ex.6, part I
    calc_interpolation_matrices(mesh)
    # Generate some function over the mesh faces
    faces_function = np.zeros((mesh['n_faces']))
    face = 0
    val = 100
    for _ in range(100):
      faces_function[face] = val
      found = False
      for cand_face, cand_vs in enumerate(mesh['faces']):
        if faces_function[cand_face] != 0:
          continue
        if np.union1d(cand_vs, mesh['faces'][face]).size == 4: # 2 vertices are the same
          val += 1
          face = cand_face
          break
    vertices_function = np.dot(mesh['interp_matrix_f2v'], faces_function)

    # Check one way and back
    f_ = np.dot(mesh['interp_matrix_v2f'], vertices_function)
    v_ = np.dot(mesh['interp_matrix_f2v'], f_)
    plt.figure()
    plt.plot(f_ - faces_function)
    plt.figure()
    plt.plot(v_ - vertices_function)
    plt.show()

    # Visualization
    visualize_mesh(mesh, faces_function=faces_function, show_tringles=True)
    visualize_mesh(mesh, vertices_function=vertices_function, show_tringles=True)
  if 0: # HW2, Ex.7, part I
    topological_measures(mesh, verbose=True)
  if 0: # HW2, Ex3, part II
    calc_interpolation_matrices(mesh)
    print('Interpolation matrices size:')
    print('  f2v: ', mesh['interp_matrix_f2v'].shape)
    print('  v2f: ', mesh['interp_matrix_v2f'].shape)
    x1 = scipy.linalg.null_space(mesh['interp_matrix_f2v'])
    print('Null space for f2v: ', x1.shape)
    x2 = scipy.linalg.null_space(mesh['interp_matrix_v2f'])
    print('Null space for v2f: ', x2.shape)

  if 1: # HW2, self question, #1
    add_edges_to_mesh(mesh)
    faces_order = bfsdfs(mesh['faces_graph'], 0, bfs_flag=True)
    faces_function = np.zeros((mesh['n_faces']))
    for i, f in enumerate(faces_order):
      faces_function[f] = i + mesh['n_faces']
    visualize_mesh(mesh, faces_function=faces_function, show_tringles=False)

  if 0: # HW2, self question, #2
    f0 = 0
    faces = cut_mesh(mesh, f0, np.array((0, 1, 1)))
    print('Number of faces used to cut the mesh:', len(faces))
    print(faces)
    faces_function = np.zeros((mesh['n_faces']))
    if 1:
      faces_function[faces] = 1
      faces_function[f0] = 2
    else:
      for i, f in enumerate(faces):
        faces_function[f] = i

    if 1:
      visualize_mesh(mesh, faces_function=faces_function, show_tringles=False)
    else:
      calc_interpolation_matrices(mesh)
      vertices_function = np.dot(mesh['interp_matrix_f2v'], faces_function)
      visualize_mesh(mesh, vertices_function=vertices_function, show_tringles=False)

