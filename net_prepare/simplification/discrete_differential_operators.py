import os

import numpy as np
from numpy import linalg as LA
import pylab as plt

import mesh_calc
import io_off_model
import visualization

def calc_operators(mesh, verbose=False):
  if 'opers' in mesh.keys():
    return
  if verbose:
    print('calc_operators')
  mesh_calc.calc_vertices_area(mesh, verbose=False)
  mesh_calc.calc_triangles_area(mesh, verbose=False)
  mesh_calc.calc_face_normals(mesh, verbose=False)
  face_norms = mesh['face_normals']
  At = mesh['faces_area']

  edge0 = mesh['vertices'][mesh['faces'][:, 2], :] - mesh['vertices'][mesh['faces'][:, 1], :]
  edge1 = mesh['vertices'][mesh['faces'][:, 0], :] - mesh['vertices'][mesh['faces'][:, 2], :]
  edge2 = mesh['vertices'][mesh['faces'][:, 1], :] - mesh['vertices'][mesh['faces'][:, 0], :]

  edge0_90 = np.cross(face_norms, edge0)
  edge1_90 = np.cross(face_norms, edge1)
  edge2_90 = np.cross(face_norms, edge2)

  edge_90_x = np.hstack((edge0_90[:, 0:1], edge1_90[:, 0:1], edge2_90[:, 0:1]))
  edge_90_y = np.hstack((edge0_90[:, 1:2], edge1_90[:, 1:2], edge2_90[:, 1:2]))
  edge_90_z = np.hstack((edge0_90[:, 2:3], edge1_90[:, 2:3], edge2_90[:, 2:3]))

  Ex = np.zeros((mesh['n_vertices'], mesh['n_faces']))
  Ey = np.zeros((mesh['n_vertices'], mesh['n_faces']))
  Ez = np.zeros((mesh['n_vertices'], mesh['n_faces']))
  for ee, edge in zip([Ex, Ey, Ez], [edge_90_x, edge_90_y, edge_90_z]):
    for i in range(3):
      e_idxs = mesh['faces'][:, i]
      f_idxs = np.arange(mesh['n_faces'], dtype=np.int)
      ee[e_idxs, f_idxs] = edge[:, i]
  Ex = Ex.T
  Ey = Ey.T
  Ez = Ez.T
  E = np.vstack((Ex, Ey, Ez))

  Gf = np.diag(np.hstack((mesh['faces_area'], mesh['faces_area'], mesh['faces_area'])))
  GfInv = np.diag(1 / np.hstack((mesh['faces_area'], mesh['faces_area'], mesh['faces_area'])))
  Gv = np.diag(mesh['vertices_area'])
  GvInv = np.diag( 1 / mesh['vertices_area'])
  W_ = 0.25 * np.dot(E.T, GfInv)
  W  = np.dot(W_, E)

  mesh['opers'] = {'E': E,            # |F| x 3 , |V|
                   'Gf': Gf,          # |F| x 3 , |F| x 3
                   'Gv': Gv,          # |V|     , |V|
                   'GfInv': GfInv,    # |F| x 3 , |F| x 3
                   'GvInv': GvInv,    # |V|     , |V|
                   'W': W,            # |V|     , |V|
                   }

def grad(mesh, f_vertices):
  calc_operators(mesh)
  grad_op = 0.5 * np.dot(mesh['opers']['GfInv'], mesh['opers']['E'])
  grad = np.dot(grad_op, f_vertices[:, None])
  grad = np.hstack((grad[:mesh['n_faces']],
                    grad[ mesh['n_faces']:mesh['n_faces']*2],
                    grad[ mesh['n_faces']*2:]))
  return grad


def divergence(mesh, vector_field):
  # vector_field on faces to scalars on vertices
  grad_op = 0.5 * np.dot(mesh['opers']['GfInv'], mesh['opers']['E'])
  div_op_1  = -np.dot(mesh['opers']['GvInv'], grad_op.T)
  div_op    =  np.dot(div_op_1, mesh['opers']['Gf'])
  vector_field_ = np.vstack((vector_field[:, 0:1], vector_field[:, 1:2], vector_field[:, 2:3]))
  diverg = np.dot(div_op, vector_field_)
  return diverg


def laplacian(mesh, f_vertices):
  # Scalars on vertices to scalars on vertices
  lap_oper = np.dot(mesh['opers']['GvInv'], mesh['opers']['W'])
  lap = np.dot(lap_oper, f_vertices)
  return lap


def remove_some_faces(mesh, n_faces_to_keep=20, start_face=0):
  faces_order = mesh_calc.bfsdfs(mesh['faces_graph'], start_face, bfs_flag=True)
  idxs = faces_order[:n_faces_to_keep]
  mesh['faces'] = mesh['faces'][idxs]
  mesh['face_centers'] = mesh['face_centers'][idxs]
  mesh['face_normals'] = mesh['face_normals'][idxs]
  return idxs


def objectives():
  mesh = get_mesh(1)
  mesh_calc.calc_face_centers(mesh)
  mesh_calc.calc_interpolation_matrices(mesh)
  mesh_calc.add_edges_to_mesh(mesh)

  # Gradient on X^2
  faces_function = mesh['face_centers'][:, 0] ** 2
  vertices_function = np.dot(mesh['interp_matrix_f2v'], faces_function)
  vector_field = grad(mesh, vertices_function)

  # Divergence on "ones"
  vf_for_dvrg = np.ones((mesh['n_faces'], 3))
  dvrgn = divergence(mesh, vf_for_dvrg)
  dvrgn_on_faces = np.dot(mesh['interp_matrix_v2f'], dvrgn)

  # Laplacian on X^2
  lap = laplacian(mesh, vertices_function)
  lap_on_faces = np.dot(mesh['interp_matrix_v2f'], lap)

  if 0:
    kept_faces = remove_some_faces(mesh, n_faces_to_keep=80, start_face=83)
    faces_function = faces_function[kept_faces]
    vector_field = vector_field[kept_faces]
    vf_for_dvrg = vf_for_dvrg[kept_faces]
    dvrgn_on_faces = dvrgn_on_faces[kept_faces]


  visualization.visualize_mesh(mesh, show_tringles=True, faces_function=faces_function, alpha=0.75,
                               faces_vector_field=vector_field, title='Gradient on X^2', pause=False)

  visualization.visualize_mesh(mesh, show_tringles=True, faces_function=dvrgn_on_faces[:, 0], alpha=0.75,
                               faces_vector_field=vf_for_dvrg, title='Divergence from vector field [1,1,1] as v.f.', pause=False)

  visualization.visualize_mesh(mesh, show_tringles=True, faces_function=lap_on_faces, alpha=0.75,
                               title='Laplacian on X^2', pause=False)

  plt.show()

def analysis_1():
  mesh = get_mesh(6)
  mesh_calc.calc_face_centers(mesh)
  mesh_calc.calc_interpolation_matrices(mesh)
  mesh_calc.add_edges_to_mesh(mesh)
  calc_operators(mesh)
  w, v = LA.eig(mesh['opers']['W'])
  epsilon = 1e-5
  w_is_pos_semi_def = np.all(np.real(w) >= -epsilon)
  print('NULL: Sum over rows(all results should be close to zero): ', mesh['opers']['W'].sum(axis=1))
  print('      All are close to 0: ', np.all(np.abs(mesh['opers']['W'].sum(axis=1)) < 0.0001))
  print('SYM:  If W is sym, this will be close to zero: ', np.sum((mesh['opers']['W'] - mesh['opers']['W'].T) ** 2))
  print('LOC: Zero / non-zero elements: ', (mesh['opers']['W'] == 0).sum(), (mesh['opers']['W'] != 0).sum())
  print('POS: Is all values are zeros? : ', np.all(mesh['opers']['W'] > 0))
  print('PSD: Is W positive sem-definite? ', w_is_pos_semi_def)

def analysis_2():
  mesh = get_mesh(6)
  mesh_calc.calc_interpolation_matrices(mesh, verbose=True)
  calc_operators(mesh, verbose=True)
  w, v = LA.eig(mesh['opers']['W'])
  sorted_idxs = np.argsort(w)
  if 0:
    N = 9
    fig = plt.figure()
    for i in range(N):
      ax = fig.add_subplot(3, 3, i + 1, projection='3d')
      print(w[sorted_idxs[i]])
      vertices_function = v[:, sorted_idxs[i]]
      faces_function = np.dot(mesh['interp_matrix_v2f'], vertices_function)
      visualization.visualize_mesh(mesh, show_tringles=True, faces_function=faces_function, alpha=0.85,
                                   title=str(i + 1) + 'eig vec', pause=False, ax=ax, vmax=0.1, vmin=-0.1) #

  vertices_function = np.zeros((mesh['n_vertices'], 1))
  vertices_function[26] = 10
  faces_function = np.dot(mesh['interp_matrix_v2f'], vertices_function)[:,0]
  plot = True
  if plot:
    fig = plt.figure()
    ax = fig.add_subplot(2, 2, 1, projection='3d')
    #visualization.visualize_mesh(mesh, show_tringles=True, vertices_function=vertices_function[:, 0],
    #                             title='hat function', pause=False, ax=ax, vmax=1.1, vmin=-0.1)
    visualization.visualize_mesh(mesh, show_tringles=True, faces_function=faces_function, alpha=0.85,
                                 title='hat function', pause=False, ax=ax, vmax=1.1, vmin=-0.1)
  next_subplot = 2
  all_err = []
  if plot:
    all_k = [1, 20, 100]
  else:
    all_k = np.arange(1, mesh['n_vertices'], 5)
  for k in all_k:
    Bi = v[:, sorted_idxs[:k]]
    vertices_function_ = np.dot(Bi, np.dot(Bi.T, vertices_function))
    err = LA.norm(vertices_function - vertices_function_)
    faces_function_ = np.dot(mesh['interp_matrix_v2f'], vertices_function_)[:, 0]
    if plot:
      ax = fig.add_subplot(2, 2, next_subplot, projection='3d')
      visualization.visualize_mesh(mesh, show_tringles=True, faces_function=faces_function_, alpha=0.85,
                                   title='hat function, estimated by k=' + str(k), pause=False, ax=ax, vmax=1.1, vmin=-0.1)
    all_err.append(err)
    next_subplot += 1
  plt.figure()
  plt.plot(all_k, all_err, '*-')
  plt.ylabel('Reconstruction Error')
  plt.xlabel('Number of eigen vector used')
  plt.title('Representation using a reduced basis, for ' + mesh['name'])

  plt.show()

def get_mesh(idx=0):
  if 0:
    mesh = io_off_model.get_simple_mesh('one_triangle')
  else:
    mesh_fns = [r"C:\Users\alon\Downloads\ModelNet40\ModelNet40\car\train\car_0016.off",
                r"C:\Users\alon\Downloads\ModelNet40\ModelNet40\bottle\train\bottle_0320.off",
                r"C:\Users\alon\Downloads\ModelNet40\ModelNet40\airplane\train\airplane_0169.off",
                r"C:\Users\alon\Downloads\ModelNet40\ModelNet40\cone\train\cone_0088.off",
                r"C:\Users\alon\Downloads\ModelNet40\ModelNet40\person\train\person_0034.off",
                r"C:\Users\alon\Downloads\ModelNet40\ModelNet40\cup\train\cup_0019.off",
                'hw2_data/sphere_s0.off',
                'hw2_data/phands.off'
                ]
    mesh = io_off_model.read_off(mesh_fns[idx], verbose=True)
    mesh['name'] = os.path.split(mesh_fns[idx])[-1]
  return mesh

if __name__ == '__main__':
  analysis_1()

if __name__ == '-__main__':
  mesh = get_mesh()
  mesh_calc.add_edges_to_mesh(mesh)
  faces_order = mesh_calc.bfsdfs(mesh['faces_graph'], 0, bfs_flag=True)

  if 1:
    faces_function = np.zeros((mesh['n_faces'],))
    faces_function[faces_order[:1]] = 1
    if 0:
      mesh_calc.calc_interpolation_matrices(mesh)
      vertices_function = np.dot(mesh['interp_matrix_f2v'], faces_function)
    elif 1:
      vertices_function = np.zeros((mesh['n_vertices'],))
      vertices_function[mesh['faces'][faces_order[:1]].flatten()] = 1
    else:
      vertices_function = np.zeros((mesh['n_vertices'],))
      vertices_function[2] = 1
      vertices_function[1] = 1
    vector_field = grad(mesh, vertices_function)
  else:
    vector_field = grad(mesh, mesh['vertices_area'])
    if 0:
      faces_function = np.zeros((mesh['n_faces'],))
      faces_function[0:N] = 1
    else:
      faces_function = mesh['faces_area']

  divr = laplacian(mesh, vertices_function)

  if 0:
    idxs = faces_order[:20]
    mesh['faces'] = mesh['faces'][idxs]
    faces_function = faces_function[idxs]
    #faces_function[1] = -10
    vector_field = vector_field[idxs]

  #visualization.visualize_mesh(mesh, show_tringles=True, faces_vector_field=vector_field,
  #                             faces_function=faces_function, alpha=0.25, normalize_quiver=True)
  #visualization.visualize_mesh(mesh, show_tringles=True, vertices_function=vertices_function,
  #                             to_show='text', alpha=0.5, faces_vector_field=vector_field)
  visualization.visualize_mesh(mesh, show_tringles=True,
                               vertices_function=divr, alpha=0.5)


