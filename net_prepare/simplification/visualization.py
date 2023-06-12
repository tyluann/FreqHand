from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import pylab as plt
import numpy as np
#from mayavi import mlab
import matplotlib.tri as mtri
from scipy.spatial import ConvexHull
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import open3d

import io_off_model
import mesh_calc


def visualize_mesh(mesh, faces_function=None, vertices_function=None, to_show='color', show_tringles=False,
                   pause=True, faces_vector_field=None, normalize_quiver=False, alpha=1.0, title=' ', ax=None,
                   vmin=None, vmax=None):
  def getColors(m, a):
    b = m.to_rgba(a)
    return [(i[0], i[1], i[2]) for i in b]

  show_colorbar = False
  if faces_function is not None:
    assert faces_function.shape[0] == mesh['faces'].shape[0]
  if vertices_function is not None and to_show != 'text':
    assert vertices_function.shape[0] == mesh['vertices'].shape[0]
    m = cm.ScalarMappable(cmap=cm.jet)
    vertices_colors = getColors(m, vertices_function)
    visualize_mesh_v(mesh, vertices_colors)
    return

  cutoffx = [np.min(mesh['vertices'][:, 0]), np.max(mesh['vertices'][:, 0])]
  cutoffy = [np.min(mesh['vertices'][:, 1]), np.max(mesh['vertices'][:, 1])]
  cutoffz = [np.min(mesh['vertices'][:, 2]), np.max(mesh['vertices'][:, 2])]

  if ax is None:
    fig = plt.figure()
    ax = Axes3D(fig)

  vertices = []
  for f in mesh['faces']:
    vertices.append(mesh['vertices'][f, :])
  if show_tringles:
    edgecolor = 'k'
  else:
    edgecolor = None
  triangles = Poly3DCollection(vertices, edgecolor=edgecolor)
  triangles.set_alpha(alpha)
  triangles.set_facecolor('b')

  if faces_function is not None and to_show == 'color':
    show_colorbar = True
    m = cm.ScalarMappable(cmap=cm.jet)
    m.set_array([min(faces_function), max(faces_function)])
    if vmin is None:
      vmin = min(faces_function)
    if vmax is None:
      vmax = max(faces_function)
    m.set_clim(vmin=vmin, vmax=vmax)
    triangles.set_facecolor(getColors(m, faces_function))

  ax.add_collection3d(triangles)

  #if show_colorbar:
  #  fig.colorbar(m, fraction=0.046, pad=0.04)

  # Show text
  if faces_function is not None and to_show == 'text':
    for i in range(mesh['faces'].shape[0]):
      vs = mesh['vertices'][mesh['faces'][i], :]
      label = str(faces_function[i])
      ax.text(np.mean(vs[:, 0]), np.mean(vs[:, 1]), np.mean(vs[:, 2]), label)

  if vertices_function is not None and to_show == 'text':
    for i in range(mesh['vertices'].shape[0]):
      vs = mesh['vertices'][i, :]
      label = str(vertices_function[i])
      ax.text(vs[0], vs[1], vs[2], label)

  if faces_vector_field is not None:
    if not 'face_centers' in mesh.keys():
      mesh_calc.calc_face_centers(mesh)

    x = mesh['face_centers'][:, 0]# + mesh['face_normals'][:, 0] * 0.03
    y = mesh['face_centers'][:, 1]# + mesh['face_normals'][:, 1] * 0.03
    z = mesh['face_centers'][:, 2]# + mesh['face_normals'][:, 2] * 0.03
    ax.quiver(x, y, z,
              faces_vector_field[:, 0],
              faces_vector_field[:, 1],
              faces_vector_field[:, 2],
              length=0.1, normalize=normalize_quiver, color='black')

  ax.set_xlim(cutoffx[0], cutoffx[1])
  ax.set_ylim(cutoffy[0], cutoffy[1])
  ax.set_zlim(cutoffz[0], cutoffz[1])

  plt.title(title)
  if pause:
    plt.show()

def visualize_mesh_v(mesh, vertices_colors):
  open3d_mesh = open3d.geometry.TriangleMesh()
  open3d_mesh.vertices = open3d.Vector3dVector(mesh['vertices'])
  #open3d_mesh.triangles = open3d.Vector3iVector(mesh['faces'][:, ::-1])
  open3d_mesh.triangles = open3d.Vector3iVector(mesh['faces'])
  if vertices_colors is not None:
    open3d_mesh.vertex_colors = open3d.Vector3dVector(vertices_colors)
  else:
    open3d_mesh.vertex_colors = open3d.Vector3dVector(np.random.uniform(0, 1, size=(mesh['n_vertices'], 3)))
  open3d.draw_geometries([open3d_mesh])


def visualize_mesh_(mesh):
  representation = 'surface'
  mlab.triangular_mesh([vert[0] for vert in mesh['vertices']],
                       [vert[1] for vert in mesh['vertices']],
                       [vert[2] for vert in mesh['vertices']],
                       mesh['faces'], representation=representation)
  mlab.show()

if __name__ == '__main__':
    fn = r"C:\Users\alon\Downloads\ModelNet40\ModelNet40\car\train\car_0016.off"
    #fn = '/home/alonlahav/datasets/ModelNet40/airplane/train/airplane_0438.off'
    #fn = 'bunny.off'
    #fn = '/home/alonlahav/datasets/ModelNet10/ModelNet10/chair/train/chair_0002.off'
    #fn = 'hw2_data/frog_s3.off'
    #fn = 'hw2_data/disk.off'
    mesh = io_off_model.read_off(fn)
    if 0:
      mesh_calc.calc_vertices_area(mesh)
      visualize_mesh(mesh, vertices_function=mesh['vertices_area'], to_show='color')
    elif 1: # Check vector_field
      faces_function = np.zeros((mesh['n_faces'], ))
      faces_function[0:10] = 1
      mesh_calc.calc_face_normals(mesh)
      vector_field = np.zeros((mesh['n_faces'], 3))
      if 1:
        vector_field[0:10] = mesh['face_normals'][:10, :]
      else:
        vector_field[0:10] = [-1, -1, 1]
      visualize_mesh(mesh, show_tringles=True, faces_vector_field=vector_field, faces_function=faces_function, alpha=0.5)
    else:
      mesh_calc.calc_triangles_area(mesh)
      visualize_mesh(mesh, faces_function=mesh['faces_area'], to_show='color')
