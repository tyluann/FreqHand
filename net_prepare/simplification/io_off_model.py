import numpy as np
import trimesh

def write_mesh(fn, mesh):
  tr_mesh = trimesh.Trimesh(vertices=mesh['vertices'], faces=mesh['faces'])
  tr_mesh.export(fn)

def write_off_mesh(fn, mesh):
  write_off(fn, mesh['vertices'], mesh['faces'])

def write_off(fn, points, polygons, colors=None):
  with open(fn, 'wt') as f:
    f.write('OFF\n')
    f.write(str(len(points)) + ' ' + str(len(polygons)) + ' 0\n')
    for p in points:
      if np.isnan(p[0]):
        p_ = [0, 0, 0]
      else:
        p_ = p
      f.write(str(p_[0]) + ' ' + str(p_[1]) + ' ' + str(p_[2]) + ' \n')
    polygons_ = []
    if colors is None:
      for p in polygons:
        p_ = p
        if type(p) is type(np.array((0))):
          p_ = p.tolist()
        polygons_.append([len(p)] + p_)
    else:
      for p, c in zip(polygons, colors):
        polygons_.append([len(p)] + p + c)
    for p in polygons_:
      for v in p:
        f.write(str(v) + ' ')
      f.write('\n')

def read_off(fn, max_point=np.inf, verbose=False):
  if verbose:
    print('Reading', fn)

  def _read_line_ignore_comments(fp):
    while 1:
      l = fp.readline()
      if l.startswith('#') or l.strip() == '':
        continue
      return l

  if not fn.endswith('off'):
    return None, None
  points = []
  polygons = []
  with open(fn) as file:
    l = _read_line_ignore_comments(file)
    assert (l.strip() == 'OFF')
    n_points, n_polygons, n_edges = [int(s) for s in file.readline().split()]
    if n_points > max_point:
      return None, None

    for i in range(n_points):
      point = [float(s) for s in _read_line_ignore_comments(file).split()]
      points.append(point)

    for i in range(n_polygons):
      polygon = [int(s) for s in _read_line_ignore_comments(file).split()][1:]
      polygons.append(polygon)

  points = np.array(points).astype('float32')
  polygons = np.array(polygons).astype('int')

  mesh = {'vertices': points, 'faces': polygons, 'n_vertices': points.shape[0], 'n_faces': polygons.shape[0]}

  if verbose:
    print('Number of vertices: ', mesh['n_vertices'])
    print('Number of faces: ', mesh['n_faces'])

  return mesh

def get_simple_mesh(type):
  if type == 'one_triangle':
    points = np.array(([0, 0, 0], [1, 0, 0], [0, 1, .1]))
    polygons = np.array(([0, 1, 2],))
  elif type == 'for_mesh_simplification_1':
    points = np.array(([4, 4, 0],
                      [5, 4, 0],
                      [7, 6, 0],
                      [9, 5, 0],
                      [8, 2, 0],
                      [5, 1, 0],
                      [3, 1, 0],
                      [2, 2, 0],
                      [1, 4, 0],
                      [2, 6, 0],
                      [4, 6, 0],
                      ), dtype=np.float)
    polygons = np.array(([0, 1, 10],
                        [0, 5, 1],
                        [0, 6, 5],
                        [0, 7, 6],
                        [0, 8, 7],
                        [0, 9, 8],
                        [0, 10, 9],
                        [1, 2, 10],
                        [1, 3, 2],
                        [1, 4, 3],
                        [1, 5, 4],
                        ))
  elif type == 'for_mesh_simplification_2':
    points = np.array(([5, 2, 0],
                       [2, 3, 0],
                       [1, 2, 0],
                       [1, 1, 0],
                       [5.2, 2, 0],
                       [8, 3, 0],
                       [9, 2, 0],
                       [8, 1, 0],
                       ), dtype=np.float)
    polygons = np.array(([0, 1, 2],
                         [0, 2, 3],
                         [4, 6, 5],
                         [4, 7, 6],
                         ))
  else:
    raise Exception('Unsupported mesh type')

  mesh = {'vertices': points, 'faces': polygons, 'n_vertices': points.shape[0], 'n_faces': polygons.shape[0]}

  return mesh

if __name__ == '__main__':
  mesh = read_off('bunny.off')