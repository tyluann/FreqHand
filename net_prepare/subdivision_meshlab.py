import pymeshlab
import os
import tqdm

def subdivison(file_in, file_out, subnum):
    if subnum == 0:
        return
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(file_in)
    #mesh = pymeshlab.Mesh(vertex_matrix=verts, face_matrix=faces)
    #ms.add_mesh(mesh)
    ms.meshing_surface_subdivision_loop(iterations=subnum, threshold=pymeshlab.Percentage(0))
    ms.save_current_mesh(file_out)
    # new_verts = ms.current_mesh().vertex_matrix()
    # new_faces = ms.current_mesh().face_matrix()
    #return new_verts, new_faces

if __name__ == "__main__":
    input_dir = '/home/tyluan/workspace/code/DeepHandMesh/output_new/0302_082056_loss_predxgt_0306_054817/vis'
    #input_dir = 'output/subdivision/mano_mesh'
    out_dir = '/home/tyluan/workspace/code/DeepHandMesh/output_new/0302_082056_loss_predxgt_0306_054817/vis'
    os.makedirs(out_dir, exist_ok=True)
    for file in tqdm.tqdm(os.listdir(input_dir)):
        if not file.endswith('eval_ori.obj'):
            continue
        file_in = os.path.join(input_dir, file)
        file_out = os.path.join(out_dir, file[:-4] + '_2.obj')
        subdivison(file_in, file_out, 2)

