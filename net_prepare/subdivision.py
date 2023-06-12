import os
from subdivision.subdiv_mano import subdivide_one
from subdivision.parsingObj_mano import _read_mano, _write_mano_obj, _write_mano
from simplification.surface_simplification import simplify


if __name__ == '__main__':
    

    subdNum = 2
    #mesh = _read_obj_file('subdivision/SampleFiles/bigguy.obj')
    #mano_model = 'MANO_LEFT.pkl'
    #mano_model = 'MANO_RIGHT.pkl'
    mano_models = ['MANO_LEFT', 'MANO_RIGHT']

    for mano_model in mano_models:
        mesh, dd = _read_mano(os.path.join('assets', 'mano', 'models', mano_model + '.pkl'))
        if 0:
            # output colored joint to identify joint sequence
            import chumpy as ch
            from chumpy.ch import MatVecMult
            import numpy as np
            J_tmpx = MatVecMult(dd['J_regressor'], dd['v_template'][:, 0])
            J_tmpy = MatVecMult(dd['J_regressor'], dd['v_template'][:, 1])
            J_tmpz = MatVecMult(dd['J_regressor'], dd['v_template'][:, 2])
            J = np.array(ch.vstack((J_tmpx, J_tmpy, J_tmpz)).T)
            color = [
                (0, 0, 0), (50, 0, 0), (100, 0, 0), (150, 0, 0), #(200, 0, 0), (250, 0, 0), 
                (0, 60, 0), (0, 120, 0), (0, 180, 0), #(0, 240, 0),
                (0, 0, 60), (0, 0, 120), (0, 0, 180), #(0, 0, 240),
                (60, 60, 0), (120, 120, 0), (180, 180, 0), #(240, 240, 0),
                (0, 60, 60), (0, 120, 120), (0, 180, 180), #(0, 240, 240)
            ]
            with open('test_joint.obj', 'w') as f:
                for i in range(J.shape[0]):
                    print('v', J[i][0], J[i][1], J[i][2], color[i][0], color[i][1], color[i][2], file=f)
            

        save_path = os.path.join('assets', 'mano', 'subdivision_model_0')
        os.makedirs(save_path, exist_ok=True)
        _write_mano_obj(os.path.join(save_path, mano_model + '.obj'), mesh)
        _write_mano(os.path.join(save_path, mano_model + '.pkl'), mesh, dd)
        for i in range(subdNum):
            print('MANO model: %s, iteration: %d' % (mano_model, i + 1))
            mesh = subdivide_one(mesh)
            #mesh = simplify(mesh, int(mesh.v.shape[0] / 2))
            save_path = os.path.join('assets', 'mano', 'subdivision_model_%d' % (i + 1))
            os.makedirs(save_path, exist_ok=True)
            _write_mano_obj(os.path.join(save_path, mano_model + '.obj'), mesh)
            _write_mano(os.path.join(save_path, mano_model + '.pkl'), mesh, dd)

        #print('')


    # _write_network_obj('mano/test_hand.obj', 'mano/test_connect.obj', mesh_refine)