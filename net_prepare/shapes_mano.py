# from geometry import *
# from ctypes import *  
import numpy as np

class TriangleMesh(): #(Structure):
    # _fields_ = [('vNum',c_uint),('v',POINTER(Vector)),
    #             ('viNum',c_uint),('vi',POINTER(c_uint))]
                # ('vnNum',c_uint),('vn',POINTER(Vector)),
                # ('vni',POINTER(c_uint))]
                
    def __init__(self, dd): #v = tuple(), vi = tuple(), vn = tuple(), vni = tuple()):
        v = np.array(dd['v_template'])
        f = np.array(dd['f'])
        vi = f.reshape(-1)
        viNum = vi.shape[0]

        self.vNum = v.shape[0]
        self.viNum = viNum
        # self.vnNum = len(vn)
        # vniNum = len(vni)
        
        if self.viNum % 3: #or vniNum % 3 :
            raise ValueError("ERROR in TriangleMesh: not enought elements !!!!!!")
        
        self.v = v #(Vector * self.vNum)(*v)
        self.vi = vi #(c_uint * self.viNum)(*vi)

        self.shapedirs = np.array(dd['shapedirs'])
        self.posedirs = np.array(dd['posedirs'])
        self.weights = np.array(dd['weights'])
        self.regressors = dd['J_regressor'].A.T
        # if self.regressors.shape[1] == 16:
        #     self.regressors = np.insert(self.regressors, 4, np.zeros(self.regressors.shape[0]), axis=1)
        #     self.regressors = np.insert(self.regressors, 8, np.zeros(self.regressors.shape[0]), axis=1)
        #     self.regressors = np.insert(self.regressors, 12, np.zeros(self.regressors.shape[0]), axis=1)
        #     self.regressors = np.insert(self.regressors, 16, np.zeros(self.regressors.shape[0]), axis=1)
        #     self.regressors = np.insert(self.regressors, 20, np.zeros(self.regressors.shape[0]), axis=1)
        #     self.regressors[320, 4] = 1
        #     self.regressors[443, 8] = 1
        #     self.regressors[672, 12] = 1
        #     self.regressors[555, 16] = 1
        #     self.regressors[744, 20] = 1
            # dd['kintree_table'] = np.array([
            #     [4294967295, 0, 1, 2, 3, 0, 5, 6, 7, 0, 9,  10, 11, 0,  13, 14, 15, 0,  17, 18, 19],
            #     [0,          1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]])
            # self.weights = np.insert(self.weights, 4, np.zeros(self.weights.shape[0]), axis=1)
            # self.weights = np.insert(self.weights, 8, np.zeros(self.weights.shape[0]), axis=1)
            # self.weights = np.insert(self.weights, 12, np.zeros(self.weights.shape[0]), axis=1)
            # self.weights = np.insert(self.weights, 16, np.zeros(self.weights.shape[0]), axis=1)
            # self.weights = np.insert(self.weights, 20, np.zeros(self.weights.shape[0]), axis=1)
            # Jtr.insert(4,v[:,:3,320].unsqueeze(2))
            # Jtr.insert(8,v[:,:3,443].unsqueeze(2))
            # Jtr.insert(12,v[:,:3,672].unsqueeze(2))
            # Jtr.insert(16,v[:,:3,555].unsqueeze(2))
            # Jtr.insert(20,v[:,:3,744].unsqueeze(2))   
            #self.regressors = np.concatenate((self.regressors, np.zeros(self.regressors.shape[0], 5)), axis=1)

        if 'relative_vertex' in dd:
            self.relative_vertex = dd['relative_vertex']
            self.relative_vertex_weight = dd['relative_vertex_weight']
        else:
            self.relative_vertex = []
            self.relative_vertex_weight = []

        # if self.vnNum:
        #     self.vn = (Vector * self.vnNum)(*vn)
        #     if not vniNum:
        #         self.vni = (c_uint * self.viNum)(*vi)
        #     else:
        #         self.vni = (c_uint * vniNum)(*vni)
        # else:
        #     if not vniNum:
        #         self.vn = get_normals(self.vNum, self.v, self.viNum, self.vi)
        #         self.vni = (c_uint * self.viNum)(*vi)
        #     else:
        #         pass

# def get_normals(vNum, vertices, viNum, indices):
#     res  = (Vector * vNum)()
    
#     for j in range(0,viNum,3):
#         for i in range(3): 
#             dirx = vertices[indices[j + (i + 1)%3]] - vertices[indices[j + i]]
#             dirz = vertices[indices[j + (i + 3 - 1)%3]] - vertices[indices[j + i]]
#             diry = dirx.cross(dirz)
            
#             res[indices[j + i]] += diry
            
#     for j in range(vNum):
#         res[j].normalize()
        
#     return res    
