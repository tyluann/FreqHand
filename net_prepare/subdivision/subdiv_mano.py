from os import kill
# from geometry import *
# from ctypes import *
import array
import numpy as np
from scipy import sparse
import sys
import os
sys.path.append(os.path.realpath('.'))
from shapes_mano import TriangleMesh
from subdivision.parsingObj_mano import _read_mano, _write_mano, _write_mano_obj, _write_network_obj

def NEXT(i):
    return (i+1)%3
def PREV(i):
    return (i+2)%3


class SDVert(): #(Structure):
    # _fields_ = [('pos', Vector),('child',c_int),('startFace',c_int),
    #             ('boundary',c_ubyte),('regular',c_ubyte),
    #             ('valence',c_uint),('ring',POINTER(c_uint))]
    
    def __init__(self, pos, pd, sd, w, rv = [], rvw = [], b = False, r = False, v = 0):
        self.pos = pos
        self.startFace = -1
        self.child = -1
        self.boundary = b
        self.regular = r
        self.valence = v
        self.posedir = pd
        self.shapedir = sd
        self.weight = w
        self.relative_vertex = rv
        self.relative_vertex_weight = rvw
        # self.regressor = reg

class SDFace(): #(Structure):
    # _fields_ = [('v', c_int * 3),('nFaces', c_int * 3),('children',c_int * 4)]
    
    def __init__(self,v0 = -1,v1 = -1,v2 = -1):
        self.v = np.zeros(3, dtype=np.int)
        self.nFaces = np.zeros(3, dtype=np.int)
        self.children = np.zeros(4, dtype=np.int)
        self.v[0] =v0 
        self.v[1] =v1
        self.v[2] =v2
        for i in range(3):
            self.nFaces[i] = -1
            self.children[i] = -1
        self.children[3] = -1
    
    def vnum(self, vert_i):
        for i in range(3):
            if vert_i == self.v[i]:
                return i
        raise ValueError("ERROR in 'vnum' method :the supplied vertex does not belong to the face")
        return -1

    def nextFace(self,vert_i):
        for i in range(3):
            if vert_i == self.v[i]:
                return self.nFaces[i]
        raise ValueError("ERROR in 'nextFace' method :the supplied vertex does not belong to the face")
        return -1
    
    def prevFace(self,vert_i):
        for i in range(3):
            if vert_i == self.v[i]:
                return self.nFaces[PREV(i)]
        raise ValueError("ERROR in 'prevFace' method :the supplied vertex does not belong to the face")
        return -1
    
    def nextVertex(self,vert_i):
        for i in range(3):
            if vert_i == self.v[i]:
                return self.v[NEXT(i)]
        raise ValueError("ERROR in 'nextVertex' method :the supplied vertex does not belong to the face")
        return -1
    
    def prevVertex(self,vert_i):
        for i in range(3):
            if vert_i == self.v[i]:
                return self.v[PREV(i)]
        raise ValueError("ERROR in 'prevVertex' method :the supplied vertex does not belong to the face")
        return -1
    
    def otherVertex(self, vert1, vert2):
        for i in range(3):
            if not ((vert1 == self.v[i]) or (vert2 == self.v[i])):
                return self.v[i]
        raise ValueError("ERROR in 'otherVertex' method !!!!!!!!")
        return -1

class SDEdge(): #(Structure):
    # _fields_ = [('v', c_int * 2)]
    
    def __init__(self, v0, v1):
        self.v = np.zeros(2, dtype=np.int)
        self.v[0] = min(v0,v1)
        self.v[1] = max(v0,v1)
    
    def __eq__(self, other):
        return ( self.v[0] == other.v[0] and self.v[1] == other.v[1] )
        
    def __lt__(self, other):
        if self.v[0] == other.v[0]:
            return self.v[1] < other.v[1]
        return self.v[0] < other.v[0]
        
    def __hash__(self):
        return hash((self.v[0], self.v[1]))
  

class SDTriangleMesh: #(Structure):
    
    # _fields_ = [('vNum',c_uint),('sd_verts',POINTER(SDVert)),
    #             ('fNum',c_uint),('sd_faces',POINTER(SDFace)),
    #             ('eNum',c_uint),
    #             ('sdNum',c_ushort)]
                
    def __init__(self, mesh, sd_num = 0): #vNum , v, viNum, vi, sd_num = 0):
        self.vNum = mesh.vNum
        sd_verts = [0] * self.vNum #(SDVert * self.vNum)()
        
        self.fNum = int(mesh.viNum / 3)
        #sd_faces = (SDFace * self.fNum)()
        sd_faces = [0] * self.fNum
        
        for i in range(self.vNum):
            # sd_verts[i].__init__(mesh.v[i])
            sd_verts[i] = SDVert(mesh.v[i], pd=mesh.posedirs[i], sd=mesh.shapedirs[i], w=mesh.weights[i])
        
        eNum = 0
        edge_map = dict()
        for i in range(0, mesh.viNum, 3):
            index = int(i / 3)
            sd_faces[index] = SDFace(mesh.vi[i], mesh.vi[i + 1], mesh.vi[i + 2])
            sd_face = sd_faces[index]
            
            for j in range(0,3):
                sd_verts[sd_face.v[j]].startFace = index
                
                sd_edge = SDEdge(sd_face.v[j], sd_face.v[ NEXT(j) ])
                other = edge_map.pop(sd_edge,None)
                
                if other != None:
                    sd_face.nFaces[j] = other[1]
                    sd_faces[other[1]].nFaces[other[0]] = index
                else:
                    eNum += 1
                    edge_map.setdefault(sd_edge, (j, index))
            
        edge_map.clear()
        
        for i in range(mesh.vNum):
            vert = sd_verts[i]
              
            ring = [] # array.array('I')
            ring.append( sd_faces[vert.startFace].nextVertex(i) )
            face_i = sd_faces[vert.startFace].nextFace(i)
            valence = 1
            
            while face_i != -1 and  face_i != vert.startFace:
                ring.append( sd_faces[face_i].nextVertex(i) )
                face_i = sd_faces[face_i].nextFace(i)
                valence += 1
            
            vert.boundary = (face_i == -1) 
            if not vert.boundary:
                vert.regular = (valence == 6)
            else:
                face_i = vert.startFace
                ring.reverse()
                while face_i != -1:
                    ring.append( sd_faces[face_i].prevVertex(i) )
                    face_i = sd_faces[face_i].prevFace(i)
                    valence += 1
                vert.regular = (valence == 4)
                
            vert.valence = valence
            vert.ring = ring #(int * vert.valence)(*ring)
            
        self.sdNum = sd_num
        self.sd_verts = sd_verts
        self.sd_faces = sd_faces
        self.eNum = eNum
        self.regs = mesh.regressors
    
    def refine(self):
        vNum = self.vNum
        fNum = self.fNum
        eNum = self.eNum
        sd_verts = self.sd_verts
        sd_faces = self.sd_faces
        
        mesh_list = []

        mesh_list.append(convert(vNum , sd_verts, fNum, sd_faces, self.regs))
        
        for k in range(self.sdNum):
            newVNum = vNum + eNum
            newFNum = 4*fNum
            newENum = 2*eNum + 3*fNum
            # new_verts = (SDVert * newVNum)()
            # new_faces = (SDFace * newFNum)()
            new_verts = [0] * newVNum
            new_faces = [0] * newFNum
            
            #~ compute new even vertices
            for i in range(vNum):
                vert = sd_verts[i]
                if vert.boundary:
                    pos, posedir, shapedir, weight, relative_vertex, relative_vertex_weight = weightBoundaryVert(vert, float(1)/8, sd_verts)
                    relative_vertex_weight.append(1.0 - 2 *  float(1)/8)
                else:
                    pos, posedir, shapedir, weight, relative_vertex, relative_vertex_weight = weightRingAround(vert, beta(vert), sd_verts)
                    relative_vertex_weight.append(1.0 - vert.valence * beta(vert))
                relative_vertex.append(i)
                
                new_verts[i] = SDVert(pos, posedir, shapedir, weight, relative_vertex, relative_vertex_weight, vert.boundary, vert.regular, vert.valence)
                
                vert.child = i
            
            #~ create new faces, compute new odd vertices and init new faces vertices and update new vertices startFace pointer
            edge_map = dict()
            v_index = vNum
            for i in range(fNum):
                face = sd_faces[i]
                #~ create 4 new successor faces
                for j in range(4):
                    new_faces[i*4 + j] = SDFace()
                    face.children[j] = i*4 + j
                
                for j in range(3):
                    edge = SDEdge(face.v[j], face.v[ NEXT(j) ])
                    
                     #~ compute new edge vertex if not computed before
                    if not edge in edge_map:
                        neighbour_face = face.nFaces[j]
                        if neighbour_face != -1:
                            # pos = (sd_verts[edge.v[0]].pos + sd_verts[edge.v[1]].pos) * (3.0 / 8.0) + (sd_verts[face.v[ PREV(j) ]].pos + sd_verts[ sd_faces[neighbour_face].otherVertex(face.v[j], face.v[ NEXT(j) ]) ].pos) * (1.0 / 8.0)
                            pos, posedir, shapedir, weight, relative_vertex, relative_vertex_weight = addVertRing(j, edge, face, neighbour_face, sd_verts, sd_faces)
                        else:
                            # pos = (sd_verts[edge.v[0]].pos + sd_verts[edge.v[1]].pos) * 0.5
                            pos, posedir, shapedir, weight, relative_vertex, relative_vertex_weight = addVertBoundary(edge, sd_verts)
                        new_verts[v_index] = SDVert(pos, posedir, shapedir, weight, relative_vertex, relative_vertex_weight, (neighbour_face == -1), True)
                        new_vert = new_verts[v_index]
                        new_vert.valence = 4 if new_vert.boundary else 6
                        new_vert.startFace = face.children[3]
                        
                        edge_map.setdefault(edge,v_index)
                        v_index +=1
                        
                    #~ set the corresonding vertex pointers in child faces
                    edge_vert_index = edge_map[edge]
                    new_faces[face.children[j]].v[j] = sd_verts[face.v[j]].child 
                    new_faces[face.children[j]].v[NEXT(j)] = edge_vert_index
                    new_faces[face.children[NEXT(j)] ].v[j] = edge_vert_index
                    new_faces[face.children[3]].v[j] = edge_vert_index
                    #~ set startFace on each even Vertex
                    new_verts[sd_verts[face.v[j]].child].startFace = face.children[j]
            edge_map.clear()
            
            #~ update new_faces neighbour_faces pointers and compute new vertex rings
            for i in range(fNum):
                face = sd_faces[i]
                for j in range(3):
                    #~ update child in the center neighbour_faces pointers
                    new_faces[face.children[3]].nFaces[j] = face.children[NEXT(j)]
                    new_faces[face.children[j]].nFaces[NEXT(j)] = face.children[3]
                    
                    neighbour_face = face.nFaces[j]
                    if neighbour_face != -1:
                        index = sd_faces[neighbour_face].vnum(face.v[j])
                        new_faces[face.children[j]].nFaces[j] = sd_faces[neighbour_face].children[index]
                        
                    neighbour_face = face.nFaces[ PREV(j) ]
                    if neighbour_face != -1:
                        index = sd_faces[neighbour_face].vnum(face.v[j])
                        new_faces[face.children[j]].nFaces[ PREV(j) ] = sd_faces[neighbour_face].children[index]
                        
            #~ prepare for next level of subdivision
            #~ compute new vertex rings
            for i in range(newVNum):
                set_vertex_ring(new_verts[i], i, new_faces)
            
            vNum = newVNum
            fNum = newFNum
            eNum = newENum
            sd_verts = new_verts
            sd_faces = new_faces
            
            mesh_list.append(convert(vNum , sd_verts, fNum, sd_faces, self.regs))
            
        return mesh_list


    def refine_one(self):
        vNum = self.vNum
        fNum = self.fNum
        eNum = self.eNum
        sd_verts = self.sd_verts
        sd_faces = self.sd_faces
        
        mesh_list = []

        mesh_list.append(convert(vNum , sd_verts, fNum, sd_faces, self.regs))
        
        newVNum = vNum + eNum
        newFNum = 4*fNum
        newENum = 2*eNum + 3*fNum
        # new_verts = (SDVert * newVNum)()
        # new_faces = (SDFace * newFNum)()
        new_verts = [0] * newVNum
        new_faces = [0] * newFNum
        
        #~ compute new even vertices
        for i in range(vNum):
            vert = sd_verts[i]
            if vert.boundary:
                pos, posedir, shapedir, weight, relative_vertex, relative_vertex_weight = weightBoundaryVert(vert, float(1)/8, sd_verts)
                relative_vertex_weight.append(1.0 - 2 *float(1)/8)
            else:
                pos, posedir, shapedir, weight, relative_vertex, relative_vertex_weight = weightRingAround(vert, beta(vert), sd_verts)
                relative_vertex_weight.append(1.0 - vert.valence * beta(vert))
            relative_vertex.append(i)
            new_verts[i] = SDVert(pos, posedir, shapedir, weight, relative_vertex, relative_vertex_weight, vert.boundary, vert.regular, vert.valence)
            
            vert.child = i
        
        #~ create new faces, compute new odd vertices and init new faces vertices and update new vertices startFace pointer
        edge_map = dict()
        if 1:
            edge_map_test = dict()
        v_index = vNum
        for i in range(fNum):
            face = sd_faces[i]
            #~ create 4 new successor faces
            for j in range(4):
                new_faces[i*4 + j] = SDFace()
                face.children[j] = i*4 + j
            
            for j in range(3):
                edge = SDEdge(face.v[j], face.v[ NEXT(j) ])
                
                    #~ compute new edge vertex if not computed before
                if not edge in edge_map:
                    neighbour_face = face.nFaces[j]
                    if neighbour_face != -1:
                        # pos = (sd_verts[edge.v[0]].pos + sd_verts[edge.v[1]].pos) * (3.0 / 8.0) + (sd_verts[face.v[ PREV(j) ]].pos + sd_verts[ sd_faces[neighbour_face].otherVertex(face.v[j], face.v[ NEXT(j) ]) ].pos) * (1.0 / 8.0)
                        pos, posedir, shapedir, weight, relative_vertex, relative_vertex_weight = addVertRing(j, edge, face, neighbour_face, sd_verts, sd_faces)
                    else:
                        # pos = (sd_verts[edge.v[0]].pos + sd_verts[edge.v[1]].pos) * 0.5
                        pos, posedir, shapedir, weight, relative_vertex, relative_vertex_weight = addVertBoundary(edge, sd_verts)
                    new_verts[v_index] = SDVert(pos, posedir, shapedir, weight, relative_vertex, relative_vertex_weight, (neighbour_face == -1), True)
                    new_vert = new_verts[v_index]
                    new_vert.valence = 4 if new_vert.boundary else 6
                    new_vert.startFace = face.children[3]
                    
                    edge_map.setdefault(edge,v_index)
                    v_index +=1
                else:
                    if 1:
                        res_test = edge_map_test.setdefault(edge,v_index)
                        if res_test != v_index:
                            print(v_index, res_test)
                    
                #~ set the corresonding vertex pointers in child faces
                edge_vert_index = edge_map[edge]
                new_faces[face.children[j]].v[j] = sd_verts[face.v[j]].child 
                new_faces[face.children[j]].v[NEXT(j)] = edge_vert_index
                new_faces[face.children[NEXT(j)] ].v[j] = edge_vert_index
                new_faces[face.children[3]].v[j] = edge_vert_index
                #~ set startFace on each even Vertex
                new_verts[sd_verts[face.v[j]].child].startFace = face.children[j]
        edge_map.clear()
        
        #~ update new_faces neighbour_faces pointers and compute new vertex rings
        for i in range(fNum):
            face = sd_faces[i]
            for j in range(3):
                #~ update child in the center neighbour_faces pointers
                new_faces[face.children[3]].nFaces[j] = face.children[NEXT(j)]
                new_faces[face.children[j]].nFaces[NEXT(j)] = face.children[3]
                
                neighbour_face = face.nFaces[j]
                if neighbour_face != -1:
                    index = sd_faces[neighbour_face].vnum(face.v[j])
                    new_faces[face.children[j]].nFaces[j] = sd_faces[neighbour_face].children[index]
                    
                neighbour_face = face.nFaces[ PREV(j) ]
                if neighbour_face != -1:
                    index = sd_faces[neighbour_face].vnum(face.v[j])
                    new_faces[face.children[j]].nFaces[ PREV(j) ] = sd_faces[neighbour_face].children[index]
                    
        #~ prepare for next level of subdivision
        #~ compute new vertex rings
        for i in range(newVNum):
            set_vertex_ring(new_verts[i], i, new_faces)
        
        vNum = newVNum
        fNum = newFNum
        eNum = newENum
        sd_verts = new_verts
        sd_faces = new_faces
        
        faceNum3, vertices, faces, posedir, shapedir, weights, relative_vertex, relative_vertex_weight, regressors = convert(vNum , sd_verts, fNum, sd_faces, self.regs)
        dd = {}
        dd['v_template'] = np.array(vertices)
        dd['f'] = np.array(faces)
        dd['shapedirs'] = np.array(shapedir)
        dd['posedirs'] = np.array(posedir)
        dd['weights'] = np.array(weights)
        dd['J_regressor'] = sparse.csc_matrix(regressors.T)
        dd['relative_vertex'] = relative_vertex
        dd['relative_vertex_weight'] = relative_vertex_weight
        
        return TriangleMesh(dd)
        # v = np.array(dd['v_template'])
        # f = np.array(dd['f'])
        # self.shapedirs = np.array(dd['shapedirs'])
        # self.posedirs = np.array(dd['posedirs'])
        # self.weights = np.array(dd['weights'])
        # self.regressors = dd['J_regressor'].A.T

def set_vertex_ring(vert, vert_index, sd_faces):
    ring = [0] * vert.valence #(int * vert.valence)()
    
    i = 0
    ring[i] = sd_faces[vert.startFace].nextVertex(vert_index)
    face_i = sd_faces[vert.startFace].nextFace(vert_index)
    i += 1
    while face_i != -1 and face_i != vert.startFace:
        ring[i] = sd_faces[face_i].nextVertex(vert_index)
        face_i = sd_faces[face_i].nextFace(vert_index)
        i += 1
            
    if vert.boundary:
        ring[0], ring[i - 1] = ring[i - 1], ring[0]
        face_i = vert.startFace
        
        while face_i != -1:
            ring[i] = sd_faces[face_i].prevVertex(vert_index)
            face_i = sd_faces[face_i].prevFace(vert_index)
            i += 1
    
    vert.ring = ring

def beta(vertex):
    if vertex.regular:
        return 1.0 / 16.0
    if vertex.valence == 3:
        return 3.0 / 16.0
    return  3.0 / (8.0 * vertex.valence)

def weightRingAround(vertex, beta, sd_verts):
    # pos
    pos = np.zeros(3, dtype=np.float) #Vector()
    for i in range(vertex.valence):
        pos += sd_verts[vertex.ring[i]].pos
    pos *= beta
    pos += vertex.pos * (1.0 - vertex.valence * beta)

    # shapedirs
    shapedir = np.zeros((3, 10), dtype=np.float) #Vector()
    for i in range(vertex.valence):
        shapedir += sd_verts[vertex.ring[i]].shapedir
    shapedir *= beta
    shapedir += vertex.shapedir * (1.0 - vertex.valence * beta)

    # posedirs
    posedir = np.zeros((3, 135), dtype=np.float) #Vector()
    for i in range(vertex.valence):
        posedir += sd_verts[vertex.ring[i]].posedir
    posedir *= beta
    posedir += vertex.posedir * (1.0 - vertex.valence * beta)

    # weights
    weight = np.zeros(16, dtype=np.float) #Vector()
    for i in range(vertex.valence):
        weight += sd_verts[vertex.ring[i]].weight
    weight *= beta
    weight += vertex.weight * (1.0 - vertex.valence * beta)
    # print(np.sum(weight))

    # relative vertex index
    relative_vertex = []
    relative_vertex_weight = []
    for i in range(vertex.valence):
        relative_vertex.append(vertex.ring[i])
        relative_vertex_weight.append(beta)

    # regressor
    # regressor = np.zeros(16, dtype=np.float) #Vector()
    # for i in range(vertex.valence):
    #     regressor += sd_verts[vertex.ring[i]].regressor
    # regressor *= beta
    # regressor += vertex.regressor * (1.0 - vertex.valence * beta)
    
    
    return pos, posedir, shapedir, weight, relative_vertex, relative_vertex_weight #, regressor

def weightBoundaryVert(vertex, beta, sd_verts):
    # pos
    pos = sd_verts[vertex.ring[0]].pos + sd_verts[vertex.ring[vertex.valence - 1]].pos
    pos *= beta
    pos += vertex.pos * (1.0 - 2 * beta)

    # shapedir
    shapedir = sd_verts[vertex.ring[0]].shapedir + sd_verts[vertex.ring[vertex.valence - 1]].shapedir
    shapedir *= beta
    shapedir += vertex.shapedir * (1.0 - 2 * beta)
    
    # posedir
    posedir = sd_verts[vertex.ring[0]].posedir + sd_verts[vertex.ring[vertex.valence - 1]].posedir
    posedir *= beta
    posedir += vertex.posedir * (1.0 - 2 * beta)

    # weight
    weight = sd_verts[vertex.ring[0]].weight + sd_verts[vertex.ring[vertex.valence - 1]].weight
    weight *= beta
    weight += vertex.weight * (1.0 - 2 * beta)
    #print(np.sum(weight))

    # relative vertex index
    relative_vertex = [vertex.ring[0], vertex.ring[vertex.valence - 1]]
    relative_vertex_weight = [beta, beta]

    # regressor
    # regressor = sd_verts[vertex.ring[0]].regressor + sd_verts[vertex.ring[vertex.valence - 1]].regressor
    # regressor *= beta
    # regressor += vertex.regressor * (1.0 - 2 * beta)

    return pos, posedir, shapedir, weight, relative_vertex, relative_vertex_weight#, regressor


def addVertRing(j, edge, face, neighbour_face, sd_verts, sd_faces):
    pos = (sd_verts[edge.v[0]].pos + sd_verts[edge.v[1]].pos) * (3.0 / 8.0) + \
        (sd_verts[face.v[ PREV(j) ]].pos + \
            sd_verts[ sd_faces[neighbour_face].otherVertex(face.v[j], face.v[ NEXT(j) ]) ].pos) * (1.0 / 8.0)

    posedir = (sd_verts[edge.v[0]].posedir + sd_verts[edge.v[1]].posedir) * (3.0 / 8.0) + \
        (sd_verts[face.v[ PREV(j) ]].posedir + \
            sd_verts[ sd_faces[neighbour_face].otherVertex(face.v[j], face.v[ NEXT(j) ]) ].posedir) * (1.0 / 8.0)
    
    shapedir = (sd_verts[edge.v[0]].shapedir + sd_verts[edge.v[1]].shapedir) * (3.0 / 8.0) + \
        (sd_verts[face.v[ PREV(j) ]].shapedir + \
            sd_verts[ sd_faces[neighbour_face].otherVertex(face.v[j], face.v[ NEXT(j) ]) ].shapedir) * (1.0 / 8.0)
        
    weight = (sd_verts[edge.v[0]].weight + sd_verts[edge.v[1]].weight) * (3.0 / 8.0) + \
        (sd_verts[face.v[ PREV(j) ]].weight + \
            sd_verts[ sd_faces[neighbour_face].otherVertex(face.v[j], face.v[ NEXT(j) ]) ].weight) * (1.0 / 8.0)

    # relative vertex index
    relative_vertex = [edge.v[0], edge.v[1], face.v[ PREV(j) ], sd_faces[neighbour_face].otherVertex(face.v[j], face.v[ NEXT(j) ])]
    relative_vertex_weight = [3.0 / 8.0, 3.0 / 8.0, 1.0 / 8.0, 1.0 / 8.0]
    # regressor = (sd_verts[edge.v[0]].regressor + sd_verts[edge.v[1]].regressor) * (3.0 / 8.0) + \
    #     (sd_verts[face.v[ PREV(j) ]].regressor + \
    #         sd_verts[ sd_faces[neighbour_face].otherVertex(face.v[j], face.v[ NEXT(j) ]) ].regressor) * (1.0 / 8.0)

    return pos, posedir, shapedir, weight, relative_vertex, relative_vertex_weight #, regressor


def addVertBoundary(edge, sd_verts):
    pos = (sd_verts[edge.v[0]].pos + sd_verts[edge.v[1]].pos) * 0.5
    posedir = (sd_verts[edge.v[0]].posedir + sd_verts[edge.v[1]].posedir) * 0.5
    shapedir = (sd_verts[edge.v[0]].shapedir + sd_verts[edge.v[1]].shapedir) * 0.5
    weight = (sd_verts[edge.v[0]].weight + sd_verts[edge.v[1]].weight) * 0.5
    relative_vertex = [edge.v[0], edge.v[1]]
    relative_vertex_weight = [0.5, 0.5]
    # regressor = (sd_verts[edge.v[0]].regressor + sd_verts[edge.v[1]].regressor) * 0.5

    return pos, posedir, shapedir, weight, relative_vertex, relative_vertex_weight #, regressor


def convert(vNum , sd_verts , fNum, sd_faces, regs):
    viNum = fNum*3
    v = [0] * vNum
    vi = [0] * viNum
    pd= [0] * vNum
    sd = [0] * vNum
    w = [0] * vNum
    rv = [0] * vNum
    rvw = [0] * vNum
    # reg = [0] * vNum
    for i in range(vNum):
        v[i] = sd_verts[i].pos
        pd[i] = sd_verts[i].posedir
        sd[i] = sd_verts[i].shapedir
        w[i] = sd_verts[i].weight
        rv[i] = sd_verts[i].relative_vertex
        #rv[i] = list(set(rv[i]))
        rvw[i] = sd_verts[i].relative_vertex_weight
        #rvw[i] = list(set(rvw[i]))
        # print('')
        #reg[i] = sd_verts[i].regressor
    
    # nomalize regressor
    # reg_sum = reg[0]
    # for i in range(1, vNum):
    #     reg_sum += reg[i]
    # for i in range(vNum):
    #     reg[i] = np.true_divide(reg[i], reg_sum)

    regs = np.pad(regs, ((0, vNum - regs.shape[0]), (0, 0)), 'constant', constant_values=0)

        
        
    for i in range(fNum):
        index = i*3
        vi[index] = sd_faces[i].v[0]
        vi[index + 1] = sd_faces[i].v[1]
        vi[index + 2] = sd_faces[i].v[2]
    
    # vn = shapes.get_normals(vNum, v, viNum, vi)
    # vni = (int * viNum)(*vi)
    # (faceNum * 3, vertices, faces, posedir, shapedir, weights, relative_vertex, regressors)
    return (viNum, v, vi, pd, sd, w, rv, rvw, regs) #, vn, vni) 

def subdivide(mesh, subdNum=1):
    subd_mesh = SDTriangleMesh(mesh, subdNum) #(mesh.vNum, mesh.v, mesh.viNum, mesh.vi, subdNum)
    return subd_mesh.refine()

def subdivide_one(mesh):
    subd_mesh = SDTriangleMesh(mesh) #(mesh.vNum, mesh.v, mesh.viNum, mesh.vi, subdNum)
    return subd_mesh.refine_one()


if __name__ == '__main__':
    import os

    subdNum = 4
    #mesh = _read_obj_file('subdivision/SampleFiles/bigguy.obj')
    mano_model = 'MANO_LEFT.pkl'
    #mano_model = 'MANO_RIGHT.pkl'
    mesh, dd = _read_mano(os.path.join(os.path.dirname(__file__), '..', 'mano', 'models', mano_model))

    mesh_refine = subdivide(mesh, subdNum)

    _write_network_obj('mano/test_hand.obj', 'mano/test_connect.obj', mesh_refine)

    # for k in range(subdNum):
    #     dir = 'mano/subdivision_models_%d' % (k + 1)
    #     os.makedirs(dir, exist_ok=True)
    #     _write_mano(os.path.join(dir, mano_model), mesh_refine[k], dd)
    #     _write_mano_obj(os.path.join(dir, 'test.obj'), mesh_refine[k])
    # print('')

    