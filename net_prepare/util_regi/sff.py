"""
Abhilash Reddy Malipeddi
Calculation of curvature using the method outlined in Szymon Rusinkiewicz et. al 2004
Per face curvature is calculated and per vertex curvature is calculated by weighting the
per-face curvatures. I have vectorized the code where possible.
"""

#import numpy as np
import torch
from torch.nn.functional import normalize
# np.seterr(all='raise')

#from numpy.core.umath_tests import inner1d
#from .utils import  fastcross,normr
import time

def RotateCoordinateSystem(up,vp,nf):
    """
    RotateCoordinateSystem performs the rotation of the vectors up and vp
    to the plane defined by nf
    INPUT:
      up,vp  - vectors to be rotated (vertex coordinate system)  # (N, F, 3)
      nf     - face normal  # (N, F, 3)
    OUTPUT:
      r_new_u,r_new_v - rotated coordinate system  # (N, F, 3)
    """
    nrp=torch.cross(up,vp, dim=-1) # (N, F, 3)
    nrp=normalize(nrp, dim=-1)
    ndot=torch.sum(nf * nrp, dim=-1)  # (N, F)
    perp=nf-ndot.unsqueeze(-1)*nrp
    dperp=torch.div((nrp+nf),(1.0+ndot.unsqueeze(-1)))
    r_new_u=up-dperp*torch.sum(perp * up, dim=-1).unsqueeze(-1)
    r_new_v=vp-dperp*torch.sum(perp * vp, dim=-1).unsqueeze(-1)  # (N, F, 3)
    r_new_u = torch.where((ndot<=-1).unsqueeze(-1), -up, r_new_u)
    r_new_v = torch.where((ndot<=-1).unsqueeze(-1), -vp, r_new_v)

    return r_new_u,r_new_v

def ProjectCurvatureTensor(uf,vf,nf,old_ku,old_kuv,old_kv,up,vp):
    """
    ProjectCurvatureTensor performs a projection
    of the tensor variables to the vertexcoordinate system
    INPUT:
        uf,vf                 - face coordinate system          # (N, F, 3)
        nf                    - face normals                    # (N, F, 3)
        old_ku,old_kuv,old_kv - face curvature tensor variables # (N, F)
        up,vp                 - vertex cordinate system         # (N, F, 3)
    OUTPUT:
        new_ku,new_kuv,new_kv - vertex curvature tensor variables  # (N, F)
    """
    r_new_u,r_new_v = RotateCoordinateSystem(up,vp,nf)  # (N, F, 3)
    u1=torch.sum(r_new_u * uf, dim=-1)  # (N, F)
    v1=torch.sum(r_new_u * vf, dim=-1)
    u2=torch.sum(r_new_v * uf, dim=-1)
    v2=torch.sum(r_new_v * vf, dim=-1)
    new_ku  = u1*(u1*old_ku+v1*old_kuv) + v1*(u1*old_kuv+v1*old_kv )
    new_kuv = u2*(u1*old_ku+v1*old_kuv) + v2*(u1*old_kuv+v1*old_kv )
    new_kv  = u2*(u2*old_ku+v2*old_kuv) + v2*(u2*old_kuv+v2*old_kv )
    return new_ku,new_kuv,new_kv

def GetVertexNormalsExtra(vertices,faces,FaceNormals,e0,e1,e2):
    """
    In addition to vertex normals this also returns the mixed area weights per vertex
    which is used in calculating the curvature at the vertex from per face curvature values
    We could have calculated them separetely, but doing both at once is efficient. 
    The calculations involve loops over the faces and vertices in serial and are not easily vectorized 
    INPUT:
    Vertices       : vertices
    Faces          : vertex connectivity
    FaceNormals    : Outer Normal per face, having magnitude equal to area of face
    e0,e1,e2       : edge vectors # (N, F, 3)

    OUTPUT:
    VertNormals    :       Unit normal at the vertex
    wfp            :  Mixed area weights per vertex, as per Meyer 2002

    OTHER:
    Avertex        :     Mixed area associated with a vertex. Meyer 2002   # (N, V, 3)
    Acorner        :     part of Avertex associated to    # (N, F, 3)
    """

    #edge lengths
    de0=torch.sqrt(e0[...,0]**2+e0[...,1]**2+e0[...,2]**2 + 1e-12)  # (N, F)
    de1=torch.sqrt(e1[...,0]**2+e1[...,1]**2+e1[...,2]**2 + 1e-12)  # (N, F)
    de2=torch.sqrt(e2[...,0]**2+e2[...,1]**2+e2[...,2]**2 + 1e-12)  # (N, F)

    L2=torch.stack((de0**2,de1**2,de2**2), dim=-1) # (N, F, 3)

    ew=torch.stack((L2[...,0]*(L2[...,1]+L2[...,2]-L2[...,0]),L2[...,1]*(L2[...,2]+L2[...,0]-L2[...,1]),L2[...,2]*(L2[...,0]+L2[...,1]-L2[...,2])), dim=-1)  # (N, F, 3)

    #calculate face area
    #Af=torch.sqrt(FaceNormals[...,0]**2+FaceNormals[...,1]**2+FaceNormals[...,2]**2 + 1e-12)  # (N, F)
    Af=torch.norm(FaceNormals, dim=-1)
    Avertex       =torch.zeros(*(vertices.shape[:-1]), dtype=vertices.dtype).to(vertices.device)  # (N, V)
    VertNormals   =torch.zeros(vertices.shape, dtype=vertices.dtype).to(vertices.device)  # (N, V, 3)

    #Calculate weights according to N.Max [1999] for normals
    wfv1=FaceNormals/(L2[...,1]*L2[...,2] + 1e-12).unsqueeze(-1)  # (N, F, 3)
    wfv2=FaceNormals/(L2[...,2]*L2[...,0] + 1e-12).unsqueeze(-1)  # (N, F, 3)
    wfv3=FaceNormals/(L2[...,0]*L2[...,1] + 1e-12).unsqueeze(-1)  # (N, F, 3)



    # sf = faces.shape
    verts=faces[...,0]  # (N, F)
    for j in [0,1,2]:
      #VertNormals[...,j]+=torch.bincount(verts,minlength=vertices.shape[-2],weights=wfv1[...,j])  # (N, F)
      VertNormals[...,j].scatter_add_(dim=-1, index=verts, src=wfv1[...,j])  # (N, F)
    verts=faces[...,1]
    for j in [0,1,2]:
      #VertNormals[...,j]+=torch.bincount(verts,minlength=vertices.shape[-2],weights=wfv2[...,j])  # (N, F)
      VertNormals[...,j].scatter_add_(dim=-1, index=verts, src=wfv2[...,j])  # (N, F)
    verts=faces[...,2]
    for j in [0,1,2]:
      #VertNormals[...,j]+=torch.bincount(verts,minlength=vertices.shape[-2],weights=wfv3[...,j])  # (N, F)
      VertNormals[...,j].scatter_add_(dim=-1, index=verts, src=wfv3[...,j])  # (N, F)

    Acorner=(0.5*Af/(ew[...,0]+ew[...,1]+ew[...,2] + 1e-12)).unsqueeze(-1)*torch.stack((ew[...,1]+ew[...,2], ew[...,2]+ew[...,0], ew[...,0]+ew[...,1]), dim=-1)  # (N, F, 3)

    #Change the area to barycentric area for obtuse triangles
    Acorner_tmp = torch.zeros(Acorner.shape, dtype=Acorner.dtype).to(Acorner.device)  # (N, F, 3)

    Acorner_tmp[...,2]=-0.25*L2[...,1]*Af/(torch.sum(e0*e1, dim=-1) + 1e-12)
    Acorner_tmp[...,1]=-0.25*L2[...,2]*Af/(torch.sum(e0*e2, dim=-1) + 1e-12)
    Acorner_tmp[...,0]=Af-Acorner_tmp[...,1]-Acorner_tmp[...,2]
    Acorner = torch.where(ew[...,0:1]<=0, Acorner_tmp, Acorner)

    Acorner_tmp[...,2]=-0.25*L2[...,0]*Af/(torch.sum(e1*e0, dim=-1) + 1e-12)
    Acorner_tmp[...,0]=-0.25*L2[...,2]*Af/(torch.sum(e1*e2, dim=-1) + 1e-12)
    Acorner_tmp[...,1]=Af-Acorner_tmp[...,0]-Acorner_tmp[...,2]
    Acorner = torch.where(ew[...,1:2]<=0, Acorner_tmp, Acorner)
    
    Acorner_tmp[...,0]=-0.25*L2[...,1]*Af/(torch.sum(e2*e1, dim=-1) + 1e-12)
    Acorner_tmp[...,1]=-0.25*L2[...,0]*Af/(torch.sum(e2*e0, dim=-1) + 1e-12)
    Acorner_tmp[...,2]=Af-Acorner_tmp[...,0]-Acorner_tmp[...,1]
    Acorner = torch.where(ew[...,2:3]<=0, Acorner_tmp, Acorner)
        

#Accumulate Avertex from Acorner.
    # for j,verts in enumerate(faces.T):
    #    Avertex+=torch.bincount(verts,minlength=vertices.shape[-2],weights=Acorner[...,j])
    for j in range(3):
        Avertex.scatter_add_(dim=-1, index=faces[...,j], src=Acorner[...,j])  # (N, F)

    VertNormals=normalize(VertNormals, dim=-1)    # (N, V, 3)

    #calculate voronoi weights
    #wfp=Acorner/Avertex[faces]   # (N, F, 3)
    if len(faces.shape) == 2:
        wfp=Acorner/Avertex[faces]   # (F, 3)
    else:
        wfp = torch.stack([Avertex[idx][faces[idx]] for idx in range(faces.shape[0])], dim=0)
        wfp = Acorner / wfp
    return VertNormals,wfp

def CalcCurvature(vertices,faces, file=None):
    """
    CalcCurvature recives a list of vertices and faces
    and the normal at each vertex and calculates the second fundamental
    matrix and the curvature by least squares, by inverting the 3x3 Normal matrix
    INPUT:
    vertices  -nX3 array of vertices
    faces     -mX3 array of faces
    VertexNormals - nX3 matrix (n=number of vertices) containing the normal at each vertex
    FaceNormals - mX3 matrix (m = number of faces) containing the normal of each face
    OUTPUT:
    FaceSFM   - a list of 2x2 np arrays of (m = number of faces) second fundamental tensor at the faces
    VertexSFM - a list of 2x2 np arrays (n = number of vertices) second fundamental tensor at the vertices

    Other Parameters
    wfp     : mx3 array of vertex voronoi cell area/Mixed area weights as given in Meyer 2002
    up,vp   : local coordinate system at each vertex
    e0,e1,e2       : edge vectors
    """
    #list of 2x2 arrays for each vertex
    #print('%s %s: %s' % (time.asctime(time.localtime(time.time())), file, 'start calculating.'))
    #VertexSFM = [np.zeros([2,2]) for i in vertices]
    VertexSFM = torch.zeros([*(vertices.shape[:-1]), 2, 2], dtype=vertices.dtype).to(vertices.device) # (N, V, 2, 2)
    up        = torch.zeros(vertices.shape, dtype=vertices.dtype).to(vertices.device) # (N, V, 3)

    bound1 = faces.shape[-2] // 3; bound2 = faces.shape[-2] * 2 // 3
    vert3x3 = []
    vert0x3 = torch.cat(3*[faces[...,0:1]], dim=-1); vert3x3.append([vert0x3[...,:bound1,:], vert0x3[...,bound1:bound2,:], vert0x3[...,bound2:,:]])
    vert1x3 = torch.cat(3*[faces[...,1:2]], dim=-1); vert3x3.append([vert1x3[...,:bound1,:], vert1x3[...,bound1:bound2,:], vert1x3[...,bound2:,:]])
    vert2x3 = torch.cat(3*[faces[...,2:3]], dim=-1); vert3x3.append([vert2x3[...,:bound1,:], vert2x3[...,bound1:bound2,:], vert2x3[...,bound2:,:]])
    for i in range(3):
        for j in range(3):
            chk = torch.nonzero(torch.where(vert3x3[j][i] >= vertices.shape[1], 1, 0))
            if chk.nelement() != 0:
                print(chk.detach().cpu().numpy())
    
    e0=torch.stack([vertices[i,faces[i,:,2]]-vertices[i,faces[i,:,1]] for i in range(vertices.shape[0])], dim=0)  # (N, F, 3) for i in range(vertices.shape[0]):
    e1=torch.stack([vertices[i,faces[i,:,0]]-vertices[i,faces[i,:,2]] for i in range(vertices.shape[0])], dim=0)  # (N, F, 3)
    e2=torch.stack([vertices[i,faces[i,:,1]]-vertices[i,faces[i,:,0]] for i in range(vertices.shape[0])], dim=0)  # (N, F, 3)
    # e0 = torch.cat([vertices.gather(-2, vert3x3[2][i])-vertices.gather(-2, vert3x3[1][i]) for i in range(3)], dim=-2)  # (N, F, 3)
    # e1 = torch.cat([vertices.gather(-2, vert3x3[0][i])-vertices.gather(-2, vert3x3[2][i]) for i in range(3)], dim=-2)  # (N, F, 3)
    # e2 = torch.cat([vertices.gather(-2, vert3x3[1][i])-vertices.gather(-2, vert3x3[0][i]) for i in range(3)], dim=-2)  # (N, F, 3)
    #print('%s %s: %s' % (time.asctime(time.localtime(time.time())), file, 'test.'))
    e0_norm=normalize(e0, dim=-1)  # (N, F, 3)
    e1_norm=normalize(e1, dim=-1)  # (N, F, 3)
    e2_norm=normalize(e2, dim=-1)  # (N, F, 3)

    FaceNormals=0.5*torch.cross(e1,e2, dim=-1) # (N, F, 3) not unit length. holds the area which is needed next
    VertNormals,wfp=GetVertexNormalsExtra(vertices,faces,FaceNormals,e0,e1,e2)

    FaceNormals=normalize(FaceNormals, dim=-1) # (N, F, 3)

    #Calculate initial coordinate system
    # up[faces[...,0]]=e2_norm  # (N, V, 3)
    # up[faces[...,1]]=e0_norm  # (N, V, 3)
    # up[faces[...,2]]=e1_norm  # (N, V, 3)
    # for i in range(3):
    #     up.scatter_add_(dim=-2, index=vert3x3[0][i], src=e2_norm) 
    #     up.scatter_add_(dim=-2, index=vert3x3[1][i], src=e0_norm)
    #     up.scatter_add_(dim=-2, index=vert3x3[2][i], src=e1_norm)
    up.scatter_add_(dim=-2, index=vert0x3, src=e2_norm) 
    up.scatter_add_(dim=-2, index=vert1x3, src=e0_norm)
    up.scatter_add_(dim=-2, index=vert2x3, src=e1_norm)

    #Calculate initial vertex coordinate system
    up=torch.cross(up,VertNormals, dim=-1)  # (N, V, 3)
    up=normalize(up, dim=-1)  # (N, V, 3)
    vp=torch.cross(VertNormals,up, dim=-1)  # (N, V, 3)

    M=normalize(torch.cross(FaceNormals,e0_norm, dim=-1), dim=-1)  # (N, F, 3)

    #nfaces=faces.shape[-2]
    #print('%s %s: %s' % (time.asctime(time.localtime(time.time())), file, 'normal done.'))

# Build a least square problem at each face to get the SFM at each face and solve it using the normal equation
    scale=1.0/torch.sqrt(torch.sum((e0[...,0,:]**2+e1[...,0,:]**2+e2[...,0,:]**2)/3.0, dim=-1) + 1e-12)
    
    X0 = torch.stack([
        torch.stack([torch.sum(e0*e0_norm, dim=-1), torch.sum(e0*M, dim=-1), torch.zeros(faces.shape[:-1], dtype=e0.dtype).to(e0.device)], dim=-1),
        torch.stack([torch.zeros(faces.shape[:-1], dtype=e0.dtype).to(e0.device), torch.sum(e0*e0_norm, dim=-1), torch.sum(e0*M, dim=-1)], dim=-1),
        torch.stack([torch.sum(e1*e0_norm, dim=-1), torch.sum(e1*M, dim=-1), torch.zeros(faces.shape[:-1], dtype=e0.dtype).to(e0.device)], dim=-1),
        torch.stack([torch.zeros(faces.shape[:-1], dtype=e0.dtype).to(e0.device), torch.sum(e1*e0_norm, dim=-1), torch.sum(e1*M, dim=-1)], dim=-1),
        torch.stack([torch.sum(e2*e0_norm, dim=-1), torch.sum(e2*M, dim=-1), torch.zeros(faces.shape[:-1], dtype=e0.dtype).to(e0.device)], dim=-1),
        torch.stack([torch.zeros(faces.shape[:-1], dtype=e0.dtype).to(e0.device), torch.sum(e2*e0_norm, dim=-1), torch.sum(e2*M, dim=-1)], dim=-1),
        ], dim=-1).transpose(-2, -1)  # (N, F, 6, 3)
    if len(X0.shape) == 4:
        scale = scale[:, None, None, None]
    X0 = scale * X0

    #A  = np.transpose(AT,axes=(0,2,1)).copy()

    # dn0=VertNormals[faces[...,2]]-VertNormals[faces[...,1]] # (N, F, 3)
    # dn1=VertNormals[faces[...,0]]-VertNormals[faces[...,2]]
    # dn2=VertNormals[faces[...,1]]-VertNormals[faces[...,0]]
    dn0=torch.cat([VertNormals.gather(-2, vert3x3[2][i])-VertNormals.gather(-2, vert3x3[1][i]) for i in range(3)], dim=-2) # (N, F, 3)
    dn1=torch.cat([VertNormals.gather(-2, vert3x3[0][i])-VertNormals.gather(-2, vert3x3[2][i]) for i in range(3)], dim=-2)
    dn2=torch.cat([VertNormals.gather(-2, vert3x3[1][i])-VertNormals.gather(-2, vert3x3[0][i]) for i in range(3)], dim=-2)

    b=  scale*torch.stack([ torch.sum(dn0*e0_norm, dim=-1),
                            torch.sum(dn0*M, dim=-1),
                            torch.sum(dn1*e0_norm, dim=-1),
                            torch.sum(dn1*M, dim=-1),
                            torch.sum(dn2*e0_norm, dim=-1),
                            torch.sum(dn2*M, dim=-1)
                        ], dim=-1).unsqueeze(-1)  # (N, F, 6, 1)

    #X1=np.array([np.linalg.pinv(a,-1) for a in A])
    # X = torch.matmul(X0.transpose(-2, -1), X0)
    # a1=X[...,0,0]; b1=X[...,0,1]; c1=X[...,0,2];
    # a2=X[...,1,0]; b2=X[...,1,1]; c2=X[...,1,2];
    # a3=X[...,2,0]; b3=X[...,2,1]; c3=X[...,2,2];
    # X[...,0,0]=b2*c3-c2*b3; X[...,0,1]=c1*b3-b1*c3; X[...,0,2]=b1*c2-c1*b2;
    # X[...,1,0]=c2*a3-a2*c3; X[...,1,1]=a1*c3-c1*a3; X[...,1,2]=a2*c1-a1*c2;
    # X[...,2,0]=a2*b3-b2*a3; X[...,2,1]=b1*a3-a1*b3; X[...,2,2]=a1*b2-a2*b1;
    # X = X * (1/(a1*(b2*c3-c2*b3) - a2*(b1*c3-c1*b3) + a3*(b1*c2-c1*b2))).unsqueeze(-1).unsqueeze(-1)
    # X = torch.matmul(X, A.transpose(-2, -1))

    X = torch.matmul(X0.transpose(-2, -1), X0)
    A=X[...,0,0].clone(); B=X[...,0,1].clone(); C=X[...,0,2].clone();
    D=X[...,1,0].clone(); E=X[...,1,1].clone(); F=X[...,1,2].clone();
    G=X[...,2,0].clone(); H=X[...,2,1].clone(); I=X[...,2,2].clone();
    X[...,0,0]=+((E*I)-(F*H)); X[...,0,1]=-((D*I)-(F*G)); X[...,0,2]=+((D*H)-(G*E));
    X[...,1,0]=-((B*I)-(C*H)); X[...,1,1]=+((A*I)-(C*G)); X[...,1,2]=-((A*H)-(B*G));
    X[...,2,0]=+((B*F)-(C*E)); X[...,2,1]=-((A*F)-(C*D)); X[...,2,2]=+((A*E)-(B*D));
    X = X * (1/(+(A*((E*I)-(F*H)))-(B*((D*I)-(F*G)))+(C*((D*H)-(G*E))))).unsqueeze(-1).unsqueeze(-1)
    X = torch.matmul(X, X0.transpose(-2, -1))

    #X = torch.matmul(X, X0)
    #X = torch.linalg.pinv(X0) # (N, F, 3, 6)
    X  = torch.matmul(X,b).squeeze(-1) # (N, F, 3)

    #print('%s %s: %s' % (time.asctime(time.localtime(time.time())), file, 'matrix solved.'))

#now calculate curvature per vertex as weighted sum of the face curvature
    # for i,f in enumerate(faces):
    # #   if i % 100000 == 0:
    # #       print('%s %s: %s %d/1000000' % (time.asctime(time.localtime(time.time())), file, 'curvatrue computing', i))
    
    for j in [0,1,2]:
        up_f = torch.cat([up.gather(-2, vert3x3[j][i]) for i in range(3)], dim=-2)
        vp_f = torch.cat([vp.gather(-2, vert3x3[j][i]) for i in range(3)], dim=-2)
        new_ku,new_kuv,new_kv = ProjectCurvatureTensor(e0_norm, M, FaceNormals, X[...,0], X[...,1], X[...,2], up_f, vp_f)
        VertexSFM[..., 0, 0].scatter_add_(dim=-1, index=faces[...,j], src=wfp[...,j]*new_ku)
        VertexSFM[..., 0, 1].scatter_add_(dim=-1, index=faces[...,j], src=wfp[...,j]*new_kuv)
        VertexSFM[..., 1, 1].scatter_add_(dim=-1, index=faces[...,j], src=wfp[...,j]*new_kv)
    VertexSFM[..., 1, 0] = VertexSFM[..., 0, 1]

    #print('%s %s: %s' % (time.asctime(time.localtime(time.time())), file, 'curvatrue done.'))
    return VertexSFM,VertNormals

    

def GetCurvatures(vertices,faces, file=None):
    """
    INPUT : vertices,faces
    OUTPUT: Gaussian Curvature, Mean Curvature
    """

    VertexSFM,VertNormals=CalcCurvature(vertices,faces,file)

    ku  = VertexSFM[..., 0, 0]
    kuv = VertexSFM[..., 0, 1]
    kv  = VertexSFM[..., 1, 1]

    return VertexSFM,(ku*kv-kuv**2),0.5*(ku+kv),VertNormals

