from cv2 import threshold
from matplotlib.pyplot import axis
from pytorch3d.io import load_obj
from scipy import sparse
import scipy.sparse.linalg
import numpy as np
from scipy.linalg import eig
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.utils.vis import *
from tqdm import tqdm
import torch
import time
import random
import csv



def read_mesh(mesh_path):
    verts, faces, _ = load_obj(mesh_path)
    verts = verts.numpy()
    faces = faces[0].numpy()
    return verts, faces

def decomp(faces):
    data = []; row = []; col = []
    for _, triangle in enumerate(faces):
        for j in range(3):
            data.append(1.0); row.append(triangle[j]); col.append(triangle[(j + 1) % 3]);
            data.append(1.0); col.append(triangle[j]); row.append(triangle[(j + 1) % 3]);
    A = sparse.coo_matrix((data, (row, col)), dtype=float)
    A = A.todense().astype(np.float64) / 2
    L = A.copy()
    for i in range(L.shape[0]):
        L[i, i] = -np.sum(L[i])
    L = -L

    S = A.copy()
    for i in range(S.shape[0]):
        S[i, i] = 1
        S[i, :] /= np.sum(S[i])
    # np.set_printoptions(threshold=np.inf)
    # with open('test.txt', 'w+') as f:
    #     print(A, file=f)
    #test = L - L.T
    start = time.time()
    if 0:
        with torch.no_grad():
            L = torch.from_numpy(L).float().cuda()
            lmbd, U = torch.linalg.eig(L)
            lmbd = lmbd.detach().cpu().numpy()
            U = U.detach().cpu().numpy()
    else:
        lmbd, U = eig(L)
    end = time.time()
    lmbd = np.float64(lmbd)
    idx = np.argsort(lmbd)
    lmbd = sorted(lmbd)
    U = U[:, idx]
    return U, lmbd


def freq(mesh_path, U, lmbd, mano_mesh_path = None, rates = None, load_path=None): 
    verts, faces = read_mesh(mesh_path)
    if 0:
        U, lmbd = decomp(faces)
    # test = verts - U @ U.T @ verts
    #vecs = vecs.T
    verts_ori = verts.copy()
    amps = []; labels = []; mpve = []; ours = []
    smooth_range = 20
    if 1:
        for j in range(rates.shape[0]):
            #mpve = np.mean(np.linalg.norm(verts_ori - verts, axis=1))
            amp = []
            # if use_gpu:
            #     new_verts = torch.zeros(verts.shape).float()
            new_verts = np.zeros(verts.shape, dtype=np.float32)
            if 0:
                for i in range(len(lmbd)):
                    #a = U[:, i:i+1].dot(U[:, i:i+1].T)
                    a = U[:, i:i+1].T
                    vert = a.dot(verts)
                    if 0 and j == 1:
                        if i > 1000: #and i < 1000:
                            #vert /= pow(2,np.log10(i) - 2)
                            #vert *= np.mean(pow(2, np.array(amps[0][i-smooth_range: i+smooth_range])))/pow(2, amps[0][i])
                            vert *= pow(2, (random.random() - 0.5) * 1)
                        # elif i >= 1000:
                        #     vert /= 2*pow(10,np.log10(i) - 3)
                    if 1:
                        vert *= rates[j][i]
                    amp.append(np.log2(np.sqrt(np.sum(np.multiply(vert,vert), axis=1)))[0])
            
                    #amp.append(np.log2(np.mean(np.sqrt(np.sum(np.multiply(vert,vert), axis=1)))))
                    vert = U[:, i:i+1].dot(vert)
                    new_verts += vert
                    
                    if 1 and i in [5, 10, 20, 50, 100, 200, 500, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 11000, 12000, 12136, 12236, 12336]:
                    #if 0 and i in (list(range(100, 778))[::20] + [777]):
                        color_verts = (np.linalg.norm(vert, axis=1) / np.max(np.linalg.norm(vert, axis=1)) * 255).astype(np.int)
                        color_verts = np.stack((color_verts, 255 - color_verts, np.zeros(color_verts.shape, dtype=np.int32))).transpose()
                        #save_obj('mesh_color_%d.obj' % i, new_verts, faces, color_verts)
                        save_obj('output/freq/000/mesh_%d.obj' % i, new_verts, faces)
            else:
                new_verts = U.T @ verts
                new_verts *= rates[j]
                amp = np.log2(np.linalg.norm(new_verts, axis=1))
                
            
            # new_verts = np.zeros(verts.shape, dtype=np.float32)
            # for i in range(len(lmbd)):
            #     #a = U[:, i:i+1].dot(U[:, i:i+1].T)
            #     a = U[:, i:i+1].T
            #     vert = a.dot(verts)
            #     if 1 and j == 1:
            #         if i > 100:
            #             vert *= sum(amp[i-10: i+10])/20/amp[i]
            #     amp.append(np.log2(np.sqrt(np.sum(np.multiply(vert,vert), axis=1)))[0])
            #     vert = U[:, i:i+1].dot(vert)
            #     new_verts += vert
            # if j == 1:
            if j == 0:
                amp0 = amp.copy()
                f0 = new_verts.copy()
            #ours_j = np.mean(np.abs(np.array(amp0) - np.array(amp)))
            ours_j = np.mean(20 * np.log10(np.linalg.norm(new_verts, axis=1) / (np.linalg.norm(new_verts - f0, axis=1) + 1e-8)))
            new_verts = U @ new_verts
            #verts = S @ verts # + verts
            amps.append(amp)
            labels.append('A%d' % j)
            if 1:
                save_obj(os.path.join(load_path, 'mesh_A%d.obj' % j), new_verts, faces)
            mpve_j = np.mean(np.linalg.norm(verts_ori - new_verts, axis=1))
            mpve.append(mpve_j); ours.append(ours_j)
            #print(mpve_j, ours_j)
            filtered_verts = new_verts
    else:
        new_verts = U.T @ verts
        new_verts = rates * new_verts
        x = np.log10(np.linalg.norm(new_verts, axis=2) + 1e-8)
        y = np.log10(np.linalg.norm(new_verts - new_verts[0], axis=2) + 1e-8)
        ours_j = np.mean(20 * (x - y), axis=1)
        new_verts = U @ new_verts
        mpve_j = np.mean(np.linalg.norm(new_verts[0] - new_verts, axis=2), axis=1)
        #mpve.append(mpve_j); ours.append(ours_j)
        #print('')

    if 0:
        amp2 = []
        verts, faces = read_mesh(mano_mesh_path)
        new_verts = np.zeros(verts.shape, dtype=np.float32)
        for i in range(len(lmbd)):
            a = U[:, i:i+1].T
            vert = a.dot(verts)
            amp2.append(np.log2(np.sqrt(np.sum(np.multiply(vert,vert), axis=1)))[0])
            vert = U[:, i:i+1].dot(vert)
            new_verts += vert
        amps.append(amp2)
        labels.append('mano')
        mpve = np.mean(np.linalg.norm(filtered_verts - verts, axis=1))
        ours = np.mean(np.abs(np.array(amp0) - np.array(amp2)))
        print(mpve, ours)
    if 0:
        plot(np.array(amps[1:] + amps[:1]), os.path.join(load_path, "freq_splits.jpg"), labels=labels[1:] + labels[:1])
    #plot(amp, "freq_splits.jpg", "freq_splits")

    #usut = U

    # amp1 = []
    # test = U.T @ S @ U
    # #verts = S @ verts # + verts
    # verts[100, 1] = verts[100, 1] + 100
    # new_verts = np.zeros(verts.shape, dtype=np.float32)
    # for i in range(len(lmbd)):
    #     #a = U[:, i:i+1].dot(U[:, i:i+1].T)
    #     a = U[:, i:i+1].T
    #     vert = a.dot(verts)
    #     amp1.append(np.log2(np.sqrt(np.sum(np.multiply(vert,vert), axis=1)))[0])
    #     #amp1.append(np.log2(np.mean(np.sqrt(np.sum(np.multiply(vert,vert), axis=1)))))
    #     vert = U[:, i:i+1].dot(vert)
    #     new_verts += vert
    # save_obj('mesh.obj', new_verts, faces)

    # amp2 = []
    # verts[100, 1] = verts[100, 1] + 1000
    # new_verts = np.zeros(verts.shape, dtype=np.float32)
    # for i in range(len(lmbd)):
    #     a = U[:, i:i+1].T
    #     vert = a.dot(verts)
    #     amp2.append(np.log2(np.sqrt(np.sum(np.multiply(vert,vert), axis=1)))[0])
    #     vert = U[:, i:i+1].dot(vert)
    #     new_verts += vert


    # plot(np.array([amp,amp1,amp2]), "freq_splits.jpg", labels=['ori', 'add100', 'add1000'])
    # save_obj('mesh.obj', new_verts, faces)

    # test = np.divide(np.abs(np.array(amp) - np.array(amp1)), np.array(amp))
    # test = verts - new_verts
    return mpve_j, ours_j


if __name__ == "__main__":
    name = '0314_00'
    load_path = os.path.join("output/freq", name)
    os.makedirs(load_path, exist_ok=True)
    
    if 1:
        os.environ['CUDA_VISIBLE_DEVICES'] = '3'
        # use_gpu = 1
        scale = 2
        
        data_dir = 'data/DeepHandMesh/annotations/mano_para'
        mesh_detailed_path = os.path.join(data_dir, 'mesh_detailed_%d' % scale)
        mesh_mano_path = os.path.join(data_dir, 'mesh_mano_%d' % scale)
        num_vertices = 12337
        #max_rate = 1.5; min_rate = 0.5

        if 1:
            band_mids = []
            band_mid = 80
            while band_mid < 12237:
                band_mids.append(band_mid)
                band_mid *= 2
            num_exp = len(band_mids)
            #amps = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
            amps = [3.0]
            num_amp = len(amps)
            print(num_exp, num_amp)
            rates = np.ones([num_exp, num_amp, num_vertices, 3]).astype(np.float32)
            for i in range(0, num_exp):
                for j in range(0, num_amp):
                    band_min = int(band_mids[i] * 0.75)
                    band_max = min(int(band_mids[i] * 1.5), num_vertices)
                    # rates[i + 1, band_min:band_max, :] = (np.exp((np.random.rand(1, band_max - band_min, 3) - 0.5) * 4)) * rates[i + 1, band_min:band_max, :]
                    #rates[i, j, band_min:band_max, :] = ((np.random.rand(1, 1, band_max - band_min, 3) - 0.5) * 0.2 + 1) * amps[j] * rates[i, j, band_min:band_max, :]
                    rates[i, j, band_min:band_max, :] = ((np.random.rand(1, 1, band_max - band_min, 3) - 0.5) * amps[j] + 1) * rates[i, j, band_min:band_max, :]
        if 0:
            # sigma = np.array(list(range(num_vertices))[::-1]) / num_vertices * (max_rate - min_rate) + min_rate
            # sigma[:2500] = 1
            num_exp = 6
            rates = np.ones([num_exp + 1, num_vertices, 3]).astype(np.float32)
            for i in range(0, num_exp):
                band_min = 2500
                rates[i + 1, band_min:, :] = np.exp((np.random.rand(1, num_vertices - band_min, 3) - 0.5) * i) * rates[i + 1, band_min:, :]
        
        if 1:
            lmbd = np.load("assets/lmbd%d.npy" % scale)
            U = np.load("assets/U%d.npy" % scale)
        else:
            for detailed_file in tqdm(os.listdir(mesh_detailed_path)):
                detailed_file = os.path.join(mesh_detailed_path, detailed_file)
                verts, faces = read_mesh(detailed_file)
                U, lmbd = decomp(faces)
                if 1:
                    np.save("assets/lmbd%d.npy" % scale, lmbd)
                    np.save("assets/U%d.npy" % scale, U)
                    print('U and lmbd saved.')
                #break
            exit()
        # lmbd = np.float64(lmbd)
        # idx = np.argsort(lmbd)
        # lmbd = sorted(lmbd)
        # U = U[:, idx]
        # np.save("lmbd2.npy", lmbd)
        # np.save("U2.npy", U)
        # print('ok')
        # exit(0)

        

        
        # if use_gpu:
        #     U = torch.from_numpy(U).float().cuda()

        mpves = []; ourss = []
        #rates = np.stack(rates, axis=0)
        shape = rates.shape[0:2]
        rates = rates.reshape(-1, num_vertices, 3)
        rates = np.concatenate([np.ones([1, num_vertices, 3]).astype(np.float32), rates], axis=0)
        for mano_file in tqdm(os.listdir(mesh_mano_path)):
            num = int(mano_file[10:-4])
            detailed_file = "mesh_detailed_%d.obj" % num
            mano_file = os.path.join(mesh_mano_path, mano_file)
            detailed_file = os.path.join(mesh_detailed_path, detailed_file)
            mpve, ours = freq(detailed_file, U, lmbd, mano_file, rates, load_path)
            mpves.append(mpve); ourss.append(ours)
            #break
        mpves = np.array(mpves); ourss = np.array(ourss)
        np.save(os.path.join(load_path, "mpve.npy"), mpves); np.save(os.path.join(load_path, "ours.npy"), ourss)
        np.save(os.path.join(load_path, "rates.npy"), rates)
    else:
        mpves = np.load(os.path.join(load_path, "mpve.npy")); 
        ourss = np.load(os.path.join(load_path, "ours.npy")); 
    mpves = mpves.mean(axis=0); ourss = ourss.mean(axis=0)
    mpves = mpves[1:].reshape(shape); ourss = ourss[1:].reshape(shape)
    print(mpves); print(ourss)
    with open(os.path.join(load_path, "result_%s.csv" % name), 'w', newline='') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        spamwriter.writerows(mpves)
        spamwriter.writerows(ourss)
        # for i in range(mpves.shape[0]):
        #     spamwriter.writerows(mpves[i].tolist())
        # for j in range(ourss.shape[0]):
        #     spamwriter.writerows(ourss[j].tolist())
    pass