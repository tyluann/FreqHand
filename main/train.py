# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import argparse
import enum
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
#from main.config import cfg
import pymeshlab
import shutil
import json
import threading
from multiprocessing import Process
from main import config as cfg
import torch
import os
import cProfile, pstats, io
from main.base import Trainer, collate_batched_multi_datasets
import torch.backends.cudnn as cudnn
from tqdm import tqdm
import time
import pymeshlab
from itertools import cycle
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from pytorch3d.io import load_obj
#from main.model_s2hand.model import feature_grad
from common.utils.vis import *
use_wandb = 0
if use_wandb:
    import wandb
    wandb.login()

# # wandb.config = {
# #   "learning_rate": 0.001,
# #   "epochs": 100,
# #   "batch_size": 128
# # }

# def parse_args():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--gpu', default='0-3', type=str, dest='gpu_ids')
#     #parser.add_argument('--continue', dest='continue_train', action='store_true')
#     parser.add_argument('--subject', default='4', type=str, dest='subject')
#     args = parser.parse_args()

#     if not args.gpu_ids:
#         assert 0, "Please set propoer gpu ids"

#     if '-' in args.gpu_ids:
#         gpus = args.gpu_ids.split('-')
#         gpus[0] = int(gpus[0])
#         gpus[1] = int(gpus[1]) + 1
#         args.gpu_ids = ','.join(map(lambda x: str(x), list(range(*gpus))))
    
#     assert args.subject, 'Training subject is required'
#     assert args.subject == '4', 'Training only supports subject_4'
#     return args
def subdivison(verts, faces, subnum):
    if subnum == 0:
        return verts, faces
    ms = pymeshlab.MeshSet()
    mesh = pymeshlab.Mesh(vertex_matrix=verts, face_matrix=faces)
    ms.add_mesh(mesh)
    ms.meshing_surface_subdivision_loop(iterations=subnum, threshold=pymeshlab.Percentage(0))
    new_verts = ms.current_mesh().vertex_matrix()
    new_faces = ms.current_mesh().face_matrix()
    return new_verts, new_faces

def save_codes(out_dir):
    os.makedirs(out_dir, exist_ok=True)
    main_dir = '.'
    for root, dirs, files in os.walk(main_dir):
        if 'output_new' in root.split('/'):
        #if root.startswith(os.path.join(main_dir, 'output_new')):
            continue
        for name in files:
            if name[-3:] == '.py':
                relative_path = root[len(main_dir):]
                if len(relative_path) > 0:
                    relative_path = relative_path[1:]
                os.makedirs(os.path.join(out_dir, relative_path), exist_ok=True)
                shutil.copy(os.path.join(root, name), os.path.join(out_dir, relative_path, name))
    paras = {}
    for item in dir(cfg.cfg):
        if item[0:2] != '__':
            paras[item] = getattr(cfg.cfg, item)
    with open(os.path.join(out_dir, 'config.json'), 'w') as f:
        json.dump(paras, f, indent=4)

def test_vis(loader_type, out, targets, inputs, model, itr, idx):
    for j in range(out['mesh_out'].shape[0]):
        if (not cfg.cfg.vis_all) and j != idx:
            continue
        file = os.path.join(cfg.cfg.vis_dir, str(targets['img_id'][j].item()) + '_%d_%d_' % (itr, idx))
        save_obj(file + loader_type + '_ori.obj', out['mesh_out'][j].detach().cpu().numpy(), 
            model.model_refine.faces0.detach().cpu().numpy())
        img_out = (np.transpose(inputs['img'][j].detach().cpu().numpy(), (1, 2, 0))[:,:,::-1] * 255).astype(np.uint8)
        cv2.imwrite(file + loader_type + '_img.png', img_out)
        for i in range(3):
            faces =  model.model_refine.get_buffer('faces%d' % i).detach().cpu().numpy()
            #save_obj(file + loader_type + '_ori_%d.obj' % i, out['mesh_out'][0].detach().cpu().numpy(), faces)
            if not cfg.cfg.vis_multithread:
                save_obj(file + loader_type + '_out_%d.obj' % i, out['vertices_list'][i][j].detach().cpu().numpy(), faces)
                save_obj(file + loader_type + '_gt_%d.obj' % i, targets['gt_mesh'][i][0][j].numpy(), faces)
            else:
                t1 = Process(target=save_obj, args=(file + loader_type + '_out_%d.obj' % i, out['vertices_list'][i][j].detach().cpu().numpy(), faces)); t1.start()
                t2 = Process(target=save_obj, args=(file + loader_type + '_gt_%d.obj' % i, targets['gt_mesh'][i][0][j].detach().cpu().numpy(), faces)); t2.start()
            if 0 and (not cfg.cfg.vis_all):
                U = model.get_buffer('U%d' % i).detach().cpu().numpy()
                freq_out = np.log2(np.linalg.norm(U.T @ out['vertices_list'][i][j].detach().cpu().numpy(), axis=1))
                freq_gt = np.log2(np.linalg.norm(U.T @ targets['gt_mesh'][i][0][j].numpy(), axis=1))
                if 1:
                    with open(file + loader_type + 'freq_gt_%d.csv' % i, 'w') as f:
                        for num in freq_gt:
                            print(num, file=f)

                if not cfg.cfg.vis_multithread:
                    plot(np.stack([freq_out, freq_gt], axis=0), file + loader_type + '_freq_%d.png' % i, labels=['freq_out', 'freq_gt'], title='')
                else:
                    t = Process(target=plot, args=(np.stack([freq_out, freq_gt], axis=0), file + loader_type + '_freq_%d.png' % i, ['freq_out', 'freq_gt'])); t.start()
            if 0 and (not cfg.cfg.vis_all):
                U = model.get_buffer('U2').detach().cpu().numpy()
                # out_verts, out_faces = subdivison(out['vertices_list'][i][j].detach().cpu().numpy(), faces, 2-i)
                # d_f = (out_faces - model.model_refine.get_buffer('faces2').detach().cpu().numpy())
                freq_out = np.log2(np.linalg.norm(U.T @ out['vertices_ori_list'][i][j].cpu().numpy(), axis=1))
                freq_gt = np.log2(np.linalg.norm(U.T @ targets['gt_mesh'][2][0][j].numpy(), axis=1))
                if 1:
                    with open(file + loader_type + 'freq_gt_%d.csv' % i, 'w') as f:
                        for num in freq_gt:
                            print(num, file=f)

                if not cfg.cfg.vis_multithread:
                    plot(np.stack([freq_out, freq_gt], axis=0), file + loader_type + '_freq_%d.png' % i, labels=['freq_out', 'freq_gt'], title='')
                    save_obj(file + loader_type + '_outsub_%d.obj' % i, out['vertices_ori_list'][i][j].cpu().numpy(), model.model_refine.faces2.detach().cpu().numpy())
                else:
                    t = Process(target=plot, args=(np.stack([freq_out, freq_gt], axis=0), file + loader_type + '_freq_%d.png' % i, ['freq_out', 'freq_gt'])); t.start()
                    t1 = Process(target=save_obj, args=(file + loader_type + '_outsub_%d.obj' % i, out['vertices_ori_list'][i][j].cpu().numpy(), model.model_refine.faces2.detach().cpu().numpy())); t1.start()
        # if not cfg.cfg.vis_all:
        #     break

# def smoothing(verts, mask_face):
#     ms = pymeshlab.MeshSet()
#     mesh = pymeshlab.Mesh(vertex_matrix=verts.numpy(), face_matrix=mask_face)
#     ms.add_mesh(mesh)
#     ms.apply_coord_taubin_smoothing(lambda_=0.4, mu=-0.42, stepsmoothnum=12, selected=False)
#     new_verts = ms.current_mesh().vertex_matrix()
#     return new_verts

def test_epoch(trainer, loader_type, dataset):
    if loader_type == 'eval':
        trainer.model.module.mode = 'test'
    if loader_type + '_' + dataset not in trainer.batch_generator:
        trainer._make_batch_generator(loader_type, dataset)
        print("make batch generator!")

    print('test on %s' % (loader_type))

    trainer.model.eval()
    trainer.tot_timer.tic()
    loss_mean = {}
    loss_cases_count = {}
    # loss_mean_t = {}; loss_cases_count_t = {}
    # if cfg.cfg.post_smooth:
    #     _, mask_face, _ = load_obj('assets/mask.obj')

    with torch.no_grad():
        # with tqdm(enumerate(trainer.batch_generator[loader_type + '_' + dataset]), 
        #     desc='Total iterations %d   ' % (trainer.itr_per_epoch[loader_type + '_' + dataset])) as tbar:
        #     for itr, (inputs, targets, meta_info) in tbar:
        if 1:
            for itr, (inputs, targets, meta_info) in enumerate(trainer.batch_generator[loader_type + '_' + dataset]):
                # forward
                out, train_loss, test_loss = trainer.model(inputs, targets, meta_info)

                
                # if cfg.cfg.post_smooth:
                #     new_vert = smoothing(out['vertices_list'][2][0].detach().cpu().numpy(), mask_face[0].numpy())
                #     test_loss['mpve_%d' % i] = torch.mean(torch.sqrt(torch.sum((out['vertices_list'][i]- gt_mesh.verts_padded())**2, dim=2)), dim=1)



                if 0:
                    start = time.time()
                    #file = os.path.join(cfg.cfg.vis_dir, str(targets['img_id'][0].item()))
                    if cfg.cfg.mesh_refine:
                        #if not cfg.cfg.vis_multithread:
                        # else:
                        for i, idx in enumerate(targets['img_id']):
                            if cfg.cfg.vis_all or idx.item() in [3801, 1015, 1039, 1275, 3825]:
                                test_vis(loader_type, out, targets, inputs, trainer.model.module, itr, i)
                        #     t = Process(target=test_vis, args=(file, loader_type, out, targets, trainer.model.module))
                        #     #t = threading.Thread(target=test_vis, args=(file, loader_type, out, targets, trainer.model.module), name='test_vis')
                        #     t.start()
                    end = time.time()
                    #print("vis time: ", end-start)
                        # save_obj(file + loader_type + '_ori.obj', out['mesh_out'][0].detach().cpu().numpy(), 
                        #     trainer.model.module.model_refine.faces0.detach().cpu().numpy())
                        # for i in range(3):
                        #     faces =  trainer.model.module.model_refine.get_buffer('faces%d' % i).detach().cpu().numpy()
                        #     #save_obj(file + loader_type + '_ori_%d.obj' % i, out['mesh_out'][0].detach().cpu().numpy(), faces)
                        #     save_obj(file + loader_type + '_out_%d.obj' % i, out['vertices_list'][i][0].detach().cpu().numpy(), faces)
                        #     save_obj(file + loader_type + '_gt_%d.obj' % i, targets['gt_mesh'][i][0][0].detach().cpu().numpy(), faces)
                        #     U = trainer.model.module.get_buffer('U%d' % i).detach().cpu().numpy()
                        #     freq_out = np.log2(np.linalg.norm(U.T @ out['vertices_list'][i][0].detach().cpu().numpy(), axis=1))
                        #     freq_gt = np.log2(np.linalg.norm(U.T @ targets['gt_mesh'][i][0][0].detach().cpu().numpy(), axis=1))
                        #     plot(np.stack([freq_out, freq_gt], axis=0), file + loader_type + '_freq_%d.png' % i, labels=['freq_out', 'freq_gt'], title='')

                # only use test losses
                for key in test_loss:
                    test_loss[key] = test_loss[key].detach().cpu()
                    total_key = key
                    #total_key = 'test_loss_' + key
                    if total_key not in loss_mean:
                        loss_mean[total_key] = 0
                        loss_cases_count[total_key] = 0
                        # if 1:
                        #     loss_mean_t[total_key] = 0
                        #     loss_cases_count_t[total_key] = 0
                    loss_mean[total_key] = (loss_mean[total_key] * loss_cases_count[total_key] + \
                        test_loss[key].sum()) / (loss_cases_count[total_key] + test_loss[key].shape[0])
                    loss_cases_count[total_key] += test_loss[key].shape[0]
                    # if 1:
                    #     loss_mean_t[total_key] += test_loss[key].sum()
                    #     loss_cases_count_t[total_key] += test_loss[key].shape[0]

                # tbar.set_postfix(ordered_dict={'mean_' + total_key: str(loss_mean[total_key].detach().item())})
                # tbar.update()
            # if 1:
            #     for total_key in loss_mean:
            #         print(total_key, (loss_mean_t[total_key] / loss_cases_count_t[total_key]).detach().item())
    
    trainer.tot_timer.toc()
    trainer.model.train()
    trainer.model.module.mode = 'train'
    for key in loss_mean:
        print(key, loss_mean[key].item())
    return loss_mean

def plot_grad_flow(named_parameters, epoch):
    #print(cfg.cfg.feature_grad.abs().mean().cpu())
    ave_grads = []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            layers.append(n)
            if p.grad is None:
                ave_grads.append(0)
            else:
                ave_grads.append(p.grad.abs().mean().cpu())
    plt.figure(figsize=(100, 50))
    plt.plot(ave_grads, alpha=0.3, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, linewidth=1, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(xmin=0, xmax=len(ave_grads))
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    os.makedirs(cfg.cfg.grad_dir, exist_ok=True)
    plt.savefig(os.path.join(cfg.cfg.grad_dir, 'grad_%d.png' % epoch))

def main():
    
    # argument parse and create log
    # args = parse_args()
    # cfg.cfg.set_args(args.subject, args.gpu_ids) #, args.continue_train)
    # os.environ["CUDA_VISIBLE_DEVICES"] = cfg.cfg.gpu_ids
    cudnn.benchmark = True

    #dir_time = time.strftime('%m%d_%H%M', time.localtime(time.time()))
    #summary_dir = os.path.join(cfg.cfg.summary_dir, dir_time)
    #grad_dir = os.path.join(cfg.cfg.grad_dir, dir_time)
    #vis_dir = os.path.join(cfg.cfg.vis_dir, dir_time)
    for key in cfg.cfg.output_dirs:
        os.makedirs(os.path.join(cfg.cfg.output_dir, cfg.cfg.dir_name, key), exist_ok=True)
    if use_wandb:
        wandb.init(project=cfg.cfg.dir_name, entity="tyluan")

    if cfg.cfg.redirect:
        f = open(os.path.join(cfg.cfg.output_dir, cfg.cfg.dir_name, cfg.cfg.dir_name + '.csv'), 'w')
        sys.stdout = f
        sys.stderr = f
    print(os.getpid())
    save_codes(cfg.cfg.code_dir)

    train_datasets = [key for key in cfg.cfg.train_batch_size]

    trainer = Trainer()
    for train_dataset in train_datasets:
        trainer._make_batch_generator('train', train_dataset)
    trainer._make_model()
    

    loss_eval = {}
    loss_train_eval = {}
    if cfg.cfg.test:
        with torch.no_grad():
            trainer.model.eval()
            for key in cfg.cfg.train_batch_size:
                #if key == "i"
                loss_eval[key] = test_epoch(trainer, 'eval', key)
                if not cfg.cfg.only_testset:
                    loss_train_eval[key] = test_epoch(trainer, 'train_eval', key)
        exit(0)
    # train
    trainer.model.train()
    writer = SummaryWriter(os.path.join(cfg.cfg.output_dir, cfg.cfg.dir_name))
    for epoch in range(trainer.start_epoch, cfg.cfg.end_epoch):

        if cfg.cfg.use_profiler:
            pr = cProfile.Profile()
            pr.enable()
              
        trainer.set_lr(epoch)
        trainer.tot_timer.tic()
        trainer.read_timer.tic()
        if len(train_datasets) == 2:
            batch_generator = zip(cycle(trainer.batch_generator['train_' + train_datasets[0]]), trainer.batch_generator['train_' + train_datasets[1]])
        elif len(train_datasets) == 1:
            batch_generator = trainer.batch_generator['train_' + train_datasets[0]]

        # loss_deephm= test_epoch(trainer, 'eval', 'Deephm')
        # for k in loss_deephm:
        #     writer.add_scalar('Eval/deephm_' + k, loss_deephm[k].item(), epoch)

        # training epoch
        for itr, data in enumerate(batch_generator):
            #break
            if len(train_datasets) == 2:
                inputs, targets, meta_info = collate_batched_multi_datasets(data)
            else:
                inputs, targets, meta_info = data
            trainer.read_timer.toc()
            trainer.gpu_timer.tic()

            # forward
            trainer.optimizer.zero_grad()
            #try:
            out, train_loss, test_loss = trainer.model(inputs, targets, meta_info, 'train')
            # wandb.log({"loss": train_loss})

            # # Optional
            # wandb.watch(trainer.model)

            train_loss = {k:train_loss[k].mean() for k in train_loss}
            test_loss = {k:test_loss[k].mean() for k in test_loss}

            # backward
            sum(train_loss[k] * cfg.cfg.loss_weight_table[k] for k in train_loss).backward()
            if 0 and itr == 0:
                plot_grad_flow(trainer.model.named_parameters(), epoch)
            trainer.optimizer.step()
            trainer.gpu_timer.toc()
            screen = [
                'Epoch %d/%d itr %d/%d:' % (epoch, cfg.cfg.end_epoch, itr, trainer.itr_per_epoch['train_' + train_dataset]),
                'lr: %g' % (trainer.get_lr()),
                'speed: %.2f(%.2fs r%.2f)s/itr' % (
                    trainer.tot_timer.average_time, trainer.gpu_timer.average_time, trainer.read_timer.average_time),
                '%.2fh/epoch' % (trainer.tot_timer.average_time / 3600. * trainer.itr_per_epoch['train_' + train_dataset]),
                ]
            screen += ['%s: %.4f' % ('train_loss_' + k, v.detach() * cfg.cfg.loss_weight_table[k]) for k,v in train_loss.items()]
            screen += ['%s: %.4f' % ('test_loss_' + k, v.detach()) for k,v in test_loss.items()]
            trainer.logger.info(','.join(screen))

            # writer per iteration
            writer.add_scalar('Loss/train_iter', 
                sum(train_loss[k] for k in train_loss).detach().item(), 
                epoch * trainer.itr_per_epoch['train_' + train_dataset] + itr)
            if cfg.cfg.mesh_refine:
                writer.add_scalar('d0', out['dv_list'][0].mean().detach().item(), epoch * trainer.itr_per_epoch['train_' + train_dataset] + itr)
                writer.add_scalar('d1', out['dv_list'][1].mean().detach().item(), epoch * trainer.itr_per_epoch['train_' + train_dataset] + itr)
                writer.add_scalar('d2', out['dv_list'][2].mean().detach().item(), epoch * trainer.itr_per_epoch['train_' + train_dataset] + itr)

            trainer.tot_timer.toc()
            trainer.tot_timer.tic()
            trainer.read_timer.tic()
            # except Exception:
            #     print(targets['img_id'], itr)
        
        if epoch % cfg.cfg.test_freq == 0 or epoch == cfg.cfg.end_epoch - 1:
            # training eval
            for key in cfg.cfg.train_batch_size:
                loss_eval[key] = test_epoch(trainer, 'eval', key)
                loss_train_eval[key] = test_epoch(trainer, 'train_eval', key)

            # writer epoch
            for key in cfg.cfg.train_batch_size:
                for k in loss_eval[key]:
                    writer.add_scalar('Eval/eval_' + key + '_' + k, loss_eval[key][k].item(), epoch)
                    writer.add_scalar('Eval/train_eval_' + key + '_' + k, loss_train_eval[key][k].item(), epoch)

            trainer.save_model(epoch)
        # if not cfg.cfg.mesh_refine:
        #     for k in loss_eval:
        #         writer.add_scalar('Eval/eval_' + k, loss_eval[k].item(), epoch)
        #     for k in loss_train_eval:
        #         writer.add_scalar('Eval/train_eval_' + k, loss_train_eval[k].item(), epoch)
        # for k in loss_deephm_eval:
        #     writer.add_scalar('Eval/deephm_eval_' + k, loss_deephm_eval[k].item(), epoch)
        # for k in loss_deephm_eval:
        #     writer.add_scalar('Eval/deephm_train_eval_' + k, loss_deephm_train_eval[k].item(), epoch)
        # writer.add_image('images', grid, 0)

        # save model
        trainer.save_model('latest')
        # if epoch % cfg.cfg.save_model_freq == cfg.cfg.save_model_freq - 1:
        #     trainer.save_model(epoch)
        
        if cfg.cfg.use_profiler:
            pr.disable()
            pr.dump_stats(os.path.join(cfg.cfg.debug_dir, 'epoch_%02d_profile.prof' % epoch))
            s = io.StringIO()
            sortby = pstats.SortKey.CUMULATIVE
            ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
            ps.print_stats()
            with open(os.path.join(cfg.cfg.debug_dir, 'epoch_%02d_profile.txt' % epoch), 'w') as f:
                print(s.getvalue(), file=f)
        #break

    writer.close()
    if cfg.cfg.redirect:
        f.close()
        

if __name__ == "__main__":
    #os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    main()
