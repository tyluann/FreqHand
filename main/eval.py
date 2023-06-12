# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import argparse
from tqdm import tqdm
import numpy as np
import cv2
#from config import cfg
import config as cfg
import torch
from main.base import Tester
import torch.backends.cudnn as cudnn
import time
import os
import random
from torch.utils.tensorboard import SummaryWriter, summary
from common.utils.vis import *
from common.utils.preprocessing import load_skeleton

def test_epoch(tester, loader_type, dataset, writer, dir_model):
    summary_count = 0
    os.makedirs(os.path.join(cfg.cfg.vis_dir, dir_model), exist_ok=True)
    if loader_type not in tester.batch_generator:
        tester._make_batch_generator(loader_type, dataset)
    loss_mean = {}
    loss_cases_count = {}
    with torch.no_grad():
        with tqdm(enumerate(tester.batch_generator[loader_type + '_' + dataset]), 
            desc='Total iterations %d   ' % (tester.itr_per_epoch[loader_type + '_' + dataset])) as tbar:
            for itr, (inputs, targets, meta_info) in enumerate(tqdm(tester.batch_generator[loader_type + '_' + dataset])):

                # forward
                out, train_loss, test_loss = tester.model(inputs, targets, meta_info)

                for key in test_loss:
                    test_loss[key] = test_loss[key].detach().cpu()
                    total_key = key
                    #total_key = 'test_loss_' + key
                    if total_key not in loss_mean:
                        loss_mean[total_key] = 0
                        loss_cases_count[total_key] = 0
                    loss_mean[total_key] = (loss_mean[total_key] * loss_cases_count[total_key] + \
                        test_loss[key].sum()) / (loss_cases_count[total_key] + test_loss[key].shape[0])
                    loss_cases_count[total_key] += test_loss[key].shape[0]

                tbar.set_postfix(ordered_dict={'mean_' + total_key: str(loss_mean[total_key].detach().item())})
                tbar.update()


                vis = True
                if vis:
                    joints_vis = torch.cat([out['joint_out'].detach().cpu(), out['joint_gt'].detach().cpu()], dim=1)
                    joints_color = torch.cat([joints_color, torch.ones(joints_color.shape) * 255], dim=0)
                    joint_valid = out['joint_valid'].detach().cpu().numpy()
                    #joints_color = torch.unsqueeze(joints_color, dim=0)#.repeat(out['joint_out'].shape[0], 1, 1)
                    img = inputs['img'].detach().cpu()
                    img_id = targets['img_id'].detach().cpu().numpy()
                    
                    for bid in range(out['joint_out'].shape[0]):
                        error = test_loss['joint'][bid].detach().cpu()
                        writer.add_scalar('vis/%s_joint_error' % loader_type, error, summary_count)
                        #writer.add_image('vis/%s_input_img' % loader_type, img[bid], summary_count)
                        #writer.add_mesh('vis/%s_joints' % loader_type, joints_vis[bid:bid+1], colors=joints_color, global_step=summary_count)
                        if random.random() < 0.01: # and error > 90:
                            filename = os.path.join(cfg.cfg.vis_dir, dir_model, "%s_i%04d_c%05d_E%05d" % \
                                (loader_type, img_id[bid], summary_count, int(error * 100 + 0.5)))
                            cv2.imwrite(filename + '.jpg', img[bid].numpy().transpose(1,2,0)[:,:,::-1]*255)
                            draw_joints_mano_42(filename + '.obj', joints_vis[bid].numpy(), joint_valid[bid])
                            #joint_valid = out['joint_valid'].detach().cpu().numpy()[bid]
                            # print(summary_count)
                        summary_count += 1
    return loss_mean


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.cfg.gpu_ids
    dir_model = cfg.cfg.pretrain_model.split('/')[-2] + '_' + cfg.cfg.pretrain_model.split('/')[-1][:-3]
    summary_dir = os.path.join(cfg.cfg.summary_dir, dir_model)
    writer = SummaryWriter(summary_dir)

    cudnn.benchmark = True

    dataset = cfg.cfg.test_dataset #'Deephm' # 'Ih26m' # 

    tester = Tester()
    tester._make_model()
    loss_eval = test_epoch(tester, 'eval', dataset, writer, dir_model)
    loss_train_eval = test_epoch(tester, 'train_eval', dataset, writer, dir_model)

    for k in loss_eval:
        print('Eval/eval_' + k, loss_eval[k].item())
    for k in loss_train_eval:
        print('Eval/train_eval_' + k, loss_train_eval[k].item())

    writer.close()
    
    
        
if __name__ == "__main__":
    main()
