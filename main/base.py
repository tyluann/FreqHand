# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
import os.path as osp
import math
from pickletools import optimize
import time
import glob
import abc
from torch.utils.data import DataLoader
import torch.optim
import torchvision.transforms as transforms
from pytorch3d.datasets import collate_batched_meshes_tyluan
# from torch.utils.tensorboard import SummaryWriter

#from main.config import cfg
from main import config as cfg
#from data.dataset import Dataset
from data.datasets import Deephm, Ih26m
from common.timer import Timer
from common.logger import colorlogger
from torch.nn.parallel.data_parallel import DataParallel
# from main.model import get_model
from main.model import Model, load_model

import torch
import re
import collections
from torch._six import string_classes

np_str_obj_array_pattern = re.compile(r'[SaUO]')

from typing import Dict, List

class Base(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, log_name='logs.txt'):
        
        self.cur_epoch = 0

        # timer
        self.tot_timer = Timer()
        self.gpu_timer = Timer()
        self.read_timer = Timer()

        # logger
        self.logger = colorlogger(log_name=log_name)

    @abc.abstractmethod
    def _make_batch_generator(self):
        return

    @abc.abstractmethod
    def _make_model(self):
        return


class Trainer(Base):
    
    def __init__(self):
        super(Trainer, self).__init__(log_name = 'train_logs.txt')
        self.batch_generator = {}
        self.itr_per_epoch = {}

    def get_optimizer(self, model):
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.cfg.lr)
        return optimizer

    def set_lr(self, epoch):
        #if len(cfg.cfg.lr_dec_epoch) == 0:
        if  cfg.cfg.lr_dec_epoch is None:
            return cfg.cfg.lr
        for epoch_decay in cfg.cfg.lr_dec_epoch:
            if epoch == epoch_decay:
                for i in range(len(self.optimizer.param_groups)):
                    # if i != 0:
                    self.optimizer.param_groups[i]['lr'] = self.optimizer.param_groups[i]['lr'] / cfg.cfg.lr_dec_factor
                    # if i == 0:
                    #     self.optimizer.param_groups[i]['lr'] = self.optimizer.param_groups[i]['lr'] / cfg.cfg.lr_dec_factor_bb
        # elif epoch == cfg.cfg.lr_dec_epoch[1]:
        #     for i in range(len(self.optimizer.param_groups)):
        #         # if i != 0:
        #         self.optimizer.param_groups[i]['lr'] = self.optimizer.param_groups[i]['lr'] / cfg.cfg.lr_dec_factor 
        #         # if i == 0:
        #         #     self.optimizer.param_groups[i]['lr'] = self.optimizer.param_groups[i]['lr'] / cfg.cfg.lr_dec_factor_bb / cfg.cfg.lr_dec_factor_bb
        return self.optimizer.param_groups[0]['lr']

        # for e in cfg.cfg.lr_dec_epoch:
        #     if epoch < e:
        #         break
        # if epoch < cfg.cfg.lr_dec_epoch[-1]:
        #     idx = cfg.cfg.lr_dec_epoch.index(e)
        #     for g in self.optimizer.param_groups:
        #         g['lr'] = cfg.cfg.lr / (cfg.cfg.lr_dec_factor ** idx)
        # else:
        #     for g in self.optimizer.param_groups:
        #         g['lr'] = cfg.cfg.lr / (cfg.cfg.lr_dec_factor ** len(cfg.cfg.lr_dec_epoch))

    def get_lr(self):
        for g in self.optimizer.param_groups:
            cur_lr = g['lr']

        return cur_lr
    def _make_batch_generator(self, loader_type, dataset_name): # ['train', 'train_eval', 'eval']
        # data load and construct batch generator
        self.logger.info("Creating dataset...")
        split = 'train' if (loader_type == 'train' or loader_type == 'train_eval') else 'test'
        shuffle = True if loader_type == 'train' else False
        #train_batch_size = sum(cfg.cfg.train_batch_size[k] for k in cfg.cfg.train_batch_size)
        batch_size = cfg.cfg.train_batch_size[dataset_name] if loader_type == 'train' else cfg.cfg.test_batch_size
        batch_size *= cfg.cfg.num_gpus
        num_thread = cfg.cfg.num_thread if loader_type == 'train' else cfg.cfg.test_num_thread
        dataset = eval(dataset_name)(transforms.ToTensor(), split)
        batch_generator = DataLoader(dataset=dataset, batch_size=batch_size,
            shuffle=shuffle, num_workers=num_thread, pin_memory=True, 
            collate_fn=collate_batched_meshes_tyluan)

        # self.mesh = dataset.mesh
        # self.root_joint_idx = dataset.root_joint_idx
        # self.align_joint_idx = dataset.align_joint_idx
        # self.non_rigid_joint_idx = dataset.non_rigid_joint_idx
        self.itr_per_epoch[loader_type + '_' + dataset_name] = math.ceil(dataset.__len__() / batch_size)
        self.batch_generator[loader_type + '_' + dataset_name] = batch_generator

    def _make_model(self):
        # prepare network
        self.logger.info("Creating graph and optimizer...")
        # if not cfg.cfg.use_s2hand:
        #     model = get_model('train')
        #     model = DataParallel(model).cuda()
        #     optimizer = self.get_optimizer(model)
        #     if cfg.cfg.continue_train:
        #         start_epoch, model, optimizer = self.load_model(model, optimizer)
        #     else:
        #         start_epoch = 0
        # if cfg.cfg.use_s2hand:
        model = Model()
        #if cfg.cfg.continue_train:
        if cfg.cfg.pretrain_model is not None:
            if cfg.cfg.old_load_model:
                model, _ = load_model(model)
            else:
                model = self.load_model(model)
            
        model = DataParallel(model).cuda()
        #optimizer = torch.optim.Adam(model.parameters(), lr=cfg.cfg.lr)
        opti_list = [{'params':model.module.encoder.parameters(), 'lr':cfg.cfg.lr_backbone},
                            {'params':model.module.hand_decoder.parameters(), 'lr':cfg.cfg.lr},
                            {'params':model.module.rgb2hm.parameters(), 'lr':0},
                        ]
        if cfg.cfg.mesh_refine:
            opti_list.append({'params':model.module.model_refine.parameters(), 'lr':cfg.cfg.lr_gcn})
        optimizer = torch.optim.Adam(
        #optimizer = torch.optim.AdamW(
            opti_list, weight_decay=cfg.cfg.weight_decay)
        start_epoch = 0
        
        self.start_epoch = start_epoch
        self.model = model
        self.optimizer = optimizer




    # if cfg.cfg.use_s2hand:
    #     def save_model(self, dir_time, epoch):
    #         model = self.model
    #         optimizer = self.optimizer
    #         state = {
    #             'optimizer': optimizer.state_dict(),
    #             'epoch': epoch,
    #         }
    #         save_file = osp.join(cfg.cfg.model_dump_dir, dir_time, 'texturehand_{}.t7'.format(str(epoch)))
    #         os.makedirs(osp.join(cfg.cfg.model_dump_dir, dir_time), exist_ok=True)
    #         if hasattr(model.module,'encoder'):
    #             state['encoder'] = model.module.encoder.state_dict()
    #         if hasattr(model.module,'hand_decoder'):
    #             state['decoder'] = model.module.hand_decoder.state_dict()
    #         if hasattr(model.module,'heatmap_attention'):
    #             state['heatmap_attention'] = model.module.heatmap_attention.state_dict()
    #         if hasattr(model.module,'rgb2hm'):
    #             state['rgb2hm'] = model.module.rgb2hm.state_dict()
    #         if hasattr(model.module,'hm2hand'):
    #             state['hm2hand'] = model.module.hm2hand.state_dict()
    #         if hasattr(model.module,'mesh2pose'):
    #             state['mesh2pose'] = model.module.mesh2pose.state_dict()

    #         if hasattr(model.module,'percep_encoder'):
    #             state['percep_encoder'] = model.module.percep_encoder.state_dict()
            
    #         if hasattr(model.module,'texture_light_from_low'):
    #             state['texture_light_from_low'] = model.module.texture_light_from_low.state_dict()
    #         if hasattr(model.module,'model_refine'):
    #             state['model_refine'] = model.module.model_refine.state_dict()
    #         print("Save model at:", save_file)
    #         torch.save(state, save_file)
    # else:

    def save_model(self, epoch):
        # os.makedirs(osp.join(cfg.cfg.model_dump_dir, dir_time), exist_ok=True)
        save_file_model = osp.join(cfg.cfg.model_dump_dir, 'freqhand_{}.t7'.format(str(epoch)))
        state = self.model.module.state_dict()
        torch.save(state, save_file_model)
        self.logger.info("Write checkpoint into {}".format(save_file_model))

        # save_file_optimizer = osp.join(cfg.cfg.model_dump_dir, 'optimizer_{}.t7'.format(str(epoch)))
        # state = self.optimizer.state_dict()
        # torch.save(state, save_file_optimizer)
        # self.logger.info("Write checkpoint into {}".format(save_file_optimizer))

    def load_model(self, model):
        model_file = cfg.cfg.pretrain_model
        # filename_split = model_file.split('/')
        #optimizer_file = osp.join('/'.join(filename_split[:-1]), 'optimizer' + filename_split[-1][11:])
        # epoch = int(filename_split[-1][12:-3])
        # self.start_epoch = epoch + 1
        
        model_dict = torch.load(model_file)
        model.load_state_dict(model_dict)
        self.logger.info('Load checkpoint from {}'.format(model_file))

        # if os.path.exists(optimizer_file):
        #     optimizer_dict = torch.load(optimizer_file)
        #     self.optimizer.load_state_dict(optimizer_dict, strict=False)
        #     self.logger.info('Load checkpoint from {}'.format(optimizer_file))

        return model

    # def load_model_backup(self):
    #     self.model, _ = load_model(self.model)


class Tester(Base):
    def __init__(self):
        super(Tester, self).__init__(log_name = 'test_logs.txt')
        self.batch_generator = {}
        self.itr_per_epoch = {}

    def _make_batch_generator(self, loader_type, dataset_name): #['train_eval', 'eval', 'test']
        # data load and construct batch generator
        self.logger.info("Creating dataset...")
        split = loader_type #'train' if loader_type == 'train_eval' else 'test'
        shuffle = False
        #train_batch_size = sum(cfg.cfg.train_batch_size[k] for k in cfg.cfg.train_batch_size)
        batch_size = cfg.cfg.num_gpus * cfg.cfg.test_batch_size
        dataset = eval(dataset_name)(transforms.ToTensor(), split)

        batch_generator = DataLoader(dataset=dataset, batch_size=batch_size,
            shuffle=shuffle, num_workers=cfg.cfg.num_thread, pin_memory=True, 
            collate_fn=collate_batched_meshes_tyluan)

        # self.mesh = dataset.mesh
        # self.root_joint_idx = dataset.root_joint_idx
        # self.align_joint_idx = dataset.align_joint_idx
        # self.non_rigid_joint_idx = dataset.non_rigid_joint_idx
        self.itr_per_epoch[loader_type + '_' + dataset_name] = math.ceil(dataset.__len__() / batch_size)
        self.batch_generator[loader_type + '_' + dataset_name] = batch_generator
    
    def _make_model(self):
        # if not cfg.cfg.use_s2hand:
        #     #model_path = os.path.join(cfg.cfg.model_dump_dir, 'snapshot_%d.pth.tar' % self.test_epoch)
        #     assert os.path.exists(cfg.cfg.pretrain_model), 'Cannot find model at ' + cfg.cfg.pretrain_model
        #     self.logger.info('Load checkpoint from {}'.format(cfg.cfg.pretrain_model))
            
        #     # prepare network
        #     self.logger.info("Creating graph...")
        #     model = get_model('test')
        #     model = DataParallel(model).cuda()
        #     ckpt = torch.load(cfg.cfg.pretrain_model)
        #     model.load_state_dict(ckpt['network'], strict=False)
        # if cfg.cfg.use_s2hand:
        model = Model()
        if cfg.cfg.pretrain_model is not None:
            self.load_model()
        model = DataParallel(model).cuda()

        model.eval()

        self.model = model
        
    def load_model(self):
        model_file = cfg.cfg.pretrain_model
        filename_split = model_file.split('/')
        # epoch = int(filename_split[-1][12:-3])
        # self.start_epoch = epoch + 1
        
        model_dict = torch.load(model_file)
        self.model.load_state_dict(model_dict)
        self.logger.info('Load checkpoint from {}'.format(model_file))

    def _evaluate(self, preds, result_save_path):
        self.testset.evaluate(preds, result_save_path)
    
    


# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the



default_collate_err_msg_format = (
    "collate_batched_multi_datasets: batch must contain tensors, numpy arrays, numbers, "
    "dicts or lists or Meshes; found {}")

def collate_batched_multi_datasets(batch):  # pragma: no cover
    r"""Puts each data field into a tensor with outer dimension batch size"""

    elem = batch[0]
    elem_type = type(elem)
    if isinstance(elem, torch.Tensor):
        out = None
        if torch.utils.data.get_worker_info() is not None:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum(x.numel() for x in batch)
            storage = elem.storage()._new_shared(numel)
            out = elem.new(storage)
        return torch.cat(batch, dim=0, out=out)
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        if elem_type.__name__ == 'ndarray' or elem_type.__name__ == 'memmap':
            # array of string classes and object
            if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                raise TypeError(default_collate_err_msg_format.format(elem.dtype))

            return collate_batched_multi_datasets([torch.as_tensor(b) for b in batch])
        elif elem.shape == ():  # scalars
            return torch.as_tensor(batch)
    elif isinstance(elem, float):
        return torch.tensor(batch, dtype=torch.float64)
    elif isinstance(elem, int):
        return torch.tensor(batch)
    elif isinstance(elem, string_classes):
        return batch
    # elif isinstance(elem, Meshes):
    #     verts = []; faces = []
    #     for mesh in batch:
    #         verts = verts + elem.verts_list()
    #         faces = faces + elem.faces_list()
    #     return Meshes(verts=verts, faces=faces)
    elif isinstance(elem, collections.abc.Mapping):
        return {key: collate_batched_multi_datasets([d[key] for d in batch]) for key in elem}
    elif isinstance(elem, tuple) and hasattr(elem, '_fields'):  # namedtuple
        return elem_type(*(collate_batched_multi_datasets(samples) for samples in zip(*batch)))
    elif isinstance(elem, collections.abc.Sequence):
        # check to make sure that the elements in batch have consistent size
        it = iter(batch)
        elem_size = len(next(it))
        if not all(len(elem) == elem_size for elem in it):
            raise RuntimeError('each element in list of batch should be of equal size')
        transposed = zip(*batch)
        return [collate_batched_multi_datasets(samples) for samples in transposed]
        

    raise TypeError(default_collate_err_msg_format.format(elem_type))


