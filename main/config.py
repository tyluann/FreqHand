# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from contextlib import redirect_stderr
import os
import os.path as osp
import sys
import math
import numpy as np

class Config:
    ## input, output
    input_img_shape = (224,224)
    rendered_img_shape = (256,256)
    depth_min = 1
    depth_max = 99999
    render_view_num = 6
    backbone_img_feat_dim = 512
    id_code_dim = 32
    bone_step_size = 0.1
    crop_projection = False

    
    ## model
    resnet_type = 50 # 18, 34, 50, 101, 152
    gcn_num_layers = 5
    use_mean_shape = True
    use_s2hand = True
    mesh_refine = True # False # 
    use_pretrained_backbone = True

    #pretrain_model = None # EfficientNet+Sratch
    #pretrain_model = 'output/model_dump/1020_0552/texturehand_29.t7' # continue training
    pretrain_model = 'output/model_dump/1222_2335/texturehand_38.t7' # baseline
    #pretrain_model = 'output/model_dump/0119_2305/texturehand_79.t7' # test
    #pretrain_model = 'output/model_dump/texturehand_freihand.t7' # S2HAND

    old_load_model = False
    feature_sample = 'index'
    

    ## training config
    #train_batch_size = {'Deephm': 6, 'Ih26m': 14} # per gpu
    train_batch_size = {'Deephm': 14} # per gpu

    weight_decay = 0
    lr_dec_epoch = [300,301,302]
    end_epoch = 400
    lr = 1e-5
    lr_backbone = 1e-5
    lr_gcn = 1e-4
    lr_dec_factor = 1
    lr_dec_factor_bb = 1
    lr_register_1 = 1e-3
    lr_register_2 = 5e-4
    loss_penet_r_weight = 1 
    loss_penet_nr_weight = 5
    loss_lap_weight = 5
    loss_scale_weight = 5e4
    save_model_freq = 40
    test_freq = 5
    continue_train = True
    pre_load_data = True

    # loss
    loss_weight_table = {
        'joint': 1,
        'joint_0': 1, 'joint_1': 1, 'joint_2': 1, 
        #'vertices_0': 1, 'vertices_1': 1, 'vertices_2': 1,
        'laplacian_0':5000, 'laplacian_1': 5000, 'laplacian_2': 5000,
        # 'tangent_0': 1, 'tangent_1': 1, 'tangent_2': 1, 
        # 'regular_0': 1, 'regular_1': 1, 'regular_2': 1, 
        # 'cd_0': 1, 'cd_1': 1, 'cd_2': 1, 
        'mpve_0': 1, 'mpve_1': 1, 'mpve_2': 1, 
        'mpfe_0': 60, 'mpfe_1': 60, 'mpfe_2': 100, 
        # 'sffd_0': 1, 'sffd_1': 0.1, 'sffd_2': 0.1,
    }

    loss_mpfe_type = "div_predxgt"
    loss_mpve = True

    ## testing config
    test = False # True #
    #test_batch_size = 1 # all gpu
    test_dataset = 'Deephm' # 'Ih26m' # 
    test_batch_multiplier = 8
    test_batch_size = {}
    total_batch = 0
    post_smooth = True
    vis_all = False
    only_testset = False
    sota = False
    

    ## directory
    name = None
    cur_dir = osp.dirname(os.path.abspath(__file__))
    root_dir = osp.join(cur_dir, '..')
    data_dir = osp.join(root_dir, 'data')
    assets_dir = osp.join(root_dir, 'assets')
    mano_dir = osp.join(assets_dir, 'mano')

    output_dir = osp.join(root_dir, 'output_new')
    output_dirs = ['model_dump', 'vis', 'code', 'result', 'grad', 'debug'] # 'summary', 'log'
    # model_dir = None #osp.join(output_dir, 'model_dump')
    # vis_dir = None #osp.join(output_dir, 'vis')
    # log_dir = None #osp.join(output_dir, 'log')
    # result_dir = None #osp.join(output_dir, 'result')
    # summary_dir = None #osp.join(output_dir, 'summary')
    # grad_dir = None #osp.join(output_dir, 'grad')
    use_shm = False
    set_run = False

    #output
    redirect = True
    vis_multithread = True

    ## resource
    num_thread = 18
    test_num_thread = 2
    #gpu_ids = '2,3'
    num_gpus = 2
    memory_per_gpu = 11000
    
    ## dataset
    subject = '4' #Deephm
    ih26m_subset_rate = 1

    ## functional
    mesh_register = False

    ## debug
    use_profiler = False

cfg = Config()

sys.path.insert(0, osp.join(cfg.root_dir))
from common.utils.dir import add_pypath, make_folder
add_pypath(osp.join(cfg.data_dir))
# make_folder(cfg.model_dir)
# make_folder(cfg.vis_dir)
# make_folder(cfg.log_dir)
# make_folder(cfg.result_dir)
