# import wandb
# wandb.init(project="my-test-project", entity="tyluan")


import os
import shutil
from copy import deepcopy
from multiprocessing import Process
import time
import argparse
import itertools

from settings.s1 import experiment_settings as experiment_settings1
from settings.s0 import experiment_settings as experiment_settings0
#from main.config import cfg
from main import config as cfg
from main.train import main
from script.cusel import cusel



def process_cfg(args):
    if hasattr(args, 'itr'):
        for key in args.itr:
            setattr(cfg.cfg, key, getattr(cfg.cfg, key) % args.case[key])

    curtime = time.strftime('%m%d_%H%M%S', time.localtime(time.time()))
    if not cfg.cfg.test:
        cfg.cfg.dir_name = curtime + '_' + cfg.cfg.name
    elif cfg.cfg.old_load_model:
        cfg.cfg.dir_name = cfg.cfg.pretrain_model.split('/')[-2] + '_' + curtime
    else:
        cfg.cfg.dir_name = cfg.cfg.pretrain_model.split('/')[-3] + '_' + curtime
    cfg.cfg.output_dir = os.path.join(cfg.cfg.output_dir, args.set_name)
    
    for key in cfg.cfg.train_batch_size:
        cfg.cfg.total_batch += cfg.cfg.train_batch_size[key]
        cfg.cfg.test_batch_size = cfg.cfg.total_batch * cfg.cfg.test_batch_multiplier
    for key in cfg.cfg.output_dirs:
        setattr(cfg.cfg, key + '_dir', os.path.join(cfg.cfg.output_dir, cfg.cfg.dir_name, key))
    cfg.cfg.redirect = args.redirect


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    #parser.add_argument('--debug', action='store_true', help='vscode pdb mode')
    parser.add_argument('--nmp', dest='multiprocess', action='store_false', help='no multiprocess')
    parser.add_argument('--nrd', dest='redirect', action='store_false', help='no redirect')
    parser.add_argument('--test', dest='redirect', action='store_false', help='no redirect')
    args = parser.parse_args()

    print(os.getpid())
    cfg0 = deepcopy(cfg.cfg)
    if args.test:
        experiment_settings = experiment_settings0
    else:
        experiment_settings = experiment_settings1
    for s_num, setting in enumerate(experiment_settings):
        if 'itr' in setting:
            args.itr = setting['itr']
            itrt = tuple(setting['itr'].values())
            curtime = time.strftime('%m%d_%H%M%S', time.localtime(time.time()))
            args.set_name = curtime + '_' + setting['name']
        else:
            itrt = ([0],)
            args.set_name = ''
        for case in itertools.product(*itrt):
            if hasattr(args, 'itr'):
                args.case = dict(zip(args.itr.keys(), case))
            cfg.cfg = deepcopy(cfg0)
            for key in setting:
                if (not hasattr(cfg.cfg, key)) and (key not in ['itr']):
                    print("Wrong setting! Can't find setting:", key)
                    break
                setattr(cfg.cfg, key, setting[key])
            else:
                gpu_ids = cusel(n=cfg.cfg.num_gpus, m=cfg.cfg.memory_per_gpu)
                os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids
                process_cfg(args)
                
                if args.multiprocess:
                    p = Process(target=main)
                    p.start()
                    time.sleep(1)
                    print(cfg.cfg.name, 'started.', '%d/%d' % (s_num + 1, len(experiment_settings)), 'pid: ', p.pid)
                    time.sleep(29)
                else:
                    main()
