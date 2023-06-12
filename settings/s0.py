experiment_settings = [
    {
        "name": 'test',
        "test": True,
        "pretrain_model": "model/freqhand.t7",
        "num_gpus": 1,
        "train_batch_size": {'Deephm': 28},
        "test_batch_multiplier": 10,
        "memory_per_gpu": 23500,
        "gcn_num_layers": 10,
        "lr_gcn": 5e-4,
        "lr_backbone": 1e-5,
        "loss_mpfe_type": 'div_predxgt',
        "loss_weight_table": {
            'joint': 10,
            'joint_0': 1, 'joint_1': 1, 'joint_2': 1, 
            'laplacian_0':5000, 'laplacian_1': 5000, 'laplacian_2': 5000,
            'mpve_0': 1, 'mpve_1': 1, 'mpve_2': 1, 
            'mpfe_0': 60, 'mpfe_1': 60, 'mpfe_2': 100, 
        },
        "save_model_freq": 1,
        "test_freq": 1,
        "only_testset": True,
        'sota': True,
        'vis_all': True,
    },

]