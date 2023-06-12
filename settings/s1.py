#/home/tyluan/workspace/code/DeepHandMesh/output_new/0313_055730_loss_predxgt159/model_dump/texturehand_24.t7

experiment_settings = [

    {
        "name": 'train',
        "old_load_model": True,
        "num_gpus": 1,
        "train_batch_size": {'Deephm': 28},
        "test_batch_multiplier": 10,
        "memory_per_gpu": 23500,
        "gcn_num_layers": 10,
        "lr_gcn": 5e-4,
        "lr_backbone": 1e-4,
        "loss_mpfe_type": 'div_predxgt',
    },

]