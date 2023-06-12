# High Fidelity 3D Hand Shape Reconstruction via Scalable Graph Frequency Decomposition

## Introduction
This repo is official implementation of **[ High Fidelity 3D Hand Shape Reconstruction via Scalable Graph Frequency Decomposition(CVPR 2023)](https://openaccess.thecvf.com/content/CVPR2023/papers/Luan_High_Fidelity_3D_Hand_Shape_Reconstruction_via_Scalable_Graph_Frequency_CVPR_2023_paper.pdf)**. 

## Install environment
```bash
conda env create -f environment.yml
```

## Dataset
We use subdivided MANO to register 3D hand Mesh from "[DeepHandMesh](https://github.com/facebookresearch/DeepHandMesh)". Download joint annotation and our processed dataset from [here](https://buffalo.box.com/s/2iuozn69nfdlaq9znjsmnicrp5fdfi7h/data).
After downloading, put the contents in `$root/data` directory.

Download images for [DeephandMesh](https://mks0601.github.io/DeepHandMesh/) and following the unzip instructions. Then, put the `images` folder under `$root/data/DeepHandMesh` directory

## Assets
Download subdivided MANO from [here](https://buffalo.box.com/s/2iuozn69nfdlaq9znjsmnicrp5fdfi7h/assets) and put the contents under `$root/assets/`.
Download MANO from [here](https://mano.is.tue.mpg.de/). Put `models` folder under `$root/assets/mano`.

## Pretrained Model
Download pretrained model from [here](https://buffalo.box.com/s/2iuozn69nfdlaq9znjsmnicrp5fdfi7h/model) and put the contents under `$root/model`.

## Testing
```bash
python main.py --test --nmp --nrd
```

## Training
```bash
python main.py --nmp --nrd
```

## Reference  
```  
@InProceedings{Luan_2023_CVPR,
    author    = {Luan, Tianyu and Zhai, Yuanhao and Meng, Jingjing and Li, Zhong and Chen, Zhang and Xu, Yi and Yuan, Junsong},
    title     = {High Fidelity 3D Hand Shape Reconstruction via Scalable Graph Frequency Decomposition},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2023},
    pages     = {16795-16804}
}
```

## Acknowledgements
This repo inherited code from [DeepHandMesh](https://github.com/facebookresearch/DeepHandMesh) and [S2HAND](https://github.com/TerenceCYJ/S2HAND).

