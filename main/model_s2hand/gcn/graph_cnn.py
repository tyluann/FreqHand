"""
This file contains the Definition of GraphCNN
GraphCNN includes ResNet50 as a submodule
"""
from __future__ import division

import torch
import torch.nn as nn

#from main.model_s2hand.GCN.graph_layers import GraphConvolution

from .graph_layers import GraphResBlock, GraphLinear, GraphConvolution
#from .resnet import resnet50
#from main.config import cfg
from main import config as cfg

class GraphCNN(nn.Module):
    
    def __init__(self, A, C_in_gcn, C_out_gcn):
        super(GraphCNN, self).__init__()
        #self.resnet = resnet50(pretrained=True)
        layers = [] # [GraphLinear(3 + 2048, 2 * num_channels)]
        for i in range(cfg.cfg.gcn_num_layers):
            layers.append(GraphResBlock(C_in_gcn, C_in_gcn, A))
        layers.append(GraphResBlock(C_in_gcn, C_out_gcn, A))

        # self.shape = nn.Sequential(GraphResBlock(num_channels, 64, A),
        #                            GraphResBlock(64, 32, A),
        #                            nn.GroupNorm(32 // 8, 32),
        #                            nn.ReLU(inplace=True),
        #                            GraphLinear(32, 3))
        self.gc = nn.Sequential(*layers)
        # self.camera_fc = nn.Sequential(nn.GroupNorm(num_channels // 8, num_channels),
        #                               nn.ReLU(inplace=True),
        #                               GraphLinear(num_channels, 1),
        #                               nn.ReLU(inplace=True),
        #                               nn.Linear(A.shape[0], 3))
        self.gc_out = GraphConvolution(C_out_gcn, 3, A)

    def forward(self, feature_input, x, vertices):
        """Forward pass
        Inputs:
            feature_input: size = (B, V, C_in)
            x: size = (B, V, C_in_gcn)
            vertices: size = (B, V, 3)
        Returns:
            feature_gcn: size = (B, V, C_out_gcn)
        """
        if x is not None:
            x = torch.cat([x, feature_input, vertices], dim=1)
        else:
            x = torch.cat([feature_input, vertices], dim=1)
        x = self.gc(x)
        dv = self.gc_out(x.transpose(1,2)).transpose(1,2)
        vertices = dv + vertices

        return x, vertices, dv