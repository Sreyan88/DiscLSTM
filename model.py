import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np, itertools, random, copy, math
from transformers import BertModel, BertConfig
from transformers import AutoTokenizer, AutoModelWithLMHead
from model_utils import *
from discLSTM import DiscLSTM

class Model_DSTM(nn.Module):
    def __init__(self,args,num_classes):
        super(Model_DSTM,self).__init__()
        self.args = args
        self.dropout = nn.Dropout(args.dropout)
        self.emb_to_hidden_dim = nn.Linear(args.emb_dim, args.hidden_dim)
        
        gats = []
        for _ in range(args.gnn_layers):
            gats += [GatLinear(args.hidden_dim)]
        self.gats = nn.ModuleList(gats)
        

        
        layers = [nn.Linear(args.hidden_dim, args.hidden_dim), nn.ReLU()]
        for _ in range(args.mlp_layers - 1):
            layers += [nn.Linear(args.hidden_dim, args.hidden_dim), nn.ReLU()]
        layers += [self.dropout]
        layers += [nn.Linear(args.hidden_dim, num_classes)]
        
        # self.sLayers = []
        # for _ in range(args.gnn_layers):
        #     self.sLayers += [SynLSTM(args.hidden_dim,args.hidden_dim,args.hidden_dim)]
        # self.sLayers = nn.ModuleList(self.sLayers)
        graph_dim = args.gnn_layers*args.hidden_dim
        self.synLSTM = DiscLSTM(args.emb_dim,args.hidden_dim,graph_dim)

        self.out_mlp = nn.Sequential(*layers)

    def forward(self, features, adj,s_mask,s_mask_onehot, lengths):
        
        num_utter = features.size()[1]  # (N)
        text_features = F.relu(self.emb_to_hidden_dim(features)) # (B,N,hidden_dim = 300)
        features_block = [text_features]
        for l in range(self.args.gnn_layers):
            # graph_rep = []
            prev_rep = features_block[l][:,0,:].unsqueeze(1)

            for i in range(1,num_utter):
                # print(features_block[l][:,i,:].shape,prev_rep.shape) 
                _,graph_outputs = self.gats[l](features_block[l][:,i,:],prev_rep,prev_rep,adj[:,i,:i], s_mask[:,i,:i])
                # graph_rep.append(graph_outputs)
                prev_rep = torch.cat([prev_rep,graph_outputs.unsqueeze(1)],dim = 1)
            # graph_rep = torch.stack(graph_rep,dim=1)
            features_block.append(prev_rep)
        graph_features = torch.cat(features_block[:-1],dim = 2)
        synLSTM_outs = self.synLSTM(features,graph_features)
        logits = self.out_mlp(synLSTM_outs)
        return logits
        
