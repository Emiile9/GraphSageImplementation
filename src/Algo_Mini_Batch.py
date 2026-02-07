import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class AlgoMiniBatch(nn.Module):
    def __init__(self,depth,weight_matrices,sigma,agg_f):
        super(AlgoMiniBatch,self).__init__()
        self.K = depth
        self.W = weight_matrices
        self.sigma = sigma
        self.agg_f = agg_f

    def sampling(self,nodes,neighbor,sampling_size):
        B = [set() for i in range(self.K+1)]
        B[self.K] = set(nodes)

        for k in range(self.K,0,-1):
            B[k-1] = B[k].copy()
            for u in B[k]:
                neighbors = neighbor.get(u,[])
                sampling_size = sampling_size[k-1]
                sampled_neighbor = np.random.choice(neighbors,size=sampling_size,replace=True)
                B[k-1] = B[k-1].union(set(sampled_neighbor))
        return B
    
    def forward_propagation(self,nodes,neighbor,feature,sampling_size):
        B = self.sampling(nodes,neighbor,sampling_size)
        h = {}
        h[0] = {node:feature[node] for node in B[0]}

        for k in range(self.K):
            current_h = {}
            for u in B[k+1]:
                neighbors = neighbor.get(u,[])
                sampling_size = sampling_size[k]
                sampled_neighbor = np.random.choice(neighbors,size=sampling_size,replace=True)
                h_k_1 = [h[k][v] for v in sampled_neighbor]
                h_k_N = self.agg_f[k](h_k_1)
                concat = torch.cat((h[k][u],h_k_N),dim=0)
                h_u_k = self.sigma(self.W[k](concat))

                if(F.normalize(h_u_k,p=2,dim=0)!=0):
                    h_u_k = h_u_k/F.normalize(h_u_k,p=2,dim=0)
                current_h[u] = h_u_k
            h[k+1] = current_h
        z = torch.stack([h[self.K][u] for u in nodes])
        return z
            

