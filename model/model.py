from einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pickle
from torch import autograd
import math
from torch.nn import Parameter
from einops.layers.torch import Rearrange, Reduce
from torchsummary import summary
from torch.nn.modules.module import _addindent
from torch import Tensor
from model_da import *


"""
FeatureCNN
"""

class FeatureCNN_Doc(nn.Module):
    def __init__(self, input_sample = (1, 1, 62, 250), emb_size=40):
        super(FeatureCNN_Doc, self).__init__()
        chn = input_sample[-2]
        self.shallownet = nn.Sequential(
            nn.Conv2d(1, 40, (1, 25), (1, 1)),
            nn.Conv2d(40, 40, (chn, 1), (1, 1)),
            nn.BatchNorm2d(40),
            nn.ELU(),  # ELU
            nn.AvgPool2d((1, 75), (1, 15)),  
            nn.Dropout(0.5),
        )

        self.projection = nn.Sequential(
            nn.Conv2d(40, emb_size, (1, 1), stride=(1, 1)),  # transpose, conv could enhance fiting ability slightly
            Rearrange('b e (h) (w) -> b e (h w)'),
            nn.Conv1d(emb_size, emb_size, kernel_size = 8, stride = 8, groups = emb_size),
            nn.ELU(),  # ELU
            nn.Dropout(0.2),
            Rearrange('b c t -> b t c')
        )

    def forward(self, x):
        x = self.shallownet(x)
        x = self.projection(x)
        x = x.contiguous().view(x.size(0), -1)

        return x # output (b, 280)

class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size, num_heads, dropout):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.keys = nn.Linear(emb_size, emb_size)
        self.queries = nn.Linear(emb_size, emb_size)
        self.values = nn.Linear(emb_size, emb_size)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        queries = rearrange(self.queries(x), "b n (h d) -> b h n d", h=self.num_heads)
        keys = rearrange(self.keys(x), "b n (h d) -> b h n d", h=self.num_heads)
        values = rearrange(self.values(x), "b n (h d) -> b h n d", h=self.num_heads)
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)  
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)

        scaling = self.emb_size ** (1 / 2)
        att = F.softmax(energy / scaling, dim=-1)
        att = self.att_drop(att)
        out = torch.einsum('bhal, bhlv -> bhav ', att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        return out


class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x

class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size, expansion, drop_p):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )


class GELU(nn.Module):
    def forward(self, input: Tensor) -> Tensor:
        return input*0.5*(1.0+torch.erf(input/math.sqrt(2.0)))


class TransformerEncoderBlock(nn.Sequential):
    def __init__(self,
                 emb_size,
                 num_heads=2,
                 drop_p=0.5,
                 forward_expansion=4,
                 forward_drop_p=0.5):
        super().__init__(
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                MultiHeadAttention(emb_size, num_heads, drop_p),
                nn.Dropout(drop_p)
            )),
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                FeedForwardBlock(
                    emb_size, expansion=forward_expansion, drop_p=forward_drop_p),
                nn.Dropout(drop_p)
            )
            ))


class TransformerEncoder(nn.Sequential):
    def __init__(self, depth, emb_size):
        super().__init__(*[TransformerEncoderBlock(emb_size) for _ in range(depth)])



"""
Core Module
"""
class Base(nn.Module):
    def __init__(self, dataset, voc_size=None, model=None, ehr_adj=None, ddi_adj=None):
        super(MulDOCTer_atten_DG, self).__init__()
        self.dataset = dataset
        self.model = model

    def forward(self, x, alpha=0.1):
        feature, out, d_out = self.feature_cnn(x, alpha)

        return feature, out, d_out


class ReverseLayerF(autograd.Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None


class CORAL(nn.Module):
    def __init__(self):
        super(CORAL, self).__init__()

    def count_BYOL(self, features):
        features = nn.functional.normalize(features, p=2, dim=1)    
        cosine_similarity_matrix = 1 - torch.mm(features, features.t()) 
        cosine_sums = (cosine_similarity_matrix.sum(dim=1))     
        return cosine_sums.mean()

    def metric_cal(self, x):
        mean_x = x.mean(0, keepdim=True)
        cent_x = x - mean_x
        cova_x = torch.mm(cent_x.transpose(0, 1), cent_x) / (x.shape[0] - 1)
        return mean_x[0], cova_x

    def metric_diff(self, means_pairs, covas_pairs):
        mean_diff = (means_pairs[:, 0, :] - means_pairs[:, 1, :]).pow(2).mean(1).mean() # .sum()
        cova_diff = (covas_pairs[:, 0, :, :] - covas_pairs[:, 1, :, :]).pow(2).mean(1).mean(1).mean() # y2
        return mean_diff, cova_diff

    def forward(self, N_pat, features, domains, device, bz = 516, th = 3):

        means = []
        covas = []
        byol = []
        pset = set()
        if features.shape[0] != bz:
            th = 1

        for i in range(N_pat):
            index = torch.where(domains == i)[0]
            features_from_domain = features[index]
            if index.shape[0] > th:
                pset.add(i)
                mean_feature, cova_feature = self.metric_cal(features_from_domain)
                means.append(mean_feature)
                covas.append(cova_feature)
            if index.shape[0] > 1:
                byol.append(self.count_BYOL(features_from_domain))

        if len(means) == 0:
            print("debug:")
            for i in range(N_pat):
                index = torch.where(domains == i)[0]
                if index.shape[0] != 0:
                    print("index:{}, shape:{}".format(i, index.shape))

        if len(byol) == 0:
            byol_loss = torch.tensor(0, dtype=torch.float)
        else:
            byol = torch.stack(byol, dim=0)
            byol_loss = byol.mean()
  
        if len(means) <= 1:
            print("return 0", "=*="*20)
            return torch.tensor(0, dtype=torch.float), torch.tensor(0, dtype=torch.float), byol_loss, pset
        means_features = torch.stack(means, dim=0) 
        covas_features = torch.stack(covas, dim=0)
        domain_num = means_features.shape[0]
        assert domain_num == len(means), "len means error!"
        domain_ids = torch.arange(0, domain_num).to(device)
        domain_pair = torch.combinations(domain_ids)
        select_num = len(domain_pair) 
        random_select_pair = torch.randperm(len(domain_pair))[:select_num]
        domain_pair = domain_pair[random_select_pair]

        means_pairs = means_features[domain_pair]
        covas_pairs = covas_features[domain_pair]
        
        mean_loss, cova_loss = self.metric_diff(means_pairs, covas_pairs)


        if torch.any(torch.isnan(mean_loss)) or torch.any(torch.isnan(cova_loss)):
            print("has nan:", mean_loss.item(), cova_loss.item(), means_features.shape, means_pairs.shape, domain_pair, domain_pair.shape, "domain_num: {} domain_ids:{}".format(domain_num, domain_ids), "!"*20)

        return mean_loss, cova_loss, byol_loss, pset

class Tmp(Base):
    def __init__(self, dataset, N_pat, voc_size=None, model=None, ehr_adj=None, ddi_adj=None):
        super(Tmp, self).__init__(dataset, voc_size, model, ehr_adj, ddi_adj)
        
        self.discriminator = nn.Sequential(
            nn.Linear(280, 128),
            nn.ReLU(), 
            nn.Dropout(0.5),
            nn.Linear(128, N_pat)
        )
        
    def forward(self, x, alpha):
        features = self.feature_cnn(x)
        out = self.g_net(features)
        reverse_feature = ReverseLayerF.apply(features, alpha)
        d_out = self.discriminator(reverse_feature)
        return features, out, d_out
    
    
if __name__ == "__main__":


    DEVICE = torch.device('cpu')
    ch, tlen = 62, 1000
    model = Base("transformer").to(DEVICE) 
    batch = torch.rand(4, 1, ch, tlen).to(DEVICE)
    print("bathcshape:", batch.shape)
    summary(model,(1, ch, tlen),device = "cpu")
    out, x = model(batch)
    print(x.shape, out.shape)