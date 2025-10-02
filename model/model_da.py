import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from model_nodg import *
from einops.layers.torch import Rearrange, Reduce
from torch import autograd

class PAM_Module(nn.Module):
    """ Position attention module"""
    def __init__(self, in_dim):
        super(PAM_Module, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//2, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//2, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width*height).permute(0, 2, 1) 
        proj_key = self.key_conv(x).view(m_batchsize, -1, width*height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width*height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma*out + x
        return out

class Cnn1dEmbedding(nn.Module):
    def __init__(self, ch_num, type = "EEG", drop_rate = 0.5, emb_size = 64):
        super(Cnn1dEmbedding, self).__init__()
        hidden = [32, emb_size//2, emb_size]
        if type == "EOG":
            hidden = [8, emb_size//2, emb_size]

        self.conv = nn.Sequential(
            nn.Conv1d(ch_num, hidden[0], kernel_size=25, stride=2), 
            nn.BatchNorm1d(hidden[0]),
            nn.ELU(),
            nn.MaxPool1d(kernel_size=9, stride=2, padding=4),              
            nn.Dropout(drop_rate),

            nn.Conv1d(hidden[0], hidden[1], kernel_size=11, stride=1), 
            nn.BatchNorm1d(hidden[1]),
            nn.ELU(),
            nn.MaxPool1d(kernel_size=2, stride=2),                                 
            nn.Dropout(drop_rate),

            nn.Conv1d(hidden[1], hidden[2], kernel_size=9, stride=1),  
            nn.BatchNorm1d(hidden[2]),
            nn.ELU(),
            nn.Dropout(drop_rate),

            nn.Conv1d(hidden[2], hidden[2], kernel_size = 109//15, stride = 109//15, groups = hidden[2]),
            nn.BatchNorm1d(hidden[2]),
            nn.ELU(),
            nn.Dropout(drop_rate),
        )
    def forward(self, x):  
        x = self.conv(x)         
        return x

class ReverseLayerF(autograd.Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None


class MulDOCTer_EEG_DA(nn.Module):
    def __init__(self, device, drop_rate=0.5, depth = 4):
        super(MulDOCTer_EEG_DA, self).__init__()
        emb_size = 64
        self.device = device
        # EEG & EOG处理模块
        self.encoderp = Cnn1dEmbedding(12)
        self.encoderc = Cnn1dEmbedding(14)
        self.fusion_EEG = nn.Sequential(
            nn.Conv1d(emb_size*5, emb_size*2, 1, 1), # transpose, conv could enhance fiting ability slightly
            nn.ELU()
        )
        
        self.pam = nn.Sequential(
            Rearrange('b k c t -> b t k c'), # k是脑区+EOG的数量
            PAM_Module(15),  # 通道数是时间
            Rearrange('b t k c -> b (k c) t'),
            )

        self.transformer = nn.Sequential(
            Rearrange('b c t -> b t c'),
            TransformerEncoder(depth = depth, emb_size = emb_size*2)
        )
        
        self.proj = nn.Sequential(
            Rearrange('b t c -> b (t c)'),
            nn.Linear(1920, 128),  # 2s->1860
            nn.ELU(),
            nn.Dropout(drop_rate),
            nn.Linear(emb_size*2, 32),
            nn.ELU(),
            nn.Dropout(drop_rate),
            nn.Linear(32, 2)
        )

    def return_param(self):
        return self.parameters()
        
    
    def forward_feature(self, x):
        xla = self.encoderp(x[:, 0:12, :]) 
        xra = self.encoderp(x[:, 12:24, :]) 
        xlp = self.encoderp(x[:, 24:36, :]) 
        xrp = self.encoderp(x[:, 36:48, :])
        xc = self.encoderc(x[:, 48:62, :])

        tmp = xla.shape[1]
        x = torch.empty(x.shape[0], 5, tmp, xla.shape[2]).to(self.device)
        x[:, 0, :, :], x[:, 1, :, :], x[:, 2, :, :], x[:, 3, :, :], x[:, 4, :, :] = xla, xra, xlp, xrp, xc

        x = self.pam(x)

        x = self.fusion_EEG(x)
        x = self.transformer(x)

        return x
    
    
    def forward(self, x): 
        xla = self.encoderp(x[:, 0:12, :]) 
        xra = self.encoderp(x[:, 12:24, :]) 
        xlp = self.encoderp(x[:, 24:36, :]) 
        xrp = self.encoderp(x[:, 36:48, :])
        xc = self.encoderc(x[:, 48:62, :])

        tmp = xla.shape[1]
        x = torch.empty(x.shape[0], 5, tmp, xla.shape[2]).to(self.device)
        x[:, 0, :, :], x[:, 1, :, :], x[:, 2, :, :], x[:, 3, :, :], x[:, 4, :, :] = xla, xra, xlp, xrp, xc
        x = self.pam(x)
        x = self.fusion_EEG(x) 
        x = self.transformer(x)
        out = self.proj(x)
        out = self.g_net(out)
        return out

class MulDOCTer_atten_DG_all(nn.Module):
    def __init__(self, device, N_pat, drop_rate=0.5, depth = 4):
        super(MulDOCTer_atten_DG_all, self).__init__()
        emb_size = 64
        self.device = device
        self.encoderp = Cnn1dEmbedding(12)
        self.encoderc = Cnn1dEmbedding(14)
        self.encodereog = Cnn1dEmbedding(1, type = "EOG")

        self.fusion_EEGEOG = nn.Sequential(
            nn.Conv1d(emb_size*6, emb_size*2, 1, 1), 
            nn.ELU()
        )
        
        self.pam = nn.Sequential(
            Rearrange('b k c t -> b t k c'),
            PAM_Module(15),  
            Rearrange('b t k c -> b (k c) t'),
            )

        self.transformer = nn.Sequential(
            Rearrange('b c t -> b t c'),
            TransformerEncoder(depth = depth, emb_size = emb_size*2)
        )
        
        self.proj = nn.Sequential(
            Rearrange('b t c -> b (t c)'),
            nn.Linear(1920, 128), 
            nn.ELU(),
            nn.Dropout(drop_rate),
        )


        self.cause_embedding = nn.Embedding(num_embeddings=3, embedding_dim=emb_size)
        self.bc_embedding = nn.Embedding(num_embeddings=12, embedding_dim=emb_size)
        self.bybc_encoder = nn.Sequential(
            nn.Linear(emb_size * 2, emb_size),
            nn.ELU(),
        )
        
      
        self.final_fusion = nn.Sequential(
            nn.Linear(emb_size*3, emb_size*3),
            nn.ELU(),
        )

        self.classifier = nn.Sequential(
            nn.Linear(emb_size*3, 32),
            nn.ELU(),
            nn.Dropout(drop_rate),
            nn.Linear(32, 2)
        )
        
        self.discriminator = nn.Sequential(
            nn.Linear(emb_size*3, emb_size*3), # 128, 128
            nn.ReLU(), 
            nn.Dropout(drop_rate),
            nn.Linear(emb_size*3, N_pat)
        )

    def return_param(self):
        return self.parameters()

    def forward(self, x, alpha=0.0): 
        x, cause_number, bc_number = x
        xla = self.encoderp(x[:, 0:12, :])
        xra = self.encoderp(x[:, 12:24, :]) 
        xlp = self.encoderp(x[:, 24:36, :]) 
        xrp = self.encoderp(x[:, 36:48, :]) 
        xc = self.encoderc(x[:, 48:62, :])
        xe = self.encodereog(rearrange(x[:, -1, :], "b t -> b 1 t"))
        tmp = xe.shape[1]
        x = torch.empty(x.shape[0], 6, tmp, xe.shape[2]).to(self.device)
        x[:, 0, :, :], x[:, 1, :, :], x[:, 2, :, :], x[:, 3, :, :], x[:, 4, :, :], x[:, 5, :, :] = xla, xra, xlp, xrp, xc, xe

        x = self.pam(x)

        x = self.fusion_EEGEOG(x)
        x = self.transformer(x)
        x = self.proj(x)
  
        cause_embed = self.cause_embedding(cause_number)  # (N, 128)
        bc_embed = self.bc_embedding(bc_number)
        bybc_embed = torch.concat((cause_embed, bc_embed), dim = 1)
        bybc_fusion = self.bybc_encoder(bybc_embed)

        fusion_x = torch.cat((x, bybc_fusion), dim=1)
        fusion_x = self.final_fusion(fusion_x) 
        features = fusion_x.clone()

        out = self.classifier(fusion_x)
        
        reverse_feature = ReverseLayerF.apply(fusion_x, alpha)
        d_out = self.discriminator(reverse_feature)
        return features, out, d_out

class MulDOCTer_atten_DG(nn.Module):
    def __init__(self, device, N_pat, drop_rate=0.5, depth = 4):
        super(MulDOCTer_atten_DG, self).__init__()
        emb_size = 64
        self.device = device
        self.encoderp = Cnn1dEmbedding(12)
        self.encoderc = Cnn1dEmbedding(14)
        self.encodereog = Cnn1dEmbedding(1, type = "EOG")

        self.fusion_EEGEOG = nn.Sequential(
            nn.Conv1d(emb_size*6, emb_size*2, 1, 1),
            nn.ELU()
        )
        
        self.pam = nn.Sequential(
            Rearrange('b k c t -> b t k c'),
            PAM_Module(15),
            Rearrange('b t k c -> b (k c) t'),
            )

        self.transformer = nn.Sequential(
            Rearrange('b c t -> b t c'),
            TransformerEncoder(depth = depth, emb_size = emb_size*2)
        )
        
        self.proj = nn.Sequential(
            Rearrange('b t c -> b (t c)'),
            nn.Linear(1920, 128),  
            nn.ELU(),
            nn.Dropout(drop_rate),
        )


        self.cause_embedding = nn.Embedding(num_embeddings=3, embedding_dim=emb_size)
        self.bc_embedding = nn.Embedding(num_embeddings=12, embedding_dim=emb_size)
        self.bybc_encoder = nn.Sequential(
            nn.Linear(emb_size * 2, emb_size),
            nn.ELU(),
        )
        
      
        self.final_fusion = nn.Sequential(
            nn.Linear(emb_size*3, 32),
            nn.ELU(),
            nn.Dropout(drop_rate),
            nn.Linear(32, 2)
        )
        
        self.discriminator = nn.Sequential(
            nn.Linear(128, 128), # 128, 128
            nn.ReLU(), 
            nn.Dropout(drop_rate),
            nn.Linear(128, N_pat)
        )

    def return_param(self):
        return self.parameters()

    def forward(self, x, alpha=0.0):  
        x, cause_number, bc_number = x
        xla = self.encoderp(x[:, 0:12, :]) 
        xra = self.encoderp(x[:, 12:24, :]) 
        xlp = self.encoderp(x[:, 24:36, :]) 
        xrp = self.encoderp(x[:, 36:48, :]) 
        xc = self.encoderc(x[:, 48:62, :])
        xe = self.encodereog(rearrange(x[:, -1, :], "b t -> b 1 t"))
        tmp = xe.shape[1]
        x = torch.empty(x.shape[0], 6, tmp, xe.shape[2]).to(self.device)
        x[:, 0, :, :], x[:, 1, :, :], x[:, 2, :, :], x[:, 3, :, :], x[:, 4, :, :], x[:, 5, :, :] = xla, xra, xlp, xrp, xc, xe
    
        x = self.pam(x)
        x = self.fusion_EEGEOG(x) 
        x = self.transformer(x)
        x = self.proj(x)
        features = x.clone()

        cause_embed = self.cause_embedding(cause_number)  # (N, 128)
        bc_embed = self.bc_embedding(bc_number)
        bybc_embed = torch.concat((cause_embed, bc_embed), dim = 1)
        bybc_fusion = self.bybc_encoder(bybc_embed)

        fusion_x = torch.cat((x, bybc_fusion), dim=1)

        out = self.final_fusion(fusion_x)

        reverse_feature = ReverseLayerF.apply(features, alpha)
        d_out = self.discriminator(reverse_feature)
        return features, out, d_out
    

if __name__ == "__main__":
    DEVICE = torch.device('cpu')
    tlen = 1000
    ch_num = 64
    model = MulDOCTer_EOG_txt(DEVICE).to(DEVICE) 
    batch = torch.rand(4, ch_num, tlen).to(DEVICE)
    cause = torch.tensor(np.array([1, 1, 1, 1])).to(DEVICE)
    bc = torch.tensor(np.array([2, 2, 2, 2])).to(DEVICE)
    input_ = (batch, cause, bc)
    print("bathcshape:", batch.shape)
    print(model(input_).shape)