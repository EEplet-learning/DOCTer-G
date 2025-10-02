import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch import autograd
from collections import OrderedDict
import sys

from tqdm import tqdm
sys.path.append(".")
from model import *
from model import SoftCEL2, BYOL, entropy_for_CondAdv, ProxyPLoss
from torch.nn import init
from model_nodg import *
from model_da import *
SoftCEL = nn.CrossEntropyLoss()


def weight_init(m):
    if isinstance(m, nn.Conv2d):
        init.kaiming_uniform_(m.weight.data)
        init.constant_(m.bias.data,0.1)
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        m.weight.data.normal_(0,0.01)
        m.bias.data.zero_()

"""
EEG for DOC
"""
class DocBase(nn.Module):
    def __init__(self, dataset, lr=1e-3, weight_decay=1e-5):
        super(DocBase, self).__init__()
        self.model = Base(dataset)
        self.model.apply(weight_init)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.dataset = dataset

    def train(self, train_loader, device):
        self.model.train()
        loss_collection = []
        count = 0
        for (X, y) in tqdm(train_loader):
            convScore, _ = self.model(X.to(device))
            y = y.to(device)
            count += convScore.shape[0]
            loss = nn.CrossEntropyLoss()(convScore, y)
            loss_collection.append(loss.item())
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        print ('train avg loss: {:.4}, count: {}, len of dataloader:{}'.format(sum(loss_collection) / len(loss_collection), len(loss_collection), len(train_loader)))
            
    def test(self, test_loader, device):
        self.model.eval()
        with torch.no_grad():
            result = np.array([])
            gt = np.array([])
            for X, y in test_loader:
                convScore, _ = self.model(X.to(device))
                result = np.append(result, torch.max(convScore, 1)[1].cpu().numpy())
                gt = np.append(gt, y.numpy())
        return result, gt


class DocTmp(nn.Module):
    def __init__(self, dataset, N_pat, device, model_name, p1 = 1e-1, p2 = 1., p3 = 1e-1, lr=5e-4, weight_decay=1e-5, bz=512, th=4, clip_value = 0, exptype = "DA0"):
        super(DocTmp, self).__init__()
        self.N_pat = N_pat
        self.bz = bz
        self.model_name = model_name
        if "Mul" in model_name:
            if "1" in model_name:
                self.model = MulDOCTer_atten_DG_1(device, N_pat)
                print("weight")
            elif "2" in model_name:
                print("attention")
                self.model = MulDOCTer_atten_DG_2(device, N_pat)
            else:
                 self.model = MulDOCTer_atten_DG(device, N_pat)
            self.model.apply(weight_init)
        else:
            self.model = Tmp(dataset, N_pat) 
            self.model.apply(weight_init)
        if "DA" in exptype:
            pass

        self.dataset = dataset
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.coral_loss = CORAL().to(device)
        self.device = device
        self.p1 = p1  # adv loss
        self.p2 = p2  # byol loss
        self.p3 = p3  # align loss
        self.th = th
        self.clip_value = clip_value
        

    # ablation 3
    def train_org(self, train_loader, device, epoch, N_epoch):
        self.model.train()
        for i, (X, label, identities) in tqdm(enumerate(train_loader), total=len(train_loader)):
            X = X.to(device)
            label = label.to(device)
            identities = identities.to(device)

            alpha = 0.0
            _, out, _ = self.model(X, alpha)
            loss = nn.CrossEntropyLoss()(out, label)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        print ('train_org: class loss: {} ---'.format(loss.item()))
        return loss
    
    # ablation 0 
    def train(self, train_loader, device, epoch, N_epoch):
        self.model.train()
        loss_collection = [[], [], [], [], []]
        all_set = set()
        for i, (X, label, identities) in tqdm(enumerate(train_loader), total=len(train_loader)):
            if "Mul" in self.model_name:
                x, cause_number, bc_number = X
                x = x.to(device, non_blocking=True)
                cause_number = cause_number.to(device, non_blocking=True)
                bc_number = bc_number.to(device, non_blocking=True)
                X = (x, cause_number, bc_number)
            else:
                X = X.to(device)
            label = label.to(device)
            identities = identities.to(device)
            
            p = float(i + epoch * len(train_loader)) / N_epoch / len(train_loader)
            alpha = 2. / (1. + np.exp(-10 * p)) - 1

            features, out, d_out = self.model(X, alpha)

            loss = nn.CrossEntropyLoss()(out, label)
            loss2 = nn.CrossEntropyLoss()(d_out, identities)
            mean_loss, cova_loss, byol_loss, _pset = self.coral_loss(self.N_pat, features, identities, self.device, self.bz, self.th)
            all_set = all_set.union(_pset)
            loss_collection[0].append(loss.item())
            loss_collection[1].append(loss2.item())
            loss_collection[2].append(byol_loss.item())
            loss_collection[3].append(mean_loss.item())
            loss_collection[4].append(cova_loss.item())
            self.optimizer.zero_grad()
            (loss + self.p1 * loss2 + self.p2 * byol_loss + self.p3 * (mean_loss + cova_loss)).backward() # no 1e-1
            if self.clip_value > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_value)
            self.optimizer.step()

        print('train_all: class loss: {}, loss2: {}, byol_loss: {}, mean_loss: {}, cova_loss: {} --- count:{}| setlen:{} per:{}'.format(
            sum(loss_collection[0])/len(train_loader),
            self.p1 * sum(loss_collection[1])/len(train_loader),
            self.p2 * sum(loss_collection[2])/len(train_loader),
            self.p3 * sum(loss_collection[3])/len(train_loader),
            self.p3 * sum(loss_collection[4])/len(train_loader),
            len(train_loader),
            len(all_set),
            len(all_set)/self.N_pat
            ))
        return sum(loss_collection[0])/len(train_loader), sum(loss_collection[1])/len(train_loader), sum(loss_collection[2])/len(train_loader), sum(loss_collection[3])/len(train_loader), sum(loss_collection[4])/len(train_loader), len(all_set)/self.N_pat
      
    

    # ablation 2
    def train_nop(self, train_loader, device, epoch, N_epoch):
        self.model.train()
        for i, (X, label, identities) in tqdm(enumerate(train_loader), total=len(train_loader)):
            if "Mul" in self.model_name:
                x, cause_number, bc_number = X
                x = x.to(device, non_blocking=True)
                cause_number = cause_number.to(device, non_blocking=True)
                bc_number = bc_number.to(device, non_blocking=True)
                X = (x, cause_number, bc_number)
            else:
                X = X.to(device)
            label = label.to(device)
            identities = identities.to(device)
            
            p = float(i + epoch * len(train_loader)) / N_epoch / len(train_loader)
            alpha = 2. / (1. + np.exp(-10 * p)) - 1
            # Forward pass
            _, out, d_out = self.model(X, alpha)
            loss = nn.CrossEntropyLoss()(out, label)
            loss2 = nn.CrossEntropyLoss()(d_out, identities)
            
            self.optimizer.zero_grad()
            (loss + self.p1 * loss2 ).backward()

            self.optimizer.step()
        print ('train_nop: class loss: {}, loss2: {}---'.format(loss.item(), loss2.item()))
        return loss, loss2

    # ablation 1
    def train_byol(self, train_loader, device, epoch, N_epoch):
        self.model.train()
        for i, (X, label, identities) in tqdm(enumerate(train_loader), total=len(train_loader)):
            if "Mul" in self.model_name:
                x, cause_number, bc_number = X
                x = x.to(device, non_blocking=True)
                cause_number = cause_number.to(device, non_blocking=True)
                bc_number = bc_number.to(device, non_blocking=True)
                X = (x, cause_number, bc_number)
            else:
                X = X.to(device)
            label = label.to(device)
            identities = identities.to(device)
            
            p = float(i + epoch * len(train_loader)) / N_epoch / len(train_loader)
            alpha = 2. / (1. + np.exp(-10 * p)) - 1
            features, out, d_out = self.model(X, alpha)
            loss = nn.CrossEntropyLoss()(out, label)
            loss2 = nn.CrossEntropyLoss()(d_out, identities)
           
            _, _, byol_loss, _ = self.coral_loss(self.N_pat, features, identities, self.device, self.bz, self.th)
            
            self.optimizer.zero_grad()
            (loss + self.p1 * loss2  +  self.p2 * byol_loss).backward()
            self.optimizer.step()
        print ('train_byol: class loss: {}, loss2: {}, byol_loss: {}---'.format(loss.item(), loss2.item(), byol_loss.item()))
        return loss, loss2, byol_loss
    
    # ablation 4
    def train_align(self, train_loader, device, epoch, N_epoch):
        self.model.train()
        for i, (X, label, identities) in tqdm(enumerate(train_loader), total=len(train_loader)):
            if "Mul" in self.model_name:
                x, cause_number, bc_number = X
                x = x.to(device, non_blocking=True)
                cause_number = cause_number.to(device, non_blocking=True)
                bc_number = bc_number.to(device, non_blocking=True)
                X = (x, cause_number, bc_number)
            else:
                X = X.to(device)
            label = label.to(device)
            identities = identities.to(device)
            
            p = float(i + epoch * len(train_loader)) / N_epoch / len(train_loader)
            alpha = 2. / (1. + np.exp(-10 * p)) - 1
            features, out, d_out = self.model(X, alpha)
            loss = nn.CrossEntropyLoss()(out, label)
            mean_loss, cova_loss, _, _ = self.coral_loss(self.N_pat, features, identities, self.device, self.bz, self.th)
            
            self.optimizer.zero_grad()
            (loss + self.p3 * (mean_loss + cova_loss)).backward() 
            if self.clip_value > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_value)
            self.optimizer.step()
        print ('train_all: class loss: {}, mean_loss: {}, cova_loss: {}---'.format(loss.item(), mean_loss.item(), cova_loss.item()))
        return loss, mean_loss, cova_loss
    
    # ablation 5
    def train_align_mmd(self, train_loader, device, epoch, N_epoch):
        self.model.train()
        for i, (X, label, identities) in tqdm(enumerate(train_loader), total=len(train_loader)):
            if "Mul" in self.model_name:
                x, cause_number, bc_number = X
                x = x.to(device, non_blocking=True)
                cause_number = cause_number.to(device, non_blocking=True)
                bc_number = bc_number.to(device, non_blocking=True)
                X = (x, cause_number, bc_number)
            else:
                X = X.to(device)
            label = label.to(device)
            identities = identities.to(device)
            
            p = float(i + epoch * len(train_loader)) / N_epoch / len(train_loader)
            alpha = 2. / (1. + np.exp(-10 * p)) - 1
            features, out, d_out = self.model(X, alpha)
            loss = nn.CrossEntropyLoss()(out, label)
            _, cova_loss, _, _ = self.coral_loss(self.N_pat, features, identities, self.device, self.bz, self.th)
            
            self.optimizer.zero_grad()
            (loss + self.p3 * (cova_loss)).backward()
            if self.clip_value > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_value)
            self.optimizer.step()
        print ('train_all: class loss: {}, mean_loss: {}---'.format(loss.item(), cova_loss.item()))
        return loss, cova_loss
    
    # ablation 6
    def train_mmd(self, train_loader, device, epoch, N_epoch):
        self.model.train()
        loss_collection = [[], [], [], []]

        for i, (X, label, identities) in tqdm(enumerate(train_loader), total=len(train_loader)):
            if "Mul" in self.model_name:
                x, cause_number, bc_number = X
                x = x.to(device, non_blocking=True)
                cause_number = cause_number.to(device, non_blocking=True)
                bc_number = bc_number.to(device, non_blocking=True)
                X = (x, cause_number, bc_number)
            else:
                X = X.to(device)
            label = label.to(device)
            identities = identities.to(device)
            
            p = float(i + epoch * len(train_loader)) / N_epoch / len(train_loader)
            alpha = 2. / (1. + np.exp(-10 * p)) - 1
            features, out, d_out = self.model(X, alpha)
            loss = nn.CrossEntropyLoss()(out, label)
            loss2 = nn.CrossEntropyLoss()(d_out, identities)
            mean_loss, _, byol_loss, _ = self.coral_loss(self.N_pat, features, identities, self.device, self.bz, self.th)
            loss_collection[0].append(loss.item())
            loss_collection[1].append(loss2.item())
            loss_collection[2].append(byol_loss.item())
            loss_collection[3].append(mean_loss.item())
            self.optimizer.zero_grad()
            (loss + self.p1 * loss2 + self.p2 * byol_loss + self.p3 * mean_loss).backward()
            if self.clip_value > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_value)
            self.optimizer.step()

        print ('train_all: class loss: {}, loss2: {}, byol_loss: {}, mean_loss: {} --- count:{}'.format(
            sum(loss_collection[0])/len(train_loader),
            self.p1 * sum(loss_collection[1])/len(train_loader),
            self.p2 * sum(loss_collection[2])/len(train_loader),
            self.p3 * sum(loss_collection[3])/len(train_loader),
            len(train_loader)
            ))
        return loss, loss2, mean_loss, byol_loss
    
    # ablation 7
    def train_cova(self, train_loader, device, epoch, N_epoch):
        self.model.train()
        loss_collection = [[], [], [], []]
        print("===^-^==="*20)
        for i, (X, label, identities) in tqdm(enumerate(train_loader), total=len(train_loader)):
            if "Mul" in self.model_name:
                x, cause_number, bc_number = X
                x = x.to(device, non_blocking=True)
                cause_number = cause_number.to(device, non_blocking=True)
                bc_number = bc_number.to(device, non_blocking=True)
                X = (x, cause_number, bc_number)
            else:
                X = X.to(device)
            label = label.to(device)
            identities = identities.to(device)
            
            p = float(i + epoch * len(train_loader)) / N_epoch / len(train_loader)
            alpha = 2. / (1. + np.exp(-10 * p)) - 1
            # Forward pass
            features, out, d_out = self.model(X, alpha)
            loss = nn.CrossEntropyLoss()(out, label)
            loss2 = nn.CrossEntropyLoss()(d_out, identities)
            _, cova_loss, byol_loss, _ = self.coral_loss(self.N_pat, features, identities, self.device, self.bz, self.th)
            loss_collection[0].append(loss.item())
            loss_collection[1].append(loss2.item())
            loss_collection[2].append(byol_loss.item())
            loss_collection[3].append(cova_loss.item())
            self.optimizer.zero_grad()
            (loss + self.p1 * loss2 + self.p2 * byol_loss + self.p3 * cova_loss).backward() 

            if self.clip_value > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_value)
            self.optimizer.step()

        print ('train_all: class loss: {}, loss2: {}, byol_loss: {}, mean_loss: {} --- count:{}'.format(
            sum(loss_collection[0])/len(train_loader),
            self.p1 * sum(loss_collection[1])/len(train_loader),
            self.p2 * sum(loss_collection[2])/len(train_loader),
            self.p3 * sum(loss_collection[3])/len(train_loader),
            len(train_loader)
            ))
        return loss, loss2, cova_loss, byol_loss
   
      
    def test(self, test_loader, device):
        self.model.eval()
        with torch.no_grad():
            result = np.array([])
            gt = np.array([])
            for X, y in test_loader:
                if "Mul" in self.model_name:
                    x, cause_number, bc_number = X
                    x = x.to(device, non_blocking=True)
                    cause_number = cause_number.to(device, non_blocking=True)
                    bc_number = bc_number.to(device, non_blocking=True)
                    X = (x, cause_number, bc_number)
                else:
                    X = X.to(device)
                _, convScore, _ = self.model(X)
                result = np.append(result, torch.max(convScore, 1)[1].cpu().numpy())
                gt = np.append(gt, y.numpy())
        return result, gt


class DocTmp_DA(MulDOCTer_EEG_DA):
    def __init__(self, device, voc_size=None, model=None, ehr_adj=None, ddi_adj=None, emb_size=40):
        super(DocTmp_DA, self).__init__(device)

        self.d_net = nn.Sequential(
            Rearrange('b t c -> b (t c)'),
            nn.Linear(1920, 128),  # 2s->1860
            nn.ELU(),
            nn.Dropout(0.5),
            nn.Linear(128, 32),
            nn.ELU(),
            nn.Dropout(0.5),
            nn.Linear(32, 2)
        )

       
    def Doc_pre_layers(self, x, x_random):
        x = self.forward_feature(x)
        x_random = self.forward_feature(x_random)
        return x, x_random

    def Doc_post_layers(self, org, SR_rep, CR_rep):
        org = self.proj(org)
        out = self.proj(SR_rep)
        out_random = self.d_net(CR_rep)
        return org, out, out_random

    def forward_train(self, x, x_random):
       
        x, x_random = self.Doc_pre_layers(x, x_random)

        x_mean = torch.mean(x, keepdim=True, dim=(2))

        x_random_mean = torch.mean(x_random, keepdim=True, dim=(2))
        x_std = torch.std(x, keepdim=True, dim=(2))
        x_random_std = torch.std(x_random, keepdim=True, dim=(2))
        gamma = np.random.uniform(0, 1)

        mix_mean, mix_std = x_random_mean, x_random_std
        SR_rep = (x - x_mean) / (x_std+1e-5) * mix_std + mix_mean

        CR_rep = (x_random - x_random_mean) / (x_random_std+1e-5) * x_std + x_mean
       
        return self.Doc_post_layers(x, SR_rep, CR_rep)

    def forward(self, x):
        x = self.forward_feature(x)
        out = self.proj(x)
        return out


class MulDOCTer(nn.Module):
    def __init__(self, model_name, device, args, lr=1e-3, weight_decay=1e-5):
        super(MulDOCTer, self).__init__()
        self.exptype = args.exptype
        print("DEBUG in MulDOCTer", "{}(device) depth={}".format(model_name, args.depth), args.exptype, "=!"*20 )
        self.model = eval("{}(device, args.dropout, args.depth, args.nheads)".format(model_name))
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)

    def train(self, train_loader, device):
        self.model.train()
        loss_collection = []
        count = 0
        for (X, y) in tqdm(train_loader):
            x, cause_number, bc_number = X
            x = x.to(device, non_blocking=True)
            cause_number = cause_number.to(device, non_blocking=True)
            bc_number = bc_number.to(device, non_blocking=True)
            X = (x, cause_number, bc_number)

            convScore = self.model(X)
            y = y.to(device)
            if "new" in self.exptype:
                class_weights = torch.tensor([2/3, 1/3], dtype=torch.float).to(device)
                loss = nn.CrossEntropyLoss(weight=class_weights)(convScore, y)
            else:
                loss = nn.CrossEntropyLoss()(convScore, y)
            count += convScore.shape[0]
            loss_collection.append(loss.item())
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        print ('train avg loss: {:.4}, count: {}, len of dataloader:{}'.format(sum(loss_collection) / len(loss_collection), len(loss_collection), len(train_loader)))
        return sum(loss_collection)/len(train_loader)  
    

    def test(self, test_loader, device):
        self.model.eval()
        with torch.no_grad():
            result = np.array([])
            gt = np.array([])
            for X, y in test_loader:
                x, cause_number, bc_number = X
                x = x.to(device, non_blocking=True)
                cause_number = cause_number.to(device, non_blocking=True)
                bc_number = bc_number.to(device, non_blocking=True)
                X = (x, cause_number, bc_number)

                convScore = self.model(X)
                result = np.append(result, torch.max(convScore, 1)[1].cpu().numpy())
                gt = np.append(gt, y.numpy())

        return result, gt