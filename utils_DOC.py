from glob import glob
import random
import re
import string
import time
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
import torch
import numpy as np
import os
import pickle
from einops import rearrange, reduce, repeat
from tqdm import tqdm
import matplotlib.pyplot as plt

def set_random_seed(seed=1234):
    seed = seed
    # Torch RNG
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Python RNG
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = True


def y_transform(y):
    if y == 'W':
        y = 0
    elif y == 'R':
        y = 4
    elif y in ['1', '2', '3']:
        y = int(y)
    elif y == '4':
        y = 3
    else:
        y = 0
    return y

def get_text_info(
        info_file = "/data3/yoro/data/DAdata/data08s_final_all_info.csv"    
    ):
    data_info = pd.read_csv(info_file, low_memory=False)
    print("len:", len(data_info))
    cause_map = {'脑出血': 0, '蛛网膜下腔出血': 0, '脑干出血': 0, '右侧颞叶硬膜外血肿': 0, '小脑出血': 0,
                  '双侧额叶脑出血': 0, '右侧枕叶脑出血': 0, '硬膜下血肿': 0,
                  'TBI': 1, 'TBI（创伤性脑损伤）': 1, '头外伤': 1,
                  '缺血缺氧性脑病': 2, '缺血缺氧性脑病（心脏骤停）': 2, '缺血缺氧': 2, '缺血缺氧性脑病（热射病）': 2,
                  '缺血缺氧性脑病（失血过多）': 2, '脑梗死': 2}  # 病因与病因标签的映射
    bc_set = set()
    bc_set2 = set()
    cause_count = {0: 0, 1: 0, 2: 0}

    map_cy = {}
    data_info_cy = pd.read_csv("/data3/yoro/data/DOC_subject_info.csv", encoding = "gb2312")
    for id, fname in zip(data_info_cy["subject_id"], data_info_cy["raw_file_path"]):
        f = fname.split("/")[-1]
        map_cy[f] = id
    
    res_info = {}
    for id, pre, f, cause, bc in zip(data_info["npy文件编号"], data_info["npy文件前缀"], data_info["vhdr位置"], data_info["病因"], data_info["病程（月）"]):
        cause_number = np.array([cause_map[cause]])
        cause_count[cause_map[cause]] += 1
        bc_number = None 
        bc_set.add(bc)
        if bc == '' or bc == '缺如' or bc == ' ' or (isinstance(bc, float) and pd.isna(bc)):
            bc_number = 0
        else:
            bc = int(float(bc))
            if 1 <= bc <= 10:
                bc_number = bc
            elif bc > 10:
                bc_number = 11
            else:
                raise NotImplementedError("bc < 1, which is Not expected.")
        bc_set2.add(bc_number)
        bc_number = np.array([bc_number])
        # print("DDD", id, cause_number, bc_number)
        res_info[id] = np.concatenate([cause_number, bc_number])        
      
    # print(bc_set)
    # print(len(res_info))
    print("cause_count:", cause_count, "res_info len:", len(res_info))
    print("bc_set2:", bc_set2)
    return res_info

region = {
    'LA': [0, 2, 10, 19, 23, 27, 31, 37, 39, 45, 51, 53],
    'RA': [1, 3, 11, 20, 24, 28, 32, 38, 40, 46, 52, 54],
    'LP': [6, 8, 14, 21, 25, 29, 35, 41, 43, 49, 55, 57],
    'RP': [7, 9, 15, 22, 26, 30, 36, 42, 44, 50, 56, 58],
    'Center': [4, 5, 12, 13, 16, 17, 18, 33, 34, 47, 48, 59, 60, 61],
    'EOG': [63]
}

class DocIDLoader(torch.utils.data.Dataset):
    def __init__(self, datalist, ID, path, mean_v = None, std_v = None, model_name = ""):
        self.ID = ID
        self.path = path
        self.datalist = datalist
        tlen = np.load(self.path + datalist[0]).shape[-1]
        self.mean = repeat(mean_v, "n -> n t", t = tlen)
        self.std = repeat(std_v, "n -> n t", t = tlen)
        self.model_name  = model_name
        if 'Mul' in self.model_name:
            self.index = region['LA'] + region['RA'] + region['LP'] + region['RP'] + region['Center'] + region["EOG"]
            self.bybc_map = get_text_info()
    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, idx):
        label = int(self.datalist[idx][0])
        if "Mul" in self.model_name:
            out = np.load(self.path + self.datalist[idx])
            out = (out - self.mean)/self.std
            out = out[self.index, ::8]
            out = torch.as_tensor(out, dtype = torch.float)
            user = self.datalist[idx].split('_')[1]
            bybc = self.bybc_map[user]
            out = (out, bybc[0], bybc[1])
        else:
            out = np.load(self.path + self.datalist[idx])
            out = (out - self.mean)/self.std
            out = out[:62, ::8] 
            out = rearrange(out, "c t -> 1 c t")
            out = torch.as_tensor(out, dtype = torch.float)
        return out, label, self.ID[idx]



class DocLoader(torch.utils.data.Dataset):
    def __init__(self, datalist, path, mean_v, std_v, model_name, len_old=0, len_new=0):
        self.path = path
        self.datalist = datalist
        self.model_name = model_name
        self.len_old, self.len_new = len_old, len_new
        tshape = np.load(self.path + datalist[0]).shape
        if "image" in path:
            self.mean = mean_v.reshape(1, 4, 1, 1)
            self.std = std_v.reshape(1, 4, 1, 1)
        else:
            self.mean = repeat(mean_v, "n -> n t", t = tshape[-1])
            self.std = repeat(std_v, "n -> n t", t = tshape[-1])
        self.index, self.bybc_map = None, None 
        if 'Mul' in model_name:
            self.index = region['LA'] + region['RA'] + region['LP'] + region['RP'] + region['Center'] + region["EOG"]
            self.bybc_map = get_text_info()

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, idx):
        label = int(self.datalist[idx][0])
        # print(self.datalist[idx])
        
        if "org" in self.model_name and "Mul" not in self.model_name:
            out = np.load(self.path + self.datalist[idx])
            out = (out - self.mean)/self.std
            out = out[self.index, ::8]
            out = torch.as_tensor(out, dtype = torch.float)
    
        elif 'Mul' in self.model_name:
            out = np.load(self.path + self.datalist[idx])
            out = (out - self.mean)/self.std
            out = out[self.index, ::8]
            out = torch.as_tensor(out, dtype = torch.float)
            user = self.datalist[idx].split('_')[1]
            bybc = self.bybc_map[user]
            out = (out, bybc[0], bybc[1])

        else: 
            out = np.load(self.path + self.datalist[idx])
            out = (out - self.mean)/self.std
            out = torch.as_tensor(out, dtype = torch.float)
        return out, label
    

class DocDoubleLoader(torch.utils.data.Dataset):
    def __init__(self, datalist, datalist_aux, path, mean_v = None, std_v = None):
        self.path = path
        self.datalist = datalist
        self.datalist_aux = datalist_aux
        tlen = np.load(self.path + datalist[0]).shape[-1]
        
        self.mean = repeat(mean_v, "n -> n t", t = tlen)
        self.std = repeat(std_v, "n -> n t", t = tlen)
        self.index = region['LA'] + region['RA'] + region['LP'] + region['RP'] + region['Center'] + region["EOG"]
        self.bybc_map = get_text_info()

    def __len__(self):
        return len(self.datalist)
        
    def __getitem__(self, idx):
        
        label = int(self.datalist[idx][0])
        out = np.load(self.path + self.datalist[idx])
        out = (out - self.mean)/self.std
        out = out[self.index, ::8] 
        out = torch.as_tensor(out, dtype = torch.float)
        user = self.datalist[idx].split('_')[1]
        bybc = self.bybc_map[user]
        out = (out, bybc[0], bybc[1])

        out_aux = np.load(self.path + self.datalist_aux[idx])
        out_aux = (out_aux - self.mean)/self.std
        out_aux = out_aux[:, ::8] 
        out_aux = torch.as_tensor(out_aux, dtype = torch.float)
        out_aux = (out_aux, bybc[0], bybc[1])
        if user != self.datalist_aux[idx].split('_')[1]:
            assert 0, "has a bug"
        return out, out_aux, label, label

def get_mean_std(nplist, path):
    t_begin = time.time()
    shape = np.load(path + nplist[0]).shape
    print("DEBUG-path:{} | number of files:{} | shape:{}".format(path, len(nplist), shape))
    sum = np.zeros([shape[0],])
    timelen = shape[1]
    for fp in tqdm(nplist):
        da = np.load(path + fp)
        sum += np.sum(da, axis = 1)
    mean = sum / (len(nplist)*timelen)
    
    t_mean = repeat(mean, "n -> n t", t = timelen)
    sumstd = np.zeros(list(shape))
    for fp in tqdm(nplist):
        da = np.load(path + fp)
        sumstd += (da - t_mean)**2 
    std = (np.sum(sumstd, axis = 1)/(len(nplist)*timelen))**0.5
    print("DEBUG mean&std------------------------------------------------time:", time.time()-t_begin)
    return mean, std


def get_meanstd_image(nplist, path):
    t_begin = time.time()
    shape = np.load(path + nplist[0]).shape
    print("DEBUG-path:{} | number of files:{} | shape:{}".format(path, len(nplist), shape))
    sum = np.zeros([shape[1],])  # (8, 4, 32, 32)
    timelen = 0
    for fp in tqdm(nplist):
        da = np.load(path + fp)
        for i in range(shape[1]):
            sum[i] += np.sum(da[:, i, :, :])
        timelen += da.shape[0]*32*32
    mean = sum / timelen
    
    t_mean = repeat(mean, "c -> t c h w", t = shape[0], h = shape[2], w = shape[3])
    sumstd = np.zeros(list(shape))
    for fp in tqdm(nplist):
        da = np.load(path + fp)
        sumstd += (da - t_mean)**2 
    std = reduce(sumstd, 't c h w -> c', reduction='sum')
    std = (std/timelen)**0.5
    
    print("DEBUG mean&std------------------------------------------------time:", time.time()-t_begin)
    return mean, std


# BrainAmp mcs:147 uws:134
def load(path, seed, exptype="DG", foldn=5, maxm=147, maxu=134):
    f_paths = glob(path + '**.npy', recursive=True)
    f_paths.sort()
    X, y, groups = [], [], []
    mcs_num, uws_num, mcs_pat_num, uws_pat_num = 0, 0, 0, 0
    DA_X, DA_y, DA_groups = [], [], []
    DA_mcs_num, DA_uws_num = 0, 0
    trash_pat = ["uws7", "mcs70", "mcs41", "mcs28", "uws74", "uws36", "mcs152", "mcs223", "mcs8", "mcs87", "mcs88", "uws34"] # mcs:8 uws:4
    trash_pat.extend( ['uws77', 'mcs21', 'uws70', 'mcs126', 'uws3', 'mcs19', 'mcs15', 'mcs73', 'mcs51', 'uws84'] )# 新处理的设备多一些个体！
    
    sub_num = {} 
    for fp in f_paths:
        fname = fp.split("/")[-1]
        tmp_num = int(re.findall("\d+", fname)[-1])+1
        if 'uws' in fp:
            res = re.findall("uws[1-9]\\d*", fname)[0]
        elif 'mcs' in fp:
            res = re.findall("mcs[1-9]\\d*", fname)[0]
        if res in sub_num:
            sub_num[res] = max(sub_num[res], tmp_num)
        else:
            sub_num[res] = tmp_num

    for fp in f_paths:
        fname = fp.split("/")[-1]
        tmp_num = re.findall("\d+", fname)
        if int(tmp_num[-1]) >= 30:
            continue
        # '(uws|mcs)[1-9]\\d*'
        if 'uws' in fp:
            res = re.findall("uws[1-9]\\d*", fname)
            # print(res, re.findall("\d+", fname))
            num = int(re.findall("\d+", res[0])[0])
            if res[0] in trash_pat or num > maxu or sub_num[res[0]] < 30 or (num >= 140 and num <= 164): # 134 139 范围应该是【140，164】（已改回），但是不够了借一个
                if ("DA" in exptype or "new" in exptype) and num >= 140 and num <= 164 and sub_num[res[0]] >= 30:
                    DA_X.append(fname)
                    DA_y.append(int(fname[0]))
                    DA_groups.extend(res)
                    DA_uws_num += 1
                continue
            uws_num += 1
        elif 'mcs' in fp:
            res = re.findall("mcs[1-9]\\d*", fname)
            num = int(re.findall("\d+", res[0])[0])
            if res[0] in trash_pat or num > maxm or sub_num[res[0]] < 30 or (num >= 161 and num <= 209): # 147 152
                if ("DA" in exptype or "new" in exptype) and num >= 161 and num <= 209 and sub_num[res[0]] >= 30:
                    DA_X.append(fname)
                    DA_y.append(int(fname[0]))
                    DA_groups.extend(res)
                    DA_mcs_num += 1
                continue 
            mcs_num += 1
        else:
            assert 0, "error! cann't found mcs and uws"
        X.append(fname)
        y.append(int(fname[0]))
        groups.extend(res)
    if "new" in exptype:
        X, y, groups = DA_X, DA_y, DA_groups
        mcs_num, uws_num = DA_mcs_num, DA_uws_num
    DA_mcs_pat_num, DA_uws_pat_num = 0, 0  
    for pat in set(groups):
        if "mcs" in pat:
            mcs_pat_num += 1
        elif "uws" in pat:
            uws_pat_num += 1
        else:
            assert 0
    for pat in set(DA_groups):
        if "mcs" in pat:
            DA_mcs_pat_num += 1
        elif "uws" in pat:
            DA_uws_pat_num += 1
        else:
            assert 0
    
    if "DG" in exptype:
        print("DEBUG-all in old device data number len(x)={} len(y)={} len(set(groups))={} mcs_slice_num={} uws_slice_num={} mcs_pat_num={} uws_pat_num={}!".format(len(X), len(y), len(set(groups)), mcs_num, uws_num, mcs_pat_num, uws_pat_num))
        sgkf = StratifiedGroupKFold(n_splits=foldn, shuffle=True, random_state=seed)  # !!!
        res_map = {}
        for fold_index, (train, test) in zip(range(foldn), sgkf.split(X, y, groups=groups)):
            print("DEBUG-fold_index:{} trainlen:{} testlen:{} samples=>{}".format(fold_index, len(train), len(test), test[:20:100]))
            train_np, pat, test_np = [], [], []
            for idx in train:
                train_np.append(X[idx])
                pat.append(groups[idx])
            for idx in test:                
                test_np.append(X[idx])

            has_pat = set()
            pat_id = []
            id = -1
            for p in pat:
                if p not in has_pat:
                    has_pat.add(p)
                    id += 1
                pat_id.append(id)
    
            if "image" in path:
                mean_v, std_v = get_meanstd_image(train_np, path) 
            else:
                mean_v, std_v = get_mean_std(train_np, path)
            print("DEBUG result len_train:{} len_test:{} (trainset)len_p:{}!!!".format(len(train_np), len(test_np), len(has_pat)))
            res_map[fold_index] = {"trainset": train_np, "testset": test_np, "pat": pat_id, "id": len(has_pat), "mean": mean_v, "std": std_v}
    elif "Mix" in exptype:
        sgkf = StratifiedGroupKFold(n_splits=foldn, shuffle=True, random_state=seed)  # !!!
        res_map = {} 

        for fold_index, (train, test) in zip(range(foldn), sgkf.split(DA_X, DA_y, groups=DA_groups)):
            print("DEBUG-fold_index:{} trainlen:{} testlen:{} samples=>{}".format(fold_index, len(train), len(test), test[:20:100]))
            train_np, pat, test_np = [], [], []
            for idx in train:
                train_np.append(DA_X[idx])
                pat.append(DA_groups[idx])
            for idx in test:                
                test_np.append(DA_X[idx])

            if "T" in exptype:
                train_np = X 
                pat = groups
            else:
                train_np.extend(X) 
                pat.extend(groups)

            has_pat = set()
            pat_id = []
            id = -1
            for p in pat:
                if p not in has_pat:
                    has_pat.add(p)
                    id += 1
                pat_id.append(id)
          
            if "image" in path:
                mean_v, std_v = get_meanstd_image(train_np, path) 
            else:
                mean_v, std_v = get_mean_std(train_np, path)
                # mean_v, std_v = np.ones([64,]), np.ones([64,])
            print("DEBUG result len_train:{} len_test:{} (trainset)len_p:{}!!!".format(len(train_np), len(test_np), len(has_pat)))
            res_map[fold_index] = {"trainset": train_np, "testset": test_np, "pat": pat_id, "id": len(has_pat), "mean": mean_v, "std": std_v, "len_old": len(X), "len_new": len(train_np)-len(X)}

    elif exptype == "DA":
        print("DEBUG-all-DA in new device data number len(DA_x)={} len(DA_y)={} len(set(DA_groups))={} DA_mcs_slice_num={} DA_uws_slice_num={} DA_mcs_pat_num={} DA_uws_pat_num={}!".format(len(DA_X), len(DA_y), len(set(DA_groups)), DA_mcs_num, DA_uws_num, DA_mcs_pat_num, DA_uws_pat_num))
        sgkf = StratifiedGroupKFold(n_splits=foldn, shuffle=True, random_state=seed)  # !!!
        res_map = {} 
        for fold_index, (train, test) in zip(range(foldn), sgkf.split(DA_X, DA_y, groups=DA_groups)):
            train_np, pat, test_np = [], [], []
            for idx in train:
                train_np.append(DA_X[idx])
                pat.append(DA_groups[idx])
            for idx in test:                
                test_np.append(DA_X[idx])

            for idx in range(len(X)):
                train_np.append(X[idx])
                pat.append(groups[idx])

            has_pat = set()
            pat_id = []
            id = -1
            for p in pat:
                if p not in has_pat:
                    has_pat.add(p)
                    id += 1
                pat_id.append(id)

            if "image" in path:
                mean_v, std_v = get_meanstd_image(train_np, path) 
            else:
                mean_v, std_v = get_mean_std(train_np, path)
            
            print("DEBUG DA result len_train:{} len_test:{} (trainset)len_p:{}!!!".format(len(train_np), len(test_np), len(has_pat)))
            res_map[fold_index] = {"trainset": train_np, "testset": test_np, "pat": pat_id, "id": len(has_pat), "mean": mean_v, "std": std_v}
    else:
        assert 0
    return res_map


# BrainAmp mcs:147 uws:134
def load_for_dev(path, seed, foldn = 5):
    f_paths = glob(path + '**.npy', recursive=True)
    f_paths.sort() 
    X, y, groups = [], [], []
    mcs_num, uws_num, mcs_pat_num, uws_pat_num = 0, 0, 0, 0
    trash_pat = ["uws7", "mcs70", "mcs41", "mcs28", "uws74", "uws36", "mcs152", "mcs223", "mcs8", "mcs87", "mcs88", "uws34"]
    sub_num = {}
    for fp in f_paths:
        fname = fp.split("/")[-1]
        tmp_num = int(re.findall("\d+", fname)[-1])+1
        if 'uws' in fp:
            res = re.findall("uws[1-9]\\d*", fname)[0]
        elif 'mcs' in fp:
            res = re.findall("mcs[1-9]\\d*", fname)[0]
        if res in sub_num:
            sub_num[res] = max(sub_num[res], tmp_num)
        else:
            sub_num[res] = tmp_num
    
    for fp in f_paths:
        fname = fp.split("/")[-1]
        tmp_num = re.findall("\d+", fname)
        if int(tmp_num[-1]) >= 30:
            continue
        if 'uws' in fp:
            res = re.findall("uws[1-9]\\d*", fname)
            num = int(re.findall("\d+", res[0])[0])
            if  res[0] in trash_pat or num > 134 or sub_num[res[0]] < 30:
                continue 
            uws_num += 1
        elif 'mcs' in fp:
            res = re.findall("mcs[1-9]\\d*", fname)
            num = int(re.findall("\d+", res[0])[0])
            if res[0] in trash_pat or num > 147 or sub_num[res[0]] < 30:
                continue 
            mcs_num += 1
        else:
            assert 0, "error! cann't found mcs and uws"
       
        X.append(fname)
        y.append(int(fname[0]))
        groups.extend(res)
    for pat in set(groups):
        if "mcs" in pat:
            mcs_pat_num += 1
        elif "uws" in pat:
            uws_pat_num += 1
        else:
            assert 0

    print("DEBUG-all data number len(x)={} len(y)={} len(set(groups))={} mcs_slice_num={} uws_slice_num={} mcs_pat_num={} uws_pat_num={}!".format(len(X), len(y), len(set(groups)), mcs_num, uws_num, mcs_pat_num, uws_pat_num))
    sgkf = StratifiedGroupKFold(n_splits=foldn, shuffle=True, random_state=seed)  # !!!
    res_map = {} 
    for fold_index, (train, test) in zip(range(foldn), sgkf.split(X, y, groups=groups)):
        print("DEBUG-fold_index:{} trainlen:{} testlen:{} samples=>{}".format(fold_index, len(train), len(test), test[:20:100]))
        train_np, train_np_aux, test_np = [], [], []
        map_pat = {}
        for idx in train:
            if groups[idx] not in map_pat:
                map_pat[groups[idx]] = []
            map_pat[groups[idx]].append(X[idx])
        for idx in test:                
            test_np.append(X[idx])

        for (k, v) in map_pat.items():
            train_np += v[:len(v)//2 + 1] 
            train_np_aux += v[-len(v)//2 - 1:]
        mean_v, std_v = get_mean_std(train_np + train_np_aux, path)
        res_map[fold_index] = {"trainset": train_np, "trainauxset": train_np_aux, "testset": test_np, "id": len(map_pat), "mean": mean_v, "std": std_v}
    
    return res_map

def convertMat(mat):
    # print(mat)
    mat = mat.strip(" ") # 去除首尾空格
    mat = mat.replace("[", "").replace("]", "") # 替换[]
    mat = mat.strip(" ")
    mat = re.sub(" +", ",", mat) # 将所有空格换为','
    
    mat_list = [int(x) for x in mat.split(",")]
    
    return np.array([[mat_list[0], mat_list[1]], [mat_list[2], mat_list[3]]])
def statcsv(path):
    if path == "" or path is None:
        print("invalid input path")
        return
    test_group = pd.read_csv(path)
    sacc, sre, sf1, cmat = np.array([]), np.array([]), np.array([]), np.array([[0, 0], [0, 0]])
    mf1 = np.array([])
    mpre = np.array([])
    num = 0
    gid, seed, trainseed = test_group["group_index"][0], test_group["seed"].iloc[0], test_group["trainseed"].iloc[0] # 获取读取元素的方法
    for acc, re, f1, mat in zip(test_group["acc"], test_group["recall"], test_group["f1"], test_group["acc_cmat"]):
        sacc = np.append(sacc, acc)
        sf1 = np.append(sf1, f1)
        sre = np.append(sre, re)
        
        cmat += convertMat(mat)
        num += 1
        mat = convertMat(mat)
        p2, r2 = mat[0][0]/(mat[0][0]+mat[1][0]), mat[0][0]/(mat[0][0]+mat[0][1])
        f1_0 = 2/(1/p2+1/r2)
        # p1, r1 = mat[1][1]/(mat[0][1]+mat[1][1]), mat[1][1]/(mat[1][0]+mat[1][1])
        mf1 = np.append(mf1, (f1+f1_0)/2)
        mpre = np.append(mpre, mat[1][1]/(mat[0][1]+mat[1][1]))
        # print("f1 = {:.4f} turef1 = {:.4f} f2 = {:.4f}".format(2/(1/p1+1/r1), f1, 2/(1/p2+1/r2)))
        np.set_printoptions(linewidth=np.inf)

    print("gid:{}\t fold:{}|\t seed:{}\t trainseed:{}\t acc:{:.4f}-{:.4f}\t recall:{:.4f}-{:.4f}\t precision:{:.4f}-{:.4f}\t f1:{:.4f}-{:.4f}\t macro-f1:{:.4f}-{:.4f} mat:{} !".format(gid, num, seed, trainseed, np.mean(sacc), np.std(sacc), np.mean(sre), np.std(sre),np.mean(mpre), np.std(mpre), np.mean(sf1), np.std(sf1), np.mean(mf1), np.std(mf1), str(cmat).replace("\n", "")))
    # print(sacc)

if __name__ == "__main__":
    statcsv(path="/data3/yoro/code/DG/log/DOC/DOC_Tmp_ablation-0_seed-2025_trainseed-99_time-20241105-12_cuda-1/20241105-12_99.csv")

  