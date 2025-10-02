from glob import glob
import mne
import pandas as pd
import numpy as np
import re
import os 
from einops import repeat
import time
from tools import *


device1_ch = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7', 'F8', 'T7', 
'T8', 'P7', 'P8', 'Fz', 'Cz', 'Pz', 'FC1', 'FC2', 'CP1', 'CP2', 'FC5', 'FC6', 'CP5', 
'CP6', 'FT9', 'FT10', 'TP9', 'TP10', 'F1', 'F2', 'C1', 'C2', 'P1', 'P2', 'AF3', 'AF4', 'FC3', 
'FC4', 'CP3', 'CP4', 'PO3', 'PO4', 'F5', 'F6', 'C5', 'C6', 'P5', 'P6', 'AF7', 'AF8', 'FT7', 'FT8', 
'TP7', 'TP8', 'PO7', 'PO8', 'CPZ', 'POZ', 'OZ', 'FPZ', 'IO']

device2_ch = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7', 'F8', 'T7', 
'T8', 'P7', 'P8', 'Fz', 'Cz', 'Pz', 'FC1', 'FC2', 'CP1', 'CP2', 'FC5', 'FC6', 
'CP5', 'CP6', 'FT9', 'FT10', 'TP9', 'TP10', 'F1', 'F2', 'C1', 'C2', 'P1', 'P2', 'AF3', 
'AF4', 'FC3', 'FC4', 'CP3', 'CP4', 'PO3', 'PO4', 'F5', 'F6', 'C5', 'C6', 'P5', 'P6', 
'AF7', 'AF8', 'FT7', 'FT8', 'TP7', 'TP8', 'PO7', 'PO8', 'CPz', 'POz', 'Oz', 'AFz', 'EYE2']


def check(data, max_v, nor_v, NORV_TH = 50):
    data = abs(data)
    if np.max(data) > max_v:
        return False
    tmp = 0
    for i in range(data.shape[1]):
        if abs(np.max(data[:, i])) > nor_v:
            tmp += 1
    if tmp > NORV_TH:
        return False
    return True
        

def generate_npy(pick_chs, 
                    data_file,
                    sub_info = "/data/info.xlsx",
                    resfile = "DOC_window_info_alldata_old_noref.csv", 
                    band=(0.5, 40), 
                    WINDOW_LENGTH_SECONDS = 8, 
                    sfile = 'allorg_{}s_notch_2/', 
                    isref = True, 
                    dawnsample_rate = -1, 
                    dfile = "/data3/yoro/data/DAdata/"):
    # Noise data
    trash_sub = [] 
    file_paths = glob(data_file, recursive=True)
    csvfile = dfile + resfile
    datafile = dfile + sfile.format(WINDOW_LENGTH_SECONDS)
    if not os.path.exists(dfile):
        os.mkdir(dfile)
        print("*** DEBUG make a csv file:", dfile)
    if not os.path.exists(datafile):
        os.mkdir(datafile)
        print("*** DEBUG make a data file:",datafile)
    MAXV, NORV = 180 * 10e-6, 180 * 1e-6 
    dataset_index_rows = []
    pattern = '[0-9]\\d*'
    all_num = 0

    info = pd.read_excel(sub_info)

    SAMPLING_FREQ = 1000
    WINDOW_LENGTH_SAMPLES = int(WINDOW_LENGTH_SECONDS * SAMPLING_FREQ)  

    for _, file_path in enumerate(file_paths):     
        vhdr_name = file_path.split("/")[-1][:-5]  
 
        mask = info["编号"] == vhdr_name
        result = info[mask]
        npy_name = result.loc[result.index[0], "生成npy编号"]
        print("=== DEBUG process sub:", npy_name, " ==> ", vhdr_name, file_path)
        # 丢弃脏数据
        if npy_name in trash_sub:
            print(npy_name, "is trash", "!"*50)
            continue

        if 'MCS' in file_path or 'mcs' in file_path:
            label = "1_"+npy_name+"_"
        elif 'UWS' in file_path or 'uws' in file_path:
            label = "0_"+npy_name+"_"
        else:
            continue
        all_num += 1
        raw_org = mne.io.read_raw_brainvision(file_path, verbose='ERROR', preload=True)
   
        Ref_channels = []

        raw_org.pick_channels(pick_chs)
        raw_org.reorder_channels(pick_chs)
        Ref_channels = pick_chs[:-1]

        print("\t channel length check: {}; data shape check: {}; Ref num check: {}".format(
                len(pick_chs) == len(raw_org.info["ch_names"]), 
                raw_org.get_data(start=0, stop=WINDOW_LENGTH_SAMPLES).shape, 
                len(Ref_channels)))
        
        raw_data = raw_org.copy().filter(l_freq=band[0], h_freq=band[1], verbose='ERROR')  
        if "raw_data4/mcs" in file_path or select == 2:  # 12/9
            print("DEBUG notch1:", file_path, "!="*20)
            raw_data.notch_filter(freqs=50, fir_design='firwin')  
        
        new_data = None
        num = 0
        rej = 0
        rej_index, acc_index = [], []
        eog_data = None
        for start_sample_index in range(0, int(int(raw_data.times[-1]) * SAMPLING_FREQ), WINDOW_LENGTH_SAMPLES):
            num += 1
            end_sample_index = start_sample_index + (WINDOW_LENGTH_SAMPLES - 1)
            if end_sample_index > raw_data.n_times:
                break
            window_data = raw_data.get_data(start=start_sample_index, stop=end_sample_index + 1)
            window_data_org = raw_org.get_data(start=start_sample_index, stop=end_sample_index + 1) # 未滤波数据
     
            if not check(window_data[:-1], MAXV, NORV): # 不包括EOG
                rej_index.append(num-1)
                rej += 1
                continue
            if new_data is None:
                acc_index.append(num-1)
                new_data = window_data_org  
            else:
                acc_index.append(num-1)
                new_data = np.concatenate((new_data, window_data_org), axis = 1)
        print("=== DEBUG rej rate=", rej/num, "-------------------")
        if rej == num or new_data is None:
            continue
        raw = mne.io.RawArray(new_data, raw_data.info)  # all band, pick channels
        raw.filter(l_freq=band[0], h_freq=band[1], verbose='ERROR')                         
        if "raw_data4/mcs" in file_path or select == 2:  # 12/9
            print("DEBUG notch2:", file_path, "!="*20)
            raw.notch_filter(freqs=50, fir_design='firwin') 
     
        eog_data = raw.copy().get_data()[-1, :]
        raw.drop_channels(pick_chs[-1])
        print("==== DEBUG after drop raw shape: {} eog shape: {}".format(raw.get_data().shape, eog_data.shape))
        if isref == True:
            raw.set_eeg_reference()  # 共平均参考
        if dawnsample_rate != -1:
            raw.resample(dawnsample_rate)
            SAMPLING_FREQ = dawnsample_rate
            WINDOW_LENGTH_SAMPLES = SAMPLING_FREQ * WINDOW_LENGTH_SECONDS
      
        j = 0
        for start_sample_index in range(0, int(int(raw.times[-1]) * SAMPLING_FREQ), WINDOW_LENGTH_SAMPLES):
            end_sample_index = start_sample_index + (WINDOW_LENGTH_SAMPLES - 1)
            window_data = raw.get_data(start=start_sample_index, stop=end_sample_index + 1)
            # fname = label + str(acc_index[j]) + ".npy"
            fname = label + str(j) + ".npy" 
            this_eog_data = np.expand_dims(eog_data[start_sample_index: end_sample_index + 1], axis=0)
            save_data = np.concatenate((window_data, this_eog_data), axis = 0)
            # print("j check:", save_data[-1] == this_eog_data, save_data.shape)
            np.save( datafile + fname, save_data)           
            j += 1

        row = {}                                        # 包含切片数量和源文件位置
        row["npy文件编号"] = npy_name
        row["编号"] = result.loc[result.index[0], "编号"]
        row["vhdr位置"] = file_path
        row["原raw文件总采集时长"] = len(raw_data)
        row["去除异常片段后的总采集时长"] = len(raw)
        row["包含的通道"] = pick_chs
        row["通道数量"] = len(raw.info["ch_names"]) 
        row["npy文件前缀"] = label
        row["有效切片数量"] = num-rej
        row["切片拒绝率"] = rej/num
        row["拒绝切片索引"] = rej_index
        row["性别"] = result.loc[result.index[0], "性别"]
        row["年龄"] = result.loc[result.index[0], "年龄"]
        row["病因"] = result.loc[result.index[0], "病因"]
        row["损伤部位"] = result.loc[result.index[0], "损伤部位"]
        row["病程（月）"] = result.loc[result.index[0], "病程（月）"]
        row["听觉"] = result.loc[result.index[0], "听觉"]
        row["视觉"] = result.loc[result.index[0], "视觉"]
        row["运动"] = result.loc[result.index[0], "运动"]
        row["语言"] = result.loc[result.index[0], "语言"]
        row["交流"] = result.loc[result.index[0], "交流"]
        row["唤醒"] = result.loc[result.index[0], "唤醒"]
        row["总分"] = result.loc[result.index[0], "总分"]
        dataset_index_rows.append(row)
        print("\t", label[:-1], "success!=================================================================================")
    df = pd.DataFrame(dataset_index_rows, columns=["npy文件编号",
                                                    "编号",
                                                    "vhdr位置",
                                                    "原raw文件总采集时长",
                                                    "去除异常片段后的总采集时长",
                                                    "包含的通道",
                                                    "通道数量",
                                                    "npy文件前缀",
                                                    "有效切片数量",
                                                    "切片拒绝率",
                                                    "拒绝切片索引",
                                                    "性别",
                                                    "年龄",
                                                    "病因",
                                                    "损伤部位",
                                                    "病程（月）",
                                                    "听觉",
                                                    "视觉",
                                                    "运动",
                                                    "语言",
                                                    "交流",
                                                    "唤醒",
                                                    "总分"
                                                    ])

    df.to_csv(csvfile, index=False, encoding="utf_8_sig", mode="a")
    print("End up with df.shape:", df.shape, all_num, csvfile)