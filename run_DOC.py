import csv
import os
import argparse
import utils_DOC
import torch
import numpy as np
import time
from model_DOC import *
from sklearn.metrics import f1_score,accuracy_score,precision_score,recall_score, confusion_matrix
import matplotlib.pyplot as plt

def plot_loss(foldi, epoch, res, savepath):
    plt.cla()
    x = np.linspace(1, epoch, epoch) 
 
    plt.ylim(0, 1)  
    for k, v in res.items():
        if len(v) == 0:
            continue
        print(k, ":", v)
        plt.plot(x, v, label=k)
        np.save(savepath + "{}_{}.npy".format(foldi, k), v)
    plt.xlabel('Number of epochs')  
    plt.ylabel('loss')  
    plt.legend()  
    plt.grid(True)  

    plt.savefig(savepath + '{}_train_loss.jpg'.format(foldi))
    plt.show()


def accuracy_score_diy(y_true, y_pred):
    return np.sum(y_pred == y_true) / len(y_true)

def confusion_matrix(actual, pred):
    actual = np.array(actual)
    pred = np.array(pred)
    n_classes = int(max(np.max(actual), np.max(pred)) + 1)
    confusion = np.zeros([n_classes, n_classes])
    for i in range(n_classes):
        for j in range(n_classes):
            confusion[i, j] = np.sum((actual == i) & (pred == j))
    return confusion.astype('int')

def weighted_f1(gt, pre):
    confusion = confusion_matrix(gt, pre)
    f1_ls = []
    for i in range(confusion.shape[0]):
        if confusion[i, i] == 0:
            f1_ls.append(0)
        else:
            precision_tmp = confusion[i, i] / confusion[i].sum()
            recall_tmp = confusion[i, i] / confusion[:, i].sum()
            f1_ls.append(2 * precision_tmp * recall_tmp / (precision_tmp + recall_tmp))
    return np.mean(f1_ls)

def cohen_kappa_score(y1, y2):
    confusion = confusion_matrix(y1, y2)
    n_classes = confusion.shape[0]
    sum0 = np.sum(confusion, axis=0)
    sum1 = np.sum(confusion, axis=1)
    expected = np.outer(sum0, sum1) / np.sum(sum0)

    w_mat = np.ones([n_classes, n_classes], dtype=int)
    w_mat.flat[:: n_classes + 1] = 0

    k = np.sum(w_mat * confusion) / np.sum(w_mat * expected)
    return 1 - k


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=2025, help="seed")
    parser.add_argument('--trainseed', type=int, default=99, help="trainseed")
    parser.add_argument('--model', type=str, default="MulTmp", help="choose from model")
    parser.add_argument('--cuda', type=int, default=1, help="which cuda")
    parser.add_argument('--exptype', type=str, default="DG", help="DG_new | DG | DA")
    parser.add_argument('--param', type=int, default=0, help="param of ablation")
    parser.add_argument('--depth', type=int, default=5, help="transformer layer") 
    parser.add_argument('--nheads', type=int, default=2, help="heads of atten 2")
    parser.add_argument('--batch_size', type=int, default=512, help='batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate.')
    parser.add_argument('--wd', type=float, default=0.00001, help='weight_decay')
    parser.add_argument('--p1', type=float, default=0.05, help='adv loss')
    parser.add_argument('--p2', type=float, default=0.1, help='byol loss')
    parser.add_argument('--p3', type=float, default=0.8, help='align loss')
    parser.add_argument('--ps', type=float, default=0.1, help='align loss')
    parser.add_argument('--th', type=float, default=4, help='simple number th')
    parser.add_argument('--clip_value', type=float, default=0, help='防止nan')
    parser.add_argument('--dropout', type=float, default=0.5, help='dropout')
    parser.add_argument('--dataset', type=str, default="DOCtransformer", help="dataset name")
    parser.add_argument('--MLDG_threshold', type=int, default=1024, help="threshold for MLDG")
    parser.add_argument('--epochs', type=int, default=100, help="N of epochs")
    parser.add_argument('--fold_num', type=int, default=5, help="N of folds")
    parser.add_argument('--num_workers', type=int, default=5, help="num work")
    parser.add_argument('--maxm', type=int, default=147)
    parser.add_argument('--maxu', type=int, default=134)
    parser.add_argument('--path', type=str, default="/data3/yoro/data/DAdata/allorg_8s_notch_2/") # /data3/yoro/data/DGdata/data08s_final_continue/ /data3/yoro/data/DAdata/allorg_8s/
    parser.add_argument('--exp', type=int, default=0, help="file in exp")
    args = parser.parse_args()

    device = torch.device("cuda:{}".format(args.cuda) if torch.cuda.is_available() else "cpu")
    print ('device:', device)

    utils_DOC.set_random_seed(args.trainseed)

    path = args.path
    if os.path.exists(path):
        if "dev" in args.model:
            dataset_map = utils_DOC.load_for_dev(path=path, seed=args.seed,foldn=args.fold_num)
        else:
            dataset_map = utils_DOC.load(path=path, seed=args.seed, exptype=args.exptype, foldn=args.fold_num, maxm=args.maxm, maxu=args.maxu)

    def trainloader_for_other(fold_idx):
        train_X, N_pat, mean_v, std_v = dataset_map[fold_idx]["trainset"], dataset_map[fold_idx]["id"], dataset_map[fold_idx]["mean"], dataset_map[fold_idx]["std"]
        train_loader = torch.utils.data.DataLoader(utils_DOC.DocLoader(train_X, path, mean_v, std_v, args.model),
                    batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        if "Mix" in args.exptype:
            if "DA" in args.model:
                train_loader = torch.utils.data.DataLoader(utils_DOC.DocLoader(train_X, path, mean_v, std_v, args.model, dataset_map[fold_idx]["len_old"], dataset_map[fold_idx]["len_new"]),
                    batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
            else:
                train_loader = torch.utils.data.DataLoader(utils_DOC.DocLoader(train_X, path, mean_v, std_v, args.model),
                    batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        return train_loader, N_pat
    
    def trainloader_for_adv(fold_idx):
        train_X, train_ID, N_pat, mean_v, std_v = dataset_map[fold_idx]["trainset"], dataset_map[fold_idx]["pat"], dataset_map[fold_idx]["id"], dataset_map[fold_idx]["mean"], dataset_map[fold_idx]["std"]
        train_loader = torch.utils.data.DataLoader(utils_DOC.DocIDLoader(train_X, train_ID, path, mean_v, std_v, args.model),
                    batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        return train_loader, N_pat

    def trainloader_for_dev(fold_idx):
        train_X, train_X_aux, N_pat, mean_v, std_v = dataset_map[fold_idx]["trainset"], dataset_map[fold_idx]["trainauxset"], dataset_map[fold_idx]["id"], dataset_map[fold_idx]["mean"], dataset_map[fold_idx]["std"] 
        train_loader = torch.utils.data.DataLoader(utils_DOC.DocDoubleLoader(train_X, train_X_aux, path, mean_v, std_v),
                batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        return train_loader, N_pat
    
    def testloader_for_all(fold_idx):
        test_X, mean_v, std_v = dataset_map[fold_idx]["testset"], dataset_map[fold_idx]["mean"], dataset_map[fold_idx]["std"]
        test_loader = torch.utils.data.DataLoader(utils_DOC.DocLoader(test_X, path, mean_v, std_v, "org" + args.model),
                batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        return test_loader
    
    thistime = time.strftime("%Y%m%d-%H", time.localtime(time.time()))
    if args.exp == 0:
        logpath = "./log/DOC11/{}/".format(args.exptype + args.dataset + '_' + args.model + '_' + "ablation-{}_seed-{}_trainseed-{}_time-{}_cuda-{}".format(args.param, args.seed, args.trainseed, thistime, args.cuda))
        modelpath = "./pre-trained/DOC11/{}".format(args.exptype + args.dataset + '_' + args.model + '_' + "ablation-{}_seed-{}_trainseed-{}_time-{}_cuda-{}".format(args.param, args.seed, args.trainseed, thistime, args.cuda))
    else:
        print("DEBUG in exp file!", "="*24)
        logpath ="./log/DOC/exp/"
        modelpath = "./pre-trained/DOC/exp/"

    if not os.path.exists(logpath):
        os.mkdir(logpath)
        print("DEBUG create log path:", logpath)
    if not os.path.exists(modelpath):
        os.mkdir(modelpath)
        print("DEBUG create model path:", modelpath)

    group_index = thistime + "_" + str(args.trainseed)
    resfile = logpath + group_index + ".csv"
    if not os.path.exists(resfile):
        with open(resfile,'w',encoding='utf-8') as f:
            f_writer = csv.writer(f)
            f_writer.writerow(("group_index", "seed", "trainseed", "foldn", "epochs", "batch_size",
                                "dropout","weight_decay","lr","acc","acc_cmat","recall","f1","best epoch", "p_adv", "p_byol", "p_align", "th", "cover_per", "depth", "nheads"))

    for fold_idx in range(args.fold_num):
        # load model
        if args.model == "base":
            train_loader, N_pat = trainloader_for_other(fold_idx)
            model = DocBase(args.dataset, lr=args.lr, weight_decay=args.wd).to(device)
        elif "DA" in args.model:
            train_loader, N_pat = trainloader_for_other(fold_idx)
            model = DocTmp_DA(device, args.ps, lr=args.lr, weight_decay=args.wd).to(device)
        elif "Tmp" in args.model:
            train_loader, N_pat = trainloader_for_adv(fold_idx)
            model = DocTmp(args.dataset, N_pat, device, args.model, p1=args.p1, p2=args.p2, p3=args.p3, lr=args.lr, weight_decay=args.wd, bz=args.batch_size, th=args.th, clip_value=args.clip_value, exptype=args.exptype+str(fold_idx)).to(device)
        else:
            train_loader, N_pat = trainloader_for_other(fold_idx)
            if "_" in args.model:
                model = eval("{}('{}', device, args, lr=args.lr, weight_decay=args.wd).to(device)".format(args.model.split('_')[0], args.model))           
            else:
                model = eval("{}(lr=args.lr, weight_decay=args.wd).to(device)".format(args.model))        

        test_loader = testloader_for_all(fold_idx)

        model_name = (args.dataset + '_' + args.model + '_' + "foldi-{}_Npat-{}").format(fold_idx, N_pat)
        print("DEBUG model_name:", model_name)

        test_array, val_array = [], []
        test_kappa_array, val_kappa_array = [], []
        test_f1_array, val_f1_array = [], []
        best_result_acc = 0
        result_map = {"acc" : 0,
                      "recall" : 0,
                      "f1" : 0,
                      "confusion_matrix" : 0,
                      "epoch": 0
                    }
        loss_res = {
            "loss": [],
            "loss_adv":[],
            "align_loss":[],
            "byol_loss":[]
        }
        cover_per = []
        for i in range (args.epochs):
            tic = time.time()
            if "DANN" in args.model:
                model.train(train_loader, device, i, args.epochs)
            elif "MulDOCTer" in args.model:
                loss = model.train(train_loader, device)
                loss_res["loss"].append(loss)
            elif "Tmp" in args.model :
                if args.param == 0:
                    loss, loss2, byol_loss, mean_loss, cova_loss, cper = model.train(train_loader, device, i, args.epochs)
                    loss_res["loss"].append(loss)
                    loss_res["loss_adv"].append(loss2*args.p1)
                    loss_res["byol_loss"].append(byol_loss*args.p2)
                    loss_res["align_loss"].append((mean_loss + cova_loss)*args.p3)
                    cover_per.append(cper)
                elif args.param == 1:
                    loss, loss2, byol_loss = model.train_byol(train_loader, device, i, args.epochs)
                    loss_res["loss"].append(loss.item())
                    loss_res["loss_adv"].append(loss2.item()*args.p1)
                    loss_res["byol_loss"].append(byol_loss.item()*args.p2)
                elif args.param == 2:
                    loss, loss2 = model.train_nop(train_loader, device, i, args.epochs)
                    loss_res["loss"].append(loss.item())
                    loss_res["loss_adv"].append(loss2.item()*args.p1)
                elif args.param == 3:
                    loss = model.train_org(train_loader, device, i, args.epochs)
                    loss_res["loss"].append(loss.item())
                elif args.param == 4:
                    loss, mean_loss, cova_loss = model.train_align(train_loader, device, i, args.epochs)
                    loss_res["loss"].append(loss.item())
                    loss_res["align_loss"].append((mean_loss + cova_loss).item()*args.p3)
                elif args.param == 5:
                    loss, mean_loss = model.train_align_mmd(train_loader, device, i, args.epochs)
                    loss_res["loss"].append(loss.item())
                    loss_res["align_loss"].append(mean_loss.item()*args.p3)
                elif args.param == 6:
                    loss, loss2, mean_loss, byol_loss = model.train_mmd(train_loader, device, i, args.epochs)
                    loss_res["loss"].append(loss.item())
                    loss_res["loss_adv"].append(loss2.item()*args.p1)
                    loss_res["byol_loss"].append(byol_loss.item()*args.p2)
                    loss_res["align_loss"].append(mean_loss.item()*args.p3)
                elif args.param == 7:
                    loss, loss2, mean_loss, byol_loss = model.train_cova(train_loader, device, i, args.epochs)
                    loss_res["loss"].append(loss.item())
                    loss_res["loss_adv"].append(loss2.item()*args.p1)
                    loss_res["byol_loss"].append(byol_loss.item()*args.p2)
                    loss_res["align_loss"].append(mean_loss.item()*args.p3)
            else:
                model.train(train_loader, device)
            
            result, gt = model.test(test_loader, device)
            print ('fold:{} {}-th test accuracy: {:.4}, kappa: {:.4}, weighted_f1: {:.4}, time: {}s'.format(
                fold_idx, i, accuracy_score(gt, result), cohen_kappa_score(gt, result), weighted_f1(gt, result), time.time() - tic))
            test_array.append(accuracy_score(gt, result))
            test_kappa_array.append(cohen_kappa_score(gt, result))
            test_f1_array.append(weighted_f1(gt, result))

            with open(logpath + '{}.log'.format(model_name), 'a') as outfile:
                stuffix = ""
                acc, recall, f1, confusion_mat = accuracy_score(gt, result), recall_score(gt, result), f1_score(gt, result), confusion_matrix(gt, result)
                if acc > result_map["acc"]:
                    result_map["acc"], result_map["recall"], result_map["f1"], result_map["confusion_matrix"], result_map["epoch"] = acc, recall, f1, confusion_mat, i
                    stuffix = "=======best result!========"
                    torch.save(model.state_dict(), '{}/{}.pt'.format(modelpath, model_name))

                print ('fold:{} {}-th test accuracy_diy: {:.4}, accuracy: {:.4}, recall: {:.4}, f1: {:.4}, kappa: {:.4}, weighted_f1: {:.4} {}'.format(
                    fold_idx, i, accuracy_score_diy(gt, result), acc, recall, f1, cohen_kappa_score(gt, result),
                    weighted_f1(gt, result), stuffix),
                    file=outfile)
            
        cover_per_count = 0
        if len(cover_per) != 0:
            cover_per_count = np.mean(cover_per)

        np.save(logpath + "{}_test_acc.npy".format(fold_idx), test_array)
        plot_loss(fold_idx, args.epochs, loss_res, logpath)
        with open(resfile, 'a', encoding="utf-8") as f:
            f_writer = csv.writer(f)
            f_writer.writerow((args.model+group_index, args.seed, args.trainseed, fold_idx, args.epochs, args.batch_size, args.dropout, args.wd, args.lr,
                result_map["acc"], result_map["confusion_matrix"], result_map["recall"], result_map["f1"], result_map["epoch"], args.p1, args.p2, args.p3, args.th, cover_per_count, args.depth, args.nheads))

        print ("Fold-{} end! and best acc={:.4f}!*******".format(fold_idx, result_map["acc"]))
        utils_DOC.set_random_seed(args.trainseed)