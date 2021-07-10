import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import os
import cv2
import argparse
import glob
import numpy as np
from scipy import signal

from model.model_merge import *
from data.data_merge import *
from utils.utils import *
from calculate import *

def merge(alist, blist, pred_value):
    alist_str = ""
    for i in alist:
        alist_str +=str(i)
    split_str = str(1-pred_value)
    num = max([len(i) for i in alist_str.split(split_str)])-1
    for i in range(num):
        i=0
        while i<(len(alist)-1):
            if (alist[i]==pred_value and alist[i+1]==pred_value) and abs(blist[i][1]-blist[i+1][0])<=31*2:
                clist = alist[:i]+[pred_value]+alist[i+2:]
                dlist = blist[:i]+[[blist[i][0],blist[i+1][1]]]+blist[i+2:]
                alist, blist = clist, dlist
            i+=1
    return alist,blist

def train(epoch, optimizer, lr, checkpoint):

    if checkpoint == None:
        net.train()
    else:
        net.load_state_dict(torch.load(path+'net_min_loss.pt'))
        net.train()

    num = 0
    loss_list = []
    all_label = []

    for batch_idx, (inputs, labels) in enumerate(trainloader):
        num = num+1
        all_label += labels.numpy()[:, 0].tolist()

        input = inputs.to(device)

        labels = labels[:, 0].to(device)

        pred,_ = net(input)

        loss  = criterion(pred, labels)

        loss_seq = [loss]
        grad_seq = [torch.ones_like(loss).to(device) for _ in range(len(loss_seq))]

        optimizer.zero_grad()
        torch.autograd.backward(loss_seq, grad_seq)

        loss_list.append(loss_seq[0].item())
        optimizer.step()

        if batch_idx%100==0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f} '.format(
                    epoch, batch_idx * len(inputs), len(trainloader.dataset),
                    100. * batch_idx / len(trainloader), 
                    loss.item(),
                    ))

    return loss_seq, sum(loss_list)/num, all_label

def test(testloader, min_loss):
    net.eval()
    net.load_state_dict(torch.load(min_loss))

    all_pred = []
    loss_list = []
    num = 0 
    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(testloader):
            num += 1
            input = inputs.to(device)

            labels = labels[:, 0].to(device)
            pred, conv_out = net(input)

            loss  = criterion(pred, labels)
            pred_cls = F.softmax(pred)

            loss_list.append(loss.item())
            temp = pred_cls[:,0].cpu().numpy().tolist()
            temp = [int(i*1000)/1000 for i in temp]
            all_pred = all_pred + temp
    try:
        print("test loss:", sum(loss_list)/num)
    except:
        print("test loss:", sum(loss_list), num)

    return all_pred

def test_one_sample(test_sample, min_loss):
    net.eval()
    net.load_state_dict(torch.load(min_loss))

    all_pred = []
    loss_list = []
    with torch.no_grad():
        test_sample = torch.FloatTensor(test_sample)
        input = test_sample.to(device)
        pred, conv_out = net(input)
        pred_cls = F.softmax(pred)
        _, max_index = torch.max(pred_cls, dim=1)

    return pred_cls, max_index

if __name__=="__main__":

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--start',     type=int,  default=0,   help='start subject ID index', metavar='N')
    parser.add_argument('--end',       type=int,  default=8,   help='end subject ID index', metavar='N')
    parser.add_argument('--P',         type=float,default=0.5, 
                                       help='parameter P, CAS:0.55 SAMM:0.5', metavar='N')
    parser.add_argument('--K',         type=int,  default=42,  
                                       help='parameter K, CAS-MAE:12,CAS-ME:31 SAMM-MAE:42, SAMM-ME:227', metavar='N')
    parser.add_argument('--L',         type=int,  default=21,  
                                       help='parameter L, samm-micro:21, samm-macro:114, cas-micro:6, cas-macro:16', metavar='N')
    parser.add_argument('--mode',      type=str, default="macro-expression", 
                                       help='CAS:micro-expression/macro-expression SAMM:Micro/Macro',metavar='N')
    parser.add_argument('--path_data', type=str,  default="./output/preprocess_npy/cas_macro", 
                                       help='the path of saving preprocess data', metavar='N')
    parser.add_argument('--path_xlsx', type=str,  default="./input/CAS.xlsx", 
                                       help="the path of CAS.xlsx/SAMM.xlsx", metavar='N')
    parser.add_argument('--dataset',   type=str, default="SAMM", 
                                       help="SAMM/CAS", metavar='N')
    parser.add_argument('--path_inter',  type=str, default="./output/step1/result_npy/pred_inter_42.npy", 
                                       help='the path of apex frame spotting', metavar='N')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    K = args.K
    L = args.L
    P = args.P
    Mode = args.mode
    print("learning rate {}".format(lr))

    softmax   = nn.Softmax(dim=1)
    criterion = torch.nn.CrossEntropyLoss()

    if args.dataset == "SAMM":
        path = args.patb_data
        path_xlsx = args.path_xlsx
        label_micro, path_micro = read_xlsx(path_xlsx+"SAMM.xlsx", Mode)
        all_subjects = list(set([i.split("_")[0] for i in path_micro]))
    elif args.dataset == "CAS":
        path =  args.patb_data
        path_xlsx = args.path_xlsx
        label_micro, path_micro = read_xlsx_cas(path_xlsx+"test.xlsx", Mode)
        all_subjects = list(set([i.split("/")[-1].split("_")[0] for i in path_micro]))
    else:
        print("dataset error!")

    all_subjects.sort()
    all_subjects = all_subjects[args.start:args.end]

    BATCH_SIZE = 128

    path_inter = args.path_inter
    all_pred_inter = np.load(path_inter, allow_pickle=True)
    all_pred_inter_static = {}
    for i in range(len(all_pred_inter)):
        all_pred_inter_static[all_pred_inter[i][0]] = all_pred_inter[i][1]

    all_TP, all_FP, all_FN = 0,0,0
    for one_subject in range(len(all_subjects)):

        net = merge_Net(init_weights=False, batch_size=BATCH_SIZE)
        net.to(device)

        optimizer = optim.Adamax(net.parameters(),lr=lr, betas=(0.9,0.99), weight_decay=5e-5)
        MIN_LOSS = 10.
        MAX_ACC  = 1.

        if args.dataset == "SAMM":
            print("current test subject:", all_subjects[one_subject])
        elif args.dataset == "CAS":
            print("current test subject:", all_subjects[one_subject].split("/")[-1].split("_")[0])
        else:
            print("dataset error!")

        subjects_train, subjects_test = [], []
        interval_train, interval_test = [], []
        for i in range(len(path_micro)):
            if path_micro[i][:3]==all_subjects[one_subject]:
                subjects_test.append(path_micro[i])
                interval_test.append(label_micro[i])
            else:
                subjects_train.append(path_micro[i])
                interval_train.append(label_micro[i])

        # data_loader
        trainset = merge_data(path,
                     subjects_train,
                     labels=interval_train,
                     length=L,
                     batch_size=BATCH_SIZE,
                     data_transforms=None,
                     mode='Train',
                     express="micro")
        trainloader = torch.utils.data.DataLoader(trainset, 
                                                batch_size=BATCH_SIZE,
                                                shuffle=True,
                                                num_workers=3)

        for epoch in range(epoches):
            losslist, epoch_loss, all_label = train(epoch, optimizer, lr, None)
            all_label = np.array(all_label)
            torch.save(net.state_dict(), './net_min_loss_'+all_subjects[one_subject].split("/")[-1].split("_")[0]\
                                         +str(epoch)+"_"+str(epoch_loss)+'.pt')

            one_subject_TP, one_subject_FP, one_subject_FN = 0,0,0
            one_subject_TP_before, one_subject_FP_before, one_subject_FN_before = 0,0,0

            for one_video in range(len(subjects_test)):
                temp = int(subjects_test[one_video].split("_")[0]+subjects_test[one_video].split("_")[1])
                ### 测试一个样本
                temp_npy = np.load(path+"/"+subjects_test[one_video]+"_micro.npy", allow_pickle=True)
                all_pred = []
                for i in all_pred_inter_static[temp]:

                    IIOU = 0 
                    for ii in range(len(interval_test[one_video])):
                        IIOU += cal_IOU(i, interval_test[one_video][ii])

                    if i[0]<0:
                        inter = [0,L]
                    else:
                        inter = [int(i[0]/4), int(i[1]/4)]

                    img_list = temp_npy[inter[0]:inter[1]]
                    final = np.zeros((3*L, 42, 42))
                    for j in range(len(img_list)):
                        final[j*3:(j+1)*3] = np.transpose(img_list[j], (2,0,1))

                    for final_index in range(len(final)):
                        final[final_index] = normal_arr(final[final_index])

                    final_batch_size = np.zeros((BATCH_SIZE,3*L, 42, 42))
                    for p in range(len(final_batch_size)):
                        final_batch_size[p] = final
                    one_sample_pred, max_index = test_one_sample(final_batch_size, './net_min_loss_'+all_subjects[one_subject].split("/")[-1].split("_")[0]\
                                                                                   +str(epoch)+"_"+str(epoch_loss)+'.pt')
                    all_pred.append(max_index.cpu().numpy().tolist()[0])

                ##合并区间
                merge_pred, megre_interval_test = merge(all_pred, all_pred_inter_static[temp], 1, K)
                TP, FP, FN = spotting_evaluation_V2(megre_interval_test, interval_test[one_video])
                TP_before, FP_before, FN_before = spotting_evaluation_V2(all_pred_inter_static[temp], interval_test[one_video])

                one_subject_TP += TP
                one_subject_FP += FP
                one_subject_FN += FN

                one_subject_TP_before += TP_before
                one_subject_FP_before += FP_before
                one_subject_FN_before += FN_before

            print("test one subject result:", "subject ID:", all_subjects[one_subject].split("/")[-1].split("_")[0],
                                              "one_subject_TP:", one_subject_TP, 
                                              "one_subject_FP:", one_subject_FP, 
                                              "one_subject_FN:", one_subject_FN)
            print("test one subject result:", "subject ID:", all_subjects[one_subject].split("/")[-1].split("_")[0],
                                              "one_subject_TP_before:", one_subject_TP_before, 
                                              "one_subject_FP_before:", one_subject_FP_before, 
                                              "one_subject_FN_before:", one_subject_FN_before)
            recall, precision, f1_score = cal_f1_score(one_subject_TP, one_subject_FP, one_subject_FN)
            recall_before, precision_before, f1_score_before = cal_f1_score(one_subject_TP_before, one_subject_FP_before, one_subject_FN_before)
            print("test one subject result:", "subjectID recall, precision, f1_score:", 
                                              all_subjects[one_subject].split("/")[-1].split("_")[0],
                                              recall, precision, f1_score)
            print("test one subject result:", "subjectID recall_before, precision_before, f1_score_before:", 
                                              all_subjects[one_subject].split("/")[-1].split("_")[0],
                                              recall_before, precision_before, f1_score_before)
        all_TP += one_subject_TP
        all_FP += one_subject_FP
        all_FN += one_subject_FN
    recall, precision, f1_score = cal_f1_score(all_TP, all_FP, all_FN)
    print("test all subjects result:", "recall, precision, f1_score:", recall, precision, f1_score)