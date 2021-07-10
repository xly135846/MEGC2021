import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import os
import cv2
import argparse
import glob
import numpy as np

from model.model import *
from data.data import *
from utils.utils import *
from calculate import *

def train(epoch, optimizer, lr, checkpoint):

    if checkpoint == None:
        net.train()
    else:
        net.load_state_dict(torch.load(path+'net_min_loss.pt'))
        net.train()

    num = 0
    loss_list = []

    for batch_idx, (inputs, labels) in enumerate(trainloader):
        num = num+1

        input_1 = inputs[:, 0].to(device)
        input_2 = inputs[:, 1].to(device)
        input_3 = inputs[:, 2].to(device)

        labels = labels.to(device)

        pred = net(input_1, input_2, input_3)

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

    return loss_seq, sum(loss_list)/num


def test(testloader, min_loss):
    net.eval()
    net.load_state_dict(torch.load(min_loss))

    all_pred = []
    loss_list = []
    num = 0 
    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(testloader):
            num += 1
            input_1 = inputs[:, 0].to(device)
            input_2 = inputs[:, 1].to(device)
            input_3 = inputs[:, 2].to(device)

            labels = labels.to(device)
            pred = net(input_1, input_2, input_3)

            loss  = criterion(pred, labels)

            loss_list.append(loss.item())
            all_pred = all_pred + pred[:,0].cpu().numpy().tolist()
    print("test loss:", sum(loss_list)/num)
    return all_pred
    
if __name__=="__main__":

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--start',     type=int,  default=0,   help='start subject ID index', metavar='N')
    parser.add_argument('--end',       type=int,  default=8,   help='end subject ID index', metavar='N')
    parser.add_argument('--P',         type=float,default=0.5, 
                                       help='parameter P, CAS:0.55 SAMM:0.5', metavar='N')
    parser.add_argument('--K',         type=int,  default=42,  
                                       help='parameter K, CAS-MAE:12,CAS-ME:31 SAMM-MAE:42, SAMM-ME:227', metavar='N')
    parser.add_argument('--mode',      type=str, default="macro-expression", 
                                       help='CAS:micro-expression/macro-expression SAMM:Micro/Macro',metavar='N')
    parser.add_argument('--path_data', type=str,  default="./output/preprocess_npy/cas_macro", 
                                       help='the path of saving preprocess data', metavar='N')
    parser.add_argument('--path_npy',  type=str,  default="./output/step1/result_npy/pred_inter_samm_micro.npy", 
                                       help='the path of apex frame spotting', metavar='N')
    parser.add_argument('--path_xlsx', type=str,  default="./input/CAS.xlsx", 
                                       help="the path of CAS.xlsx/SAMM.xlsx", metavar='N')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    epoches = 60
    lr      = 0.001
    K = args.K
    P = args.P
    Mode = args.mode

    softmax   = nn.Softmax(dim=1)
    criterion = nn.MSELoss()

    path = args.path_data
    path_xlsx = args.path_xlsx

    label_micro, path_micro = read_xlsx(path_xlsx+"SAMM.xlsx", Mode)

    all_subjects = list(set([i.split("_")[0] for i in path_micro]))
    all_subjects.sort()
    all_subjects = all_subjects[args.start:args.end]

    all_TP, all_FP, all_FN = 0,0,0
    for one_subject in range(len(all_subjects)):

        MIN_LOSS = 10.
        net = spot_Net(init_weights=False)
        net.to(device)
        optimizer = optim.Adamax(net.parameters(),lr=lr, betas=(0.9,0.99), weight_decay=5e-5)

        subjects_train, subjects_test = [], []
        interval_train, interval_test = [], []
        for i in range(len(path_micro)):
            if path_micro[i][:3]==all_subjects[one_subject]:
                subjects_test.append(path_micro[i])
                interval_test.append(label_micro[i])
            else:
                subjects_train.append(path_micro[i])
                interval_train.append(label_micro[i])

        ## data_loader
        trainset = spot_data(path,
                     subjects_train,
                     data_transforms=None,
                     mode='Train',
                     express=Mode)
        trainloader = torch.utils.data.DataLoader(trainset, 
                                                batch_size=512, 
                                                shuffle=True, 
                                                num_workers=3)

        for epoch in range(epoches):
            losslist, epoch_loss = train(epoch, optimizer, lr, None)
            torch.save(net.state_dict(), './net_min_loss_'+all_subjects[one_subject]+"_"+str(epoch)+"_"+str(epoch_loss)+'.pt')

            one_subject_TP, one_subject_FP, one_subject_FN = 0,0,0
            if epoch_loss<MIN_LOSS:
                MIN_LOSS = epoch_loss

                for one_video in range(len(subjects_test)):
                    testset = spot_data(path,
                         [subjects_test[one_video]],
                         data_transforms=None,
                         mode='Test',
                         express=Mode)
                    testloader = torch.utils.data.DataLoader(testset, 
                                                    batch_size=512, 
                                                    shuffle=False, 
                                                    num_workers=3)

                    one_video_pred = test(testloader, './net_min_loss_'+all_subjects[one_subject]+"_"+str(epoch)+"_"+str(epoch_loss)+'.pt')
                    TP, FP, FN = spotting_evaluation(one_video_pred, interval_test[one_video], K, P)
                    one_subject_TP += TP
                    one_subject_FP += FP
                    one_subject_FN += FN

                print("test one subject result:", "subject_ID", all_subjects[one_subject],
                                                  "one_subject_TP:", one_subject_TP, 
                                                  "one_subject_TP:", one_subject_FP, 
                                                  "one_subject_FN:", one_subject_FN)
                recall, precision, f1_score = cal_f1_score(one_subject_TP, one_subject_FP, one_subject_FN)
                print("test one subject result:", "subject_ID recall, precision, f1_score:", all_subjects[one_subject], recall, precision, f1_score)

        all_TP += one_subject_TP
        all_FP += one_subject_FP
        all_FN += one_subject_FN
    recall, precision, f1_score = cal_f1_score(all_TP, all_FP, all_FN)
    print("test all subjects result:", "recall, precision, f1_score:", recall, precision, f1_score)


