
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

import argparse
import numpy as np

from utils.utils import *
from calculate import *

if __name__=="__main__":

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--path_npy',  type=str, default="./output/step2/result_npy/pred_inter_samm_micro.npy", 
                                       help="the path of saving expression spotting result", metavar='N')
    parser.add_argument('--path_xlsx', type=str, default="./input/CAS.xlsx", 
                                       help="the path of CAS.xlsx/SAMM.xlsx", metavar='N')
    parser.add_argument('--dataset',   type=str, default="SAMM", 
                                       help="SAMM/CAS", metavar='N')
    parser.add_argument('--mode',      type=str, default="macro-expression", 
                                       help='CAS:micro-expression/macro-expression SAMM:micro/macro',metavar='N')
    parser.add_argument('--save_path',  type=str, default="./output/step3", metavar='N')
    args = parser.parse_args()

    mode = args.mode
    path_xlsx = args.path_xlsx
    path_npy = args.path_npy
    dataset = args.dataset
    save_path = args.save_path

    expression_type = mode
    if mode == 'Micro' and dataset == 'CAS':
        expression_type = 'micro-expression'
    elif mode == 'Macro' and dataset == 'CAS':
        expression_type = 'macro-expression'

    print(mode)

    if dataset == 'CAS':
        label_micro, path_micro = read_xlsx_cas(path_xlsx, expression_type)
    elif dataset == 'SAMM':
        label_micro, path_micro = read_xlsx(path_xlsx, expression_type)
    else:
        print("dataset error!")

    all_pred_inter = np.load(path_npy, allow_pickle=True)

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    save_txt = os.path.join(save_path, 'LOG_' + dataset + '_' + mode + '.txt')


    tp,fp,fn=0,0,0
    for i in range(len(all_pred_inter)):
        for interval_2 in label_micro[i]:
            IIOU = []
            for interval_1 in all_pred_inter[i][1]:
                IIOU.append(cal_IOU(interval_1, interval_2))
                if cal_IOU(interval_1, interval_2)>=0.5:
                    tp+=1
                    f = open(save_txt,'a')
                    f.write('\n'+str(path_micro[i])+" "+str(int(interval_2[0]))+" "+\
                            str(int(interval_2[1]))+" "+str(int(interval_1[0]))+" "+str(int(interval_1[1]))+" "+"TP")
                    f.close()
            IIOU = np.array(IIOU)
            temp = len(np.where(IIOU>=0.5)[0])
            if temp<=0:
                fn+=1
                f = open(save_txt,'a')
                f.write('\n'+str(path_micro[i])+" "+str(int(interval_2[0]))+" "+str(int(interval_2[1]))+" "+"__ __"+" "+"FN")
                f.close()
        for interval_1 in all_pred_inter[i][1]:
            IIOU = [] 
            for interval_2 in label_micro[i]:
                IIOU.append(cal_IOU(interval_1, interval_2))
            IIOU = np.array(IIOU)
            temp = len(np.where(IIOU>=0.5)[0])
            if temp<=0:
                fp+=1
                f = open(save_txt,'a')
                f.write('\n'+str(path_micro[i])+" "+"__ __"+" "+str(int(interval_1[0]))+" "+str(int(interval_1[1]))+" "+"FP")
                f.close()