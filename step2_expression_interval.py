import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import argparse
import glob
import numpy as np
from scipy import signal

from model.model_merge import *
from data.data import *
from utils.utils import *
from calculate import *

def test_one_sample(test_sample, min_loss):
    net.eval()
    net.load_state_dict(torch.load(min_loss))

    all_pred = []
    loss_list = []
    with torch.no_grad():
        test_sample = torch.FloatTensor(test_sample)
        input_1 = test_sample.to(device)
        pred, conv_out = net(input_1)
        pred_cls = F.softmax(pred)
        _, max_index = torch.max(pred_cls, dim=1)

    return pred_cls, max_index

if __name__=="__main__":

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--dataset',     type=str,  default='SAMM',   help='dataset name,options--SAMM/CAS', metavar='N')
    parser.add_argument('--start', type=int,   default=0, help='start subject ID index', metavar='N')
    parser.add_argument('--end',   type=int,   default=8, help='end subject ID index', metavar='N')
    parser.add_argument('--P',         type=float,default=0.5, 
                                       help='parameter P, CAS:0.55 SAMM:0.5', metavar='N')
    parser.add_argument('--K',         type=int,  default=42,  
                                       help='parameter K, CAS-MAE:12,CAS-ME:31 SAMM-MAE:42, SAMM-ME:227', metavar='N')
    parser.add_argument('--L',         type=int,  default=21,  
                                       help='parameter L, samm-micro:21, samm-macro:114, cas-micro:6, cas-macro:16', metavar='N')
    parser.add_argument('--pre_data_path', type=str, default="./output/preprocess_npy/samm_micro_merge", 
                                       help='the path of saving preprocess data', metavar='N')
    parser.add_argument('--step1_npy_path',  type=str, default="./output/step1/result_npy/pred_inter_SAMM_Micro_42.npy", 
                                       help='the path of apex frame spotting', metavar='N')
    parser.add_argument('--xlsx_path', type=str, default="./input/SAMM.xlsx", 
                                       help="the path of CAS.xlsx/SAMM.xlsx", metavar='N')
    parser.add_argument('--save_path', type=str, default="./output/step2/result_npy", metavar='N')
    parser.add_argument('--base_weight_path', type=str, default="./weights/2nd/samm-micro", 
                                              help="the path of weights",metavar='N')
    parser.add_argument('--mode', type=str, default='Micro', 
                                  help='CAS:micro-expression/macro-expression SAMM:Micro/Macro',metavar='N')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    K = args.K
    P = args.P
    L = args.L
    BATCH_SIZE = 128
    mode = args.mode

    softmax   = nn.Softmax(dim=1)

    dataset = args.dataset
    path_npy  = args.step1_npy_path
    path_xlsx = args.xlsx_path
    save_path = args.save_path
    pre_path = args.pre_data_path

    path_base_weight = args.base_weight_path
    all_best_weight = glob.glob(path_base_weight+'/*.pt')
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

    all_subjects = list(set([i.split("_")[0] for i in path_micro]))
    all_subjects.sort()
    all_subjects = all_subjects[args.start:args.end]

    all_pred_inter = np.load(path_npy, allow_pickle=True)
    all_pred_inter_static = {}
    for i in range(len(all_pred_inter)):
        all_pred_inter_static[all_pred_inter[i][0]] = all_pred_inter[i][1]

    all_TP, all_FP, all_FN = 0,0,0
    all_pred_inter_merge = []

    for one_subject in range(len(all_subjects)):
        net = merge_Net(init_weights=False, batch_size=BATCH_SIZE, dataset=dataset, mode=mode)
        net.to(device)

        subjects_train, subjects_test = [], []
        interval_train, interval_test = [], []
        for i in range(len(path_micro)):
            if path_micro[i].split('_')[0]==all_subjects[one_subject]:
                subjects_test.append(path_micro[i])
                interval_test.append(label_micro[i])
            else:
                subjects_train.append(path_micro[i])
                interval_train.append(label_micro[i])

        for one_weight in all_best_weight:
            if "net_min_loss_"+all_subjects[one_subject].split("/")[-1].split('_')[0] in one_weight:
                best_weight=one_weight

        one_subject_TP, one_subject_FP, one_subject_FN = 0,0,0
        one_subject_TP_before, one_subject_FP_before, one_subject_FN_before = 0,0,0
        for one_video in range(len(subjects_test)):
            if dataset == 'SAMM':
                temp = int(subjects_test[one_video].split('_')[0]+subjects_test[one_video].split('_')[1])
                temp_npy = np.load(pre_path+'/'+subjects_test[one_video]+'_' + mode + '.npy', allow_pickle=True)
            elif dataset == 'CAS':
                temp = int(subjects_test[one_video].split('/')[1][:-1])
                temp_npy = np.load(pre_path+'/'+subjects_test[one_video].split('/')[1][:-1]+'_'+mode+'.npy', allow_pickle=True)
            else:
                print('not exits dataset', dataset)

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
                one_sample_pred, max_index = test_one_sample(final_batch_size, best_weight)
                all_pred.append(max_index.cpu().numpy().tolist()[0])

            merge_pred, megre_interval_test = merge(all_pred, all_pred_inter_static[temp], 1, K)
            all_pred_inter_merge.append([all_subjects[one_subject].split("/")[-1].split("_")[0], megre_interval_test])
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
    all_pred_inter_merge = np.array(all_pred_inter_merge)

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    save_file_name = 'pred_inter_merge_' + dataset + '_' + mode + '_' + str(K) + '.npy'

    np.save(os.path.join(save_path, save_file_name), all_pred_inter_merge)
    recall, precision, f1_score = cal_f1_score(all_TP, all_FP, all_FN)
    print("test all subjects result:", "recall, precision, f1_score:", recall, precision, f1_score)