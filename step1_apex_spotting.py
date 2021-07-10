import torch
import torch.nn.functional as F

import os
import argparse
import glob
import numpy as np
from scipy import signal

from model.model import *
from data.data import *
from utils.utils import *
from calculate import *

def test(testloader, min_loss):
    net.eval()
    net.load_state_dict(torch.load(min_loss))
    all_pred = []
    num = 0 
    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(testloader):
            num += 1
            input_1 = inputs[:, 0].to(device)
            input_2 = inputs[:, 1].to(device)
            input_3 = inputs[:, 2].to(device)
            labels = labels.to(device)
            pred = net(input_1, input_2, input_3)
            all_pred = all_pred + pred[:,0].cpu().numpy().tolist()
    return all_pred

if __name__=="__main__":

    parser = argparse.ArgumentParser(description='')

    parser.add_argument('--dataset',     type=str,  default='SAMM',   help='dataset name,options--SAMM/CAS', metavar='N')
    parser.add_argument('--start',     type=int,  default=0,   help='start subject ID index', metavar='N')
    parser.add_argument('--end',       type=int,  default=8,   help='end subject ID index', metavar='N')
    parser.add_argument('--P',         type=float,default=0.5, 
                                       help='parameter P, CAS:0.55 SAMM:0.5', metavar='N')
    parser.add_argument('--K',         type=int,  default=42,  
                                       help='parameter K, CAS-MAE:12,CAS-ME:31 SAMM-MAE:42, SAMM-ME:227', metavar='N')
    parser.add_argument('--mode',      type=str, default='Micro', 
                                       help='expression type: Micro/Macro',metavar='N')
    parser.add_argument('--pre_data_path', type=str,  default='./output/preprocess_npy/samm_macro', 
                                       help='the path of saving preprocess data', metavar='N')
    parser.add_argument('--xlsx_path', type=str,  default='./input/SAMM.xlsx', 
                                       help='the path of CAS.xlsx/SAMM.xlsx', metavar='N')
    parser.add_argument('--base_weight_path', type=str, default='./weights/1st/samm-micro', 
                                              help='the path of weights',metavar='N')
    parser.add_argument('--save_path', type=str,  default='./output/step1/result_npy', metavar='N')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    K = args.K
    P = args.P
    mode = args.mode

    dataset = args.dataset
    xlsx_path = args.xlsx_path
    pre_data_path = args.pre_data_path
    base_weight_path = args.base_weight_path
    save_path = args.save_path

    all_best_weight = glob.glob(base_weight_path+'/*.pt')

    expression_type = mode
    if mode == 'Micro' and dataset == 'CAS':
        expression_type = 'micro-expression'
    elif mode == 'Macro' and dataset == 'CAS':
        expression_type = 'macro-expression'

    print(mode)

    if dataset == 'CAS':
        label_micro, path_micro = read_xlsx_cas(xlsx_path, expression_type)
        all_subjects = list(set([i.split("/")[-1].split("_")[0] for i in path_micro]))
    elif dataset == 'SAMM':
        label_micro, path_micro = read_xlsx(xlsx_path, expression_type)
        all_subjects = list(set([i.split("_")[0] for i in path_micro]))

    all_subjects.sort()
    all_subjects = all_subjects[args.start:args.end]

    all_TP, all_FP, all_FN = 0,0,0
    all_pred_inter = []
    for one_subject in range(len(all_subjects)):

        net = spot_Net(init_weights=False)
        net.to(device)

        for one_weight in all_best_weight:
            if "net_min_loss_"+all_subjects[one_subject] in one_weight:
                best_weight = one_weight

        subjects_train, subjects_test = [], []
        interval_train, interval_test = [], []
        for i in range(len(path_micro)):
            if dataset=='SAMM':
                if path_micro[i][:3]==all_subjects[one_subject]:
                    subjects_test.append(path_micro[i])
                    interval_test.append(label_micro[i])
                else:
                    subjects_train.append(path_micro[i])
                    interval_train.append(label_micro[i])
            elif dataset=='CAS':
                if path_micro[i].split("/")[-1].split("_")[0]==all_subjects[one_subject]:
                    subjects_test.append(path_micro[i].split("/")[-1][:-1])
                    interval_test.append(label_micro[i])
                else:
                    subjects_train.append(path_micro[i].split("/")[-1][:-1])
                    interval_train.append(label_micro[i])
            else:
                print('not exists dataset', dataset)

        one_subject_TP, one_subject_FP, one_subject_FN = 0,0,0
        for one_video in range(len(subjects_test)):
            testset = spot_data(pre_data_path,
                 [subjects_test[one_video]],
                 dataset=dataset,
                 express=mode)
            testloader = torch.utils.data.DataLoader(testset, 
                                            batch_size=512, 
                                            shuffle=False, 
                                            num_workers=3)

            one_video_pred = test(testloader, best_weight)
            TP, FP, FN, pred_inter= spotting_evaluation(one_video_pred, interval_test[one_video], K, P)
            
            all_pred_inter.append([int(subjects_test[one_video].split("_")[0]+subjects_test[one_video].split("_")[1]), pred_inter])

            one_subject_TP += TP
            one_subject_FP += FP
            one_subject_FN += FN
        print("test one subject result:", "subject ID:", all_subjects[one_subject].split("/")[-1].split("_")[0],
                                          "one_subject_TP:", one_subject_TP,
                                          "one_subject_FP:", one_subject_FP,
                                          "one_subject_FN:", one_subject_FN)
        recall, precision, f1_score = cal_f1_score(one_subject_TP, one_subject_FP, one_subject_FN)
        print("test one subject result:", "subjectID recall, precision, f1_score:", 
                                          all_subjects[one_subject].split("/")[-1].split("_")[0],
                                          recall, precision, f1_score)

        all_TP += one_subject_TP
        all_FP += one_subject_FP
        all_FN += one_subject_FN
    all_pred_inter = np.array(all_pred_inter)

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    save_file_name = 'pred_inter_' + dataset + '_' + mode + '_' + str(K) + '.npy'

    np.save(os.path.join(save_path, save_file_name), all_pred_inter)
    recall, precision, f1_score = cal_f1_score(all_TP, all_FP, all_FN)
    print("test all subjects result: all_TP, all_FP, all_FN", all_TP, all_FP, all_FN)
    print("test all subjects result:", "recall, precision, f1_score:", recall, precision, f1_score)