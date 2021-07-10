import numpy as np
from scipy import signal

from utils.utils import *

def cal_TP(left_count_1, label):
    result = []
    for inter_2 in label:
        temp = 0
        for inter_1 in left_count_1:
            if cal_IOU(inter_1, inter_2)>=0.5:
                temp += 1
        result.append(temp)
    return result

def spotting_evaluation(pred, express_inter, K, P):
    pred = np.array(pred)
    threshold = np.mean(pred)+ P*(np.max(pred)-np.mean(pred))
    num_peak = signal.find_peaks(pred, height=threshold, distance=K*2)
    pred_inter = []
    
    for peak in num_peak[0]:
        pred_inter.append([peak-K, peak+K])

    result = cal_TP(pred_inter, express_inter)
    result = np.array(result)
    TP = len(np.where(result!=0)[0])
    n = len(pred_inter)-(sum(result)-TP)
    m = len(express_inter)
    FP = n-TP
    FN = m-TP

    return TP, FP, FN, pred_inter

def spotting_evaluation_V2(pred_inter, express_inter):
    result = cal_TP(pred_inter, express_inter)
    result = np.array(result)
    TP = len(np.where(result!=0)[0])
    n = len(pred_inter)-(sum(result)-TP)
    m = len(express_inter)
    FP = n-TP
    FN = m-TP

    return TP, FP, FN

def cal_f1_score(TP, FP, FN):
    recall = TP/(TP+FP)
    precision = TP/(TP+FN)
    f1_score = 2*recall*precision/(recall+precision)
    return recall, precision, f1_score

def merge(alist, blist, pred_value, K):
    alist_str = ""
    for i in alist:
        alist_str +=str(i)
    split_str = str(1-pred_value)
    num = max([len(i) for i in alist_str.split(split_str)])-1
    for i in range(num):
        i=0
        while i<(len(alist)-1):
            if (alist[i]==pred_value and alist[i+1]==pred_value) and abs(blist[i][1]-blist[i+1][0])<=K*2:
                clist = alist[:i]+[pred_value]+alist[i+2:]
                dlist = blist[:i]+[[blist[i][0],blist[i+1][1]]]+blist[i+2:]
                alist, blist = clist, dlist
            i+=1
    return alist,blist