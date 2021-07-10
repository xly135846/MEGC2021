import json
import pandas as pd
import numpy as np
import cv2

def read_xlsx(path, expression_type):
    sheet = pd.read_excel(path)
    all_label = []
    all_path = []
    for i in sheet.values[:]:
        if expression_type in i[7]:
            if i[3]!=0:
                all_label.append([i[3], i[5]])
            else:
                all_label.append([i[4], i[5]])
            all_path.append(i[1].split("_")[0]+"_"+i[1].split("_")[1])
    path_set = list(set(all_path))
    path_set.sort()
    all_path = np.array(all_path)
    all_label = np.array(all_label)
    path = []
    label = []
    for one_path in path_set:
        path.append(one_path)
        label.append(all_label[np.where(all_path==one_path)])
    return label, path

def read_xlsx_cas(path, expression_type):
    sheet = pd.read_excel(path)
    name_rule_2 = pd.read_excel(path, 
                                sheet_name="naming rule2")
    name_rule_1 = pd.read_excel(path, 
                                sheet_name="naming rule1")

    name_1 = {}
    for i in name_rule_1.values:
        name_1[i[2]] = i[0]
    name_2 = {}
    name_2["disgust1"] = 101
    for i in name_rule_2.values:
        name_2[i[1]] = i[0]
        
    path_list = []
    label_interval = []
    aaa = "anger1"
    for i in range(len(sheet.values)-2):
        express = sheet.values[i][7]
        if expression_type==express:
            a = name_2[sheet.values[i][1].split("_")[0]]
            b = name_1[sheet.values[i][0]]
            tmp = sheet.values[i][1].split("_")[0]
            if tmp==aaa:
                if sheet.values[i][4]!=0:
                    label_interval.append([sheet.values[i][2], sheet.values[i][4]])
                else:
                    label_interval.append([sheet.values[i][2], sheet.values[i][3]])
            else:
                aaa = tmp
                label_interval.append(",")
                if sheet.values[i][4]!=0:
                    label_interval.append([sheet.values[i][2], sheet.values[i][4]])
                else:
                    label_interval.append([sheet.values[i][2], sheet.values[i][3]])
            path_list.append("s"+str(b)+"/"+str(b)+"_0"+str(a)+"*")
    label_interval.append(",")

    label = []
    tmp = []
    for i in label_interval:
        if i !=",":
            tmp.append(i)
        else:
            label.append(tmp)
            tmp = []
    label = np.array(label)
    path = []
    tmp = "###"
    for i in path_list:
        if i!=tmp:
            path.append(i)
            tmp = i
    if expression_type=="micro-expression":
        return label[1:],path
    elif expression_type=="micro-expression":
        return label,path
    else:
        return label,path

def cal_IOU(interval_1, interval_2):
    intersection = [max(interval_1[0], interval_2[0]), min(interval_1[1], interval_2[1])]
    union_set    = [min(interval_1[0], interval_2[0]), max(interval_1[1], interval_2[1])]
    if intersection[0]<=intersection[1]:
        len_inter = intersection[1]-intersection[0]+1
        len_union = union_set[1]-union_set[0]+1
        return len_inter/len_union
    else:
        return 0

def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return (x, y)

def computeStrain(u, v):
    u_x= u - pd.DataFrame(u).shift(-1, axis=1)
    v_y= v - pd.DataFrame(v).shift(-1, axis=0)
    u_y= u - pd.DataFrame(u).shift(-1, axis=0)
    v_x= v - pd.DataFrame(v).shift(-1, axis=1)
    o_s = np.array(np.sqrt(u_x**2 + v_y**2 + 1/2 * (u_y+v_x)**2).ffill(1).ffill(0))
    return o_s

def landmark_rechange(fll):
    x11=max(fll[2*36] - 15, 0)
    y11=fll[2*26+1] 
    x12=fll[2*37]
    y12=max(fll[2*37+1] - 15, 0)
    x13=fll[2*38] 
    y13=max(fll[2*38+1] - 15, 0)
    x14=min(fll[2*39] + 15, 128)
    y14=fll[2*39+1] 
    x15=fll[2*40] 
    y15=min(fll[2*40+1] + 15, 128)
    x16=fll[2*41] 
    y16=min(fll[2*41+1] + 15, 128)
    
    #Right Eye
    x21=max(fll[2*42] - 15, 0)
    y21=fll[2*42+1]
    x22=fll[2*43] 
    y22=max(fll[2*43+1] - 15, 0)
    x23=fll[2*44] 
    y23=max(fll[2*44+1] - 15, 0)
    x24=min(fll[2*45] + 15, 128)
    y24=fll[2*45+1] 
    x25=fll[2*46] 
    y25=min(fll[2*46+1] + 15, 128)
    x26=fll[2*47] 
    y26=min(fll[2*47+1] + 15, 128)
    
    #ROI 1 (Left Eyebrow)
    x31=max(fll[2*17]- 12, 0)
    y32=max(fll[2*19+1] - 12, 0)
    x33=min(fll[2*21] + 12, 128)
    y34=min(fll[2*41+1] + 12, 128)
    
    #ROI 2 (Right Eyebrow)
    x41=max(fll[2*22] - 12, 0)
    y42=max(fll[2*24+1] - 12, 0)
    x43=min(fll[2*26] + 12, 128)
    y44=min(fll[2*46+1] + 12, 128)
    
    #ROI 3 #Mouth
    x51=max(fll[2*60] - 12, 0)
    y52=max(fll[2*50+1] - 12, 0)
    x53=min(fll[2*64] + 12, 128)
    y54=min(fll[2*57+1] + 12, 128)
    
    #Nose landmark
    x61=fll[2*28]
    y61=fll[2*28+1]

    left_eye = [(x11, y11), (x12, y12), (x13, y13), (x14, y14), (x15, y15), (x16, y16)]
    right_eye = [(x21, y21), (x22, y22), (x23, y23), (x24, y24), (x25, y25), (x26, y26)]

    return left_eye, right_eye, x31, y32, x33, y34, x41, y42, x43, y44, x51, y52, x53, y54, x61, y61

def read_json(json_path):
    with open(json_path) as f:
        jpg_info = json.loads(f.read())
        fd = jpg_info['fd_info']["face_rect"]
        fll = jpg_info['fll_info']["landmarks"]
    return fd, fll

def img_crop_resize(img_path, fd, shape=(128,128)):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    try:
        img = img.crop((int(fd[0]), int(fd[1]), int(fd[2]), int(fd[3])))
    except:
        img = img[int(fd[1]):int(fd[3]), int(fd[0]):int(fd[2])]
    img = cv2.resize(img, (128, 128))
    return img

def normal_arr(arr):
    result = (arr-np.min(arr))/(np.max(arr)- np.min(arr))
    return result