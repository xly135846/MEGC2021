import argparse
import os
import glob
import numpy as np
import cv2

from utils.utils import *

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--start', type=int, default=0, help='start subject ID index', metavar='N')
    parser.add_argument('--end',   type=int, default=8, help='end subject ID index', metavar='N')
    parser.add_argument('--K',         type=int,  default=42,  
                                       help='parameter K, CAS-MAE:6,CAS-ME:18 SAMM-MAE:36, SAMM-ME:174', metavar='N')
    parser.add_argument('--mode',  type=str, default="macro-expression", 
                                   help='CAS:micro-expression/macro-expression SAMM:micro/macro',metavar='N')
    parser.add_argument('--path_xlsx', type=str, default="./input/CAS.xlsx", 
                                       help="the path of CAS.xlsx/SAMM.xlsx", metavar='N')
    parser.add_argument('--path_save', type=str, default="./output/preprocess_npy", 
                                       help="the path of saving the data after preprocess", metavar='N')
    args = parser.parse_args()

    save_path = args.path_save
    path = args.path_xlsx
    K= args.K
    Mode = args.mode

    label_micro, path_micro = read_xlsx_cas(path+"test.xlsx", Mode)
    print("Macro num:", len(label_micro), len(path_micro))

    path_micro = path_micro[args.start:args.end]
    label_micro = label_micro[args.start:args.end]

    optical_flow = cv2.optflow.DualTVL1OpticalFlow_create()

    for file in range(len(path_micro)):

        if os.path.exists(save_path+"/"+str(path_micro[file].split("/")[-1][:-1])+"_"+Mode+".npy"):
            print(path_micro[file]+"_macro.npy", "is exists")
        else:
            all_jpgs = glob.glob(path+"longVideoFaceCropped/"+path_micro[file]+"/*.jpg")
            all_jpgs.sort(key=lambda x: int(x.split("/")[-1].split("_")[-1][:-4]))
            print(path_micro[file], label_micro[file], len(all_jpgs), all_jpgs[:10])

            fd, fll = read_json(all_jpgs[0].replace("longVideoFaceCropped", "longVideoFaceCropped")+"_0.json")
            for iii in range(len(fll)):
                # CAS_size=400*400
                # SAMM_size=600*600
                fll[iii] = int(fll[iii]/400*128)
            left_eye, right_eye, x31, y32, x33, y34, x41, y42, x43, y44, x51, y52, x53, y54, x61, y61 = landmark_rechange(fll)

            process_img = []
            for i in tqdm(range(len(all_jpgs)-K)):
                
                count = 0
                for label in label_micro[file]:
                    if cal_IOU([i,i+K], label)>0:
                        count = count + 1
                if count!=0:
                    label = 1
                else:
                    label = 0

                img_path_1 = all_jpgs[i]
                img_path_2 = all_jpgs[i+K]
                img1 = img_crop_resize(img_path_1, fd, (128,128))
                img2 = img_crop_resize(img_path_2, fd, (128,128))

                flow = optical_flow.calc(img1, img2, None)

                magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1]) 
                u, v = pol2cart(magnitude, angle)
                o_s = computeStrain(u, v)

                final = np.zeros((128, 128, 3))
                final[:,:,0] = u
                final[:,:,1] = v
                final[:,:,2] = o_s

                final[:, :, 0] = abs(final[:, :, 0] - final[y61-5:y61+6, x61-5:x61+6, 0].mean())
                final[:, :, 1] = abs(final[:, :, 1] - final[y61-5:y61+6, x61-5:x61+6, 1].mean())
                final[:, :, 2] = final[:, :, 2] - final[y61-5:y61+6, x61-5:x61+6, 2].mean()
                
                cv2.fillPoly(final, [np.array(left_eye)], 0)
                cv2.fillPoly(final, [np.array(right_eye)], 0)
                
                final_image = np.zeros((42, 42, 3))
                final_image[:21, :, :] = cv2.resize(final[min(y32, y42) : max(y34, y44), x31:x43, :], (42, 21))
                final_image[21:42, :, :] = cv2.resize(final[y52:y54, x51:x53, :], (42, 21))
                process_img.append([final_image, label])

            process_img = np.array(process_img)
            np.save(save_path+"/"+str(path_micro[file].split("/")[-1][:-1])+"_"+Mode+".npy", process_img)