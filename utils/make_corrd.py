import argparse
import glob
import multiprocessing
import os
import random
import sys
import warnings
from multiprocessing.pool import Pool
from random import shuffle
import numpy as np
import cv2
import pandas as pd
from PIL import Image
from tqdm.std import tqdm

sys.path.append(os.path.join(os.path.abspath(__file__), "/../../"))

parser = argparse.ArgumentParser(description="Get center points of patches "
                                             "from mask")

parser.add_argument("--mask_path_df_path", default='/public/project/home/chenhh/code/prostate/bin_2024年7月30日/exps/data/prostate-cancer-grade-assessment/train_mask_paths.csv', type=str)
# parser.add_argument("--coords_path", default="/public/project/home/chenhh/code/prostate/bin_2024年7月30日/exps/data/prostate-cancer-grade-assessment/train_size256_step1_level0_label0_data_coords_df.csv", type=str)
parser.add_argument("--coords_path", default="/public/project/home/chenhh/code/prostate/bin_2024年7月30日/exps/data/prostate-cancer-grade-assessment/train_size256_step1_level0_label1_data_coords_df.csv", type=str)
# parser.add_argument("--coords_path", default="/public/project/home/chenhh/code/prostate/bin_2024年7月30日/exps/data/prostate-cancer-grade-assessment/train_size256_step1_level0_label2_data_coords_df.csv", type=str)
parser.add_argument('--num_process', default=32, type=int,
                    help='number of mutli-process, default 8')
parser.add_argument("--patch_size", default=256//16, type=int)
parser.add_argument("--patch_step", default=256//16, type=int)
parser.add_argument("--level", default=0, type=int)
# parser.add_argument("--columns", default="mask_path_0", type=str)
parser.add_argument("--columns", default="mask_path_1", type=str)
# parser.add_argument("--columns", default="mask_path_2", type=str)
parser.add_argument('--debug', default=False, type=bool, help='')
parser.add_argument("--seed", default=0, type=int)
def getDirContent(dirPath,fileType,isJoin):
    files_name=[]
    files_name_temp = os.listdir(dirPath)
    if fileType=='file':
        for file_name_temp in files_name_temp:
            if os.path.isfile((os.path.join(dirPath, file_name_temp))):
                files_name.append(file_name_temp)
    elif fileType=='dir':
        for file_name_temp in files_name_temp:
            if not os.path.isfile((os.path.join(dirPath, file_name_temp))):
                files_name.append(file_name_temp)

def list_shuffle_split(ls,ratio):
    lss=[]
    while len(ls)!=0:
        shuffle(ls)
        lt=int(ratio[0]*len(ls)+0.5)
        lss.append(ls[0:lt])
        ls=ls[lt+1:]
        ratio=[v/sum(ratio[1:]) for i,v in enumerate(ratio) if i!=0]
    return lss

def process(opts):
    try:
        id, wsi_name,mask_path, patch_size, patch_step=opts


        mask=cv2.imread(mask_path)
        padding = [(mask.shape[0] % patch_step) // 2, (mask.shape[1] % patch_step) // 2]
        # 1白色背景、3肿瘤（Tumor）、2正常（Other）、0未标记；
        # 优先级有小到大0,1,2,3,0
        centers=[]
        for row in range(0, mask.shape[0]+padding[0], patch_step):
            for col in range(0, mask.shape[1]+padding[1], patch_step):
                patch_mask=mask[row:row+patch_size,col:col+patch_size,0]


                if patch_mask.__contains__(1) \
                        or patch_mask.__contains__(2) \
                        or patch_mask.__contains__(3) \
                        or patch_mask.__contains__(4) \
                        or patch_mask.__contains__(5) \
                        or patch_mask.__contains__(6):
                    tissue_ratio={
                        (patch_mask == 0).sum():0,
                        (patch_mask == 1).sum():1, # +:0 ,isup_grade:0
                        (patch_mask == 2).sum():2, # ,isup_grade:1
                        (patch_mask == 3).sum():3, # ,isup_grade:2
                        (patch_mask == 4).sum():4, # +:3 ,isup_grade:3
                        (patch_mask == 5).sum():5, # +:4 ,isup_grade:4
                        (patch_mask == 6).sum():6, # +:5 ,isup_grade:5
                        (patch_mask == 255).sum():255,
                    }
                    k=max(list(tissue_ratio.keys()))
                    # if (k!=0) and (k!=1):
                    #     if tissue_ratio[k]!=0:
                    centers.append([
                        wsi_name,
                        tissue_ratio[k],
                        args.level,
                        (col)*16,
                        (row)*16,
                        patch_size*16,
                        patch_size*16])

        column_names = ["wsi_name", "label",'level',"start_w","start_h","size_w","size_h"]
        data_coords_df=pd.DataFrame(centers,columns=column_names)
        if len(data_coords_df)==0:
            print(wsi_name)
        return data_coords_df
    except Exception as e:
        warnings.warn(str(id) + ": " + str(mask_path))
        print(e)
        column_names = ["wsi_name", "mask_path", "center_w", "center_h"]
        data_coords_df = pd.DataFrame([], columns=column_names)
        return data_coords_df
def run(args):
    patch_size=args.patch_size
    patch_step=args.patch_step
    mask_path_df=pd.read_csv(args.mask_path_df_path,index_col=0)

    mask_path_df_0=mask_path_df.loc[:,["wsi_name",args.columns]].rename(columns={args.columns: 'mask_path'})
    mask_path_df_0=mask_path_df_0.loc[mask_path_df_0["mask_path"].notnull(),:]
    opts_list = []
    for id, (wsi_name,mask_path) in enumerate(mask_path_df_0.values):
        opts_list.append((id, wsi_name,mask_path, patch_size, patch_step))
    data_coords_dfs=[]
    if (args.debug):
        for (id, wsi_name,mask_path, patch_size, patch_step) in tqdm(opts_list):
            data_coords_dfs.append(process((id, wsi_name,mask_path, patch_size, patch_step)))
    else:
        with multiprocessing.Pool(processes=args.num_process)as pool:
            with tqdm(total=len(opts_list)) as tbar:
                for ret in pool.imap(process, opts_list):
                    data_coords_dfs.append(ret)
                    tbar.update()
    data_coords_df=pd.concat(data_coords_dfs,axis=0).reset_index(drop=True)
    print(data_coords_df)
    data_coords_df.to_csv(args.coords_path)

if __name__=="__main__":
    args = parser.parse_args()
    random.seed(args.seed)
    print(args)
    run(args)