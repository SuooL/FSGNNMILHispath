import glob
import os
import random
from sklearn.model_selection._split import KFold
import pandas as pd
from tqdm.std import tqdm
import json
def get_prostate_B1(data_paths):
    type_name = "prostate-cancer-grade-assessment"
    error_names = [
    ]

    slice_dir = '/public/datastore/病理/前列腺/train_images/*.tiff'
    slice_paths=glob.glob(slice_dir)
    df_path="/public/datastore/病理/前列腺/train.csv"
    df=pd.read_csv(df_path)

    count_all = 0
    count_slide = 0
    data_paths={}
    for row in tqdm(df.iterrows(),total=len(df)):
        image_id=row[1]["image_id"]
        data_provider=row[1]["data_provider"]
        isup_grade=str(row[1]["isup_grade"])
        gleason_score=row[1]["gleason_score"]
        if gleason_score=="negative":
            gleason_score="0+0"
        gleason_score=[
            gleason_score.split("+")[0],
            gleason_score.split("+")[1],
        ]
        for slice_path in slice_paths:
            if image_id in slice_path:
                data_paths[image_id]={
                    "wsi_path": slice_path,
                    "sketch_path": None,
                    "type": type_name,
                    "other":json.dumps({
                        "data_provider":data_provider,
                        "isup_grade":isup_grade,
                        "gleason_score":gleason_score,
                    })
                }
                count_slide += 1
                break
        count_all+=1
    print(type_name,"：count_mask：", count_all, "count_slide：", count_slide)
    return data_paths,error_names, type_name


def get_train_data_index():
    data_paths = {}
    error_names=[]
    type_names=[]
    # 加载get_train_yxbx_npc_tumor
    data_paths, error_names_, type_names_ = get_prostate_B1(data_paths)
    error_names.extend(error_names_)
    type_names.append(type_names_)

    # 使用表格存储
    column_names = ["wsi_name", "wsi_path", "sketch_path", "type","other"]
    rows = []
    for name in data_paths.keys():
        if name not in error_names:
            rows.append([
                name,
                data_paths[name]["wsi_path"],
                data_paths[name]["sketch_path"],
                data_paths[name]["type"],
                data_paths[name]["other"],
            ])
            # rows.append([name]+[data_paths[name][key] for key in data_paths[name].keys()])
    data_paths_df = pd.DataFrame(rows, columns=column_names)
    type_names = list(set(type_names))
    data_paths_df = data_paths_df.sort_values(by='wsi_name').reset_index(drop=True)
    return data_paths_df, type_names

if __name__=="__main__":
    root="/public/project/home/chenhh/code/prostate/bin_2024年7月30日/exps/data/prostate-cancer-grade-assessment"
    os.makedirs(root, exist_ok=True)
    # 生成路径数据
    train_data_paths_df,train_type_names=get_train_data_index()
    train_data_paths_df_path = os.path.join(root, "train_data_paths.csv")
    train_data_paths_df.to_csv(train_data_paths_df_path)
    print("train_type_names: ", train_type_names)