import glob
import os
import random
from sklearn.model_selection._split import KFold
import pandas as pd

def get_split_idx(samples, ratio):
    random_index = list(range(len(samples)))
    random.shuffle(random_index)
    return random_index[:int(len(random_index) * ratio)], random_index[int(len(random_index) * ratio):]
def get_k_fold_idx(samples, fold=3,val_ratio=0.2, test_ratio=0.2, random_state=0):
    if fold==1:
        indexs, test_index = get_split_idx(samples, 1 - test_ratio)
        train_index, val_index = get_split_idx(indexs, 1 - val_ratio)
        all_indexs = []
        all_indexs.append([
            [indexs[i] for i in train_index],
            [indexs[i] for i in val_index],
            test_index
        ])
        return all_indexs
    else:
        indexs, test_index = get_split_idx(samples, 1 - test_ratio)
        # 交叉验证数据
        kf = KFold(n_splits=fold, shuffle=True, random_state=random_state)
        all_indexs = []
        for train_index, valid_index in kf.split(indexs):
            all_indexs.append([
                [indexs[i] for i in train_index],
                [indexs[i] for i in valid_index],
                test_index
            ])
        return all_indexs
def get_train_fold(data_paths_df,type_names,fold=3,random_state=0):
    train_namess = []
    val_namess = []
    test_namess = []

    for type_name in list(type_names):
        samples = data_paths_df.loc[data_paths_df["type"] == type_name, "wsi_name"].values.tolist()
        _train_namess = []
        _val_namess = []
        _test_namess = []
        for train_indexs, val_indexs, test_indexs in get_k_fold_idx(samples, fold=fold, test_ratio=0,
                                                                    random_state=random_state):
            _train_namess.append([samples[index] for index in train_indexs])
            _val_namess.append([samples[index] for index in val_indexs])
            _test_namess.append([samples[index] for index in test_indexs])
        train_namess.append(_train_namess)
        val_namess.append(_val_namess)
        test_namess.append(_test_namess)

    val_namess_=[]
    for fold_i in range(fold):
        val_names_=[]
        for type_i in range(len(type_names)):
            val_names_.extend(val_namess[type_i][fold_i])
        val_namess_.append(val_names_)
    val_namess=val_namess_

    # 折叠
    namess = val_namess
    max_len = max([len(names) for names in namess])
    new_namess = [[None] * max_len for i in range(len(namess))]
    for i, (names, new_names) in enumerate(zip(namess, new_namess)):
        for j, name in enumerate(names):
            new_namess[i][j] = name

    # 表格
    column_names = ["wsi_name","fold"]
    rows = []
    for names in zip(*new_namess):
        for fold_i,name in enumerate(names):
            if name is None:
                continue
            rows.append([name,fold_i])

    data_index_df = pd.DataFrame(rows, columns=column_names)
    data_index_df = data_index_df.sort_values(by='wsi_name').reset_index(drop=True)
    return data_index_df
if __name__=="__main__":
    data_paths_df_path="/public/project/home/chenhh/code/prostate/bin_2024年7月30日/exps/data/prostate-cancer-grade-assessment/train_data_paths.csv"
    root=os.path.dirname(data_paths_df_path)

    # 生成交叉验证和验证集索引
    data_paths_df=pd.read_csv(data_paths_df_path)
    type_names=list(set(data_paths_df["type"].tolist()))
    data_index_df = get_train_fold(data_paths_df, type_names, fold=5, random_state=0)
    data_index_df_path = os.path.join(root, "train_fold5.csv")
    data_index_df.to_csv(data_index_df_path)