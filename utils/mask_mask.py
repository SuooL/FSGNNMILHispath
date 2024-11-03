import datetime
import multiprocessing
import os
import sys
import logging
import argparse

import numpy as np
import openslide
import cv2
import json
import pandas as pd
from PIL import Image
from openslide.deepzoom import DeepZoomGenerator
from skimage import measure
from skimage.color import rgb2hsv
from skimage.filters import threshold_otsu

from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../')
parser = argparse.ArgumentParser(description='Get tumor mask of tumor-WSI and '
                                             'save it in npy format')
parser.add_argument('--data_path', default="/public/project/home/chenhh/code/prostate/bin_2024年7月30日/exps/data/prostate-cancer-grade-assessment/train_data_paths.csv", type=str, help='')
parser.add_argument('--mask_path', default="/public/project/home/chenhh/code/prostate/bin_2024年7月30日/exps/data/prostate-cancer-grade-assessment/train_mask_paths.csv", type=str, help='')

parser.add_argument('--mask_dir', default="/public/project/home/chenhh/code/prostate/bin_2024年7月30日/exps/data/prostate-cancer-grade-assessment/mask_png",metavar='PNG_PATH', type=str,help='Path to the output npy mask file')
parser.add_argument('--image_dir', default="/public/project/home/chenhh/code/prostate/bin_2024年7月30日/exps/data/prostate-cancer-grade-assessment/image_png",metavar='PNG_PATH', type=str,help='Path to the output npy mask file')
parser.add_argument('--tissueImage_dir', default="/public/project/home/chenhh/code/prostate/bin_2024年7月30日/exps/data/prostate-cancer-grade-assessment/tissueImage_png",metavar='PNG_PATH', type=str,help='Path to the output npy mask file')
parser.add_argument('--tissueMask_dir', default="/public/project/home/chenhh/code/prostate/bin_2024年7月30日/exps/data/prostate-cancer-grade-assessment/tissueMask_png",metavar='PNG_PATH', type=str,help='Path to the output npy mask file')
parser.add_argument('--cover', default=True, type=bool, help='')

parser.add_argument('--RGB_min', default=0, type=int, help='')
parser.add_argument('--RGB_max', default=256, type=int, help='')
parser.add_argument('--area_min', default=40, type=int, help='')

parser.add_argument('--erode_kernelSize', default=(3, 3), type=tuple, help='')
parser.add_argument('--dilate_kernelSize', default=(15, 15), type=tuple, help='')
parser.add_argument('--debug', default=False, type=bool, help='')
parser.add_argument('--num_process', default=36, type=int,help='number of mutli-process, default 8')
parser.add_argument('--bg_level', default=-5, type=int, help='')# 54321
parser.add_argument('--ratio', default=16, type=int, help='')# 54321

def get_rgb_mask(img_RGB, RGB_min,RGB_max):
    img_HSV = rgb2hsv(img_RGB)
    S = img_HSV[:, :, 1] < threshold_otsu(img_HSV[:, :, 1])
    background_R = img_RGB[:, :, 0] > threshold_otsu(img_RGB[:, :, 0])
    background_G = img_RGB[:, :, 1] > threshold_otsu(img_RGB[:, :, 1])
    background_B = img_RGB[:, :, 2] > threshold_otsu(img_RGB[:, :, 2])
    RGB = background_R & background_G & background_B
    # RGB =background_G
    min_R = img_RGB[:, :, 0] > RGB_min
    min_G = img_RGB[:, :, 1] > RGB_min
    min_B = img_RGB[:, :, 2] > RGB_min
    max_R = img_RGB[:, :, 0] < RGB_max
    max_G = img_RGB[:, :, 1] < RGB_max
    max_B = img_RGB[:, :, 2] < RGB_max
    mask = S & RGB & min_R & min_G & min_B & max_R & max_G & max_B
    # mask = S& min_R & min_G & min_B& max_R & max_G
    mask = mask.astype(np.uint8)
    return mask

def erode(images,kernelSize=(3,3)):
    kernel=cv2.getStructuringElement(cv2.MORPH_RECT,kernelSize)
    erodeds = [cv2.erode(image,kernel) for image in images]
    return erodeds

def dilate(images,kernelSize=(3,3)):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,kernelSize)
    dilateds = [cv2.dilate(image, kernel) for image in images]
    return dilateds
def bg_mask_task(image,RGB_min,RGB_max,area_min):
    bg_mask = get_rgb_mask(image, RGB_min=RGB_min,RGB_max=RGB_max)
    # [bg_mask] = erode([bg_mask], kernelSize=(2, 2))
    # 连通域
    def ltyyouhua(mask):
        label_mask = measure.label(mask, connectivity=2)
        regions = measure.regionprops(label_mask)
        labels = set()
        for region in regions:
            if region.area < area_min:
                labels.add(region.label)
        for label in labels:
            mask[label_mask == label] = 0
        return mask
    bg_mask=ltyyouhua(bg_mask)|(1-ltyyouhua(1-bg_mask))
    return 1-bg_mask

def get_tissue(dzSlide,size, level,RGB_min,RGB_max,area_min):
    img_RGB = np.array(dzSlide.get_tile(dzSlide.level_count - 1 + level, (0, 0)).convert('RGB'))
    # img_RGB[(img_RGB[:,:,0]>=150)&(img_RGB[:,:,1]>=150)&(img_RGB[:,:,2]>=150),:]=132

    tissue_mask = bg_mask_task(img_RGB,RGB_min,RGB_max,area_min)
    tissue_mask = cv2.resize(tissue_mask.astype(np.uint8), (size[0], size[1]), interpolation=cv2.INTER_NEAREST)
    return img_RGB,tissue_mask




def fillPolygon(mask, chunk, value, ratio):
    original_grasp_bboxes = np.array(np.array(chunk, dtype=np.float32) * ratio, np.int32)
    original_grasp_mask = cv2.fillPoly(mask, [original_grasp_bboxes], value)
    return original_grasp_mask


def colormap(mask, color_list):
    mask=Image.fromarray(mask.astype(np.uint8),mode='P')
    mask.putpalette(np.array(color_list,dtype=np.uint8))
    return mask



def vectorgraphAnalysis(vectorgraph,endUpMaskSize, maskSize, ratio=1):
    # 遍历所有对象，统计组织类型获得tissueTypes
    tissueTypes = []
    if "childObjects" in vectorgraph.keys():
        objectQueue = vectorgraph["childObjects"]
        for parentObject in objectQueue:
            objectQueue.extend(parentObject["childObjects"])

            tissueTypes.append(parentObject["pathObject"]["properties"]["classification"]["name"])

    else:
        objectQueue = vectorgraph["features"]
        for parentObject in objectQueue:
            if "objectType" in parentObject["properties"].keys():
                if parentObject["properties"]["objectType"] == "cell":
                    if "classification" not in parentObject["properties"].keys():
                        none_colorRGB = (0, 255, 0)
                        parentObject["properties"]["classification"] = {
                            'name': "Negative",
                            "colorRGB": int(
                                none_colorRGB[0] * 0x10000 + none_colorRGB[1] * 0x100 + none_colorRGB[2]) - 0x1000000
                        }
                if parentObject["properties"]["objectType"] == "annotation":
                    if "classification" not in parentObject["properties"].keys():
                        none_colorRGB = (245, 245, 245)
                        parentObject["properties"]["classification"] = {
                            'name': "Area",
                            "colorRGB": int(
                                none_colorRGB[0] * 0x10000 + none_colorRGB[1] * 0x100 + none_colorRGB[2]) - 0x1000000
                        }
            if "classification" in  parentObject["properties"].keys():
                tissueTypes.append(parentObject["properties"]["classification"]["name"])
    tissueTypes = list(set(tissueTypes))

    print("tissueTypes",tissueTypes)
    # 初始化masksInfo=[]，把所有{mask, geometry}准备好
    masksInfo = {}
    for tissueType in tissueTypes:
        masksInfo[tissueType] = {
            'mask': np.zeros((int(maskSize[1] * ratio + 0.5), int(maskSize[0] * ratio + 0.5)), dtype=np.uint8),
            'geometrys': []
        }
    # 把矢量图中的坐标分析提取到geometrys，
    # 这份代码检查不到无标签切坡，需要升级到cell形状才能准确获取阴性细胞的区域，不过影响不大，参考cell_profiler.py文件
    if "childObjects" in vectorgraph.keys():
        objectQueue = vectorgraph["childObjects"]
        for object in objectQueue:
            objectQueue.extend(object["childObjects"])
            try:
                tissueType = object["pathObject"]["properties"]["classification"]["name"]
            except:
                continue
            geometry = object["pathObject"]["geometry"]
            masksInfo[tissueType]['geometrys'].append(geometry)
    else:
        objectQueue = vectorgraph["features"]
        for parentObject in objectQueue:
            try:
                tissueType = parentObject["properties"]["classification"]["name"]
            except:
                continue
            geometry = parentObject["geometry"]
            masksInfo[tissueType]['geometrys'].append(geometry)

    # 画mask
    for tissueType, maskInfo in masksInfo.items():
        for geometry in maskInfo['geometrys']:
            # 圆形策略
            if geometry['type'] == 'Ellipse':
                object = geometry['coordinates']
                if len(object) > 0:
                    masksInfo[tissueType]['mask'] = fillPolygon(masksInfo[tissueType]['mask'], chunk=object[0],
                                                                value=1, ratio=ratio)
                    for outline in object[1:]:
                        masksInfo[tissueType]['mask'] = fillPolygon(masksInfo[tissueType]['mask'], chunk=outline,
                                                                    value=0, ratio=ratio)
            # 矩形策略
            if geometry['type'] == 'Rectangel':
                object = geometry['coordinates']
                if len(object) > 0:
                    masksInfo[tissueType]['mask'] = fillPolygon(masksInfo[tissueType]['mask'], chunk=object[0],
                                                                value=1, ratio=ratio)
                    for outline in object[1:]:
                        masksInfo[tissueType]['mask'] = fillPolygon(masksInfo[tissueType]['mask'], chunk=outline,
                                                                    value=0, ratio=ratio)
            # 多边形策略
            if geometry['type'] == 'Polygon':
                object = geometry['coordinates']
                if len(object) > 0:
                    masksInfo[tissueType]['mask'] = fillPolygon(masksInfo[tissueType]['mask'], chunk=object[0],
                                                                value=1, ratio=ratio)
                    for outline in object[1:]:
                        masksInfo[tissueType]['mask'] = fillPolygon(masksInfo[tissueType]['mask'], chunk=outline,
                                                                    value=0, ratio=ratio)
            # 多个多边形策略
            if geometry['type'] == 'MultiPolygon':
                objects = geometry['coordinates']
                for object in objects:
                    if len(object) > 0:
                        masksInfo[tissueType]['mask'] = fillPolygon(masksInfo[tissueType]['mask'], chunk=object[0],
                                                                    value=1, ratio=ratio)
                        for outline in object[1:]:
                            masksInfo[tissueType]['mask'] = fillPolygon(masksInfo[tissueType]['mask'], chunk=outline,
                                                                        value=0, ratio=ratio)
        masksInfo[tissueType]['mask']=cv2.resize(masksInfo[tissueType]['mask'], (endUpMaskSize[0], endUpMaskSize[1]), interpolation=cv2.INTER_NEAREST)
    return masksInfo

def getCKMask(path, size):
    img_RGB = np.array(Image.open(path))[:, :, :3]
    ck_mask = cv2.cvtColor(255 - img_RGB, cv2.COLOR_BGR2GRAY) // 255
    ck_mask = cv2.resize(ck_mask, size)
    return ck_mask


def process(opts):
    index, wsi_name, wsi_path, sketch_path, type,other, args = opts
    other=json.loads(other)
    args.sketch_path = sketch_path
    # if (not str(args.sketch_path) == 'nan') and (not args.sketch_path is None):
    #     mask_path = os.path.join(args.mask_dir, os.path.splitext(wsi_name)[0] + '.png')
    #     return [wsi_name,mask_path]
    args.wsi_path = wsi_path
    tissueImage_path = os.path.join(args.tissueImage_dir, "{}.png".format(wsi_name))
    tissueMask_path = os.path.join(args.tissueMask_dir, "{}.png".format(wsi_name))
    mask_path_0 = os.path.join(args.mask_dir, "{}_{}.png".format(wsi_name,str(0)))
    mask_path_1 = os.path.join(args.mask_dir, "{}_{}.png".format(wsi_name,str(1)))
    mask_path_2 = os.path.join(args.mask_dir, "{}_{}.png".format(wsi_name,str(2)))
    if not args.cover:
        if os.path.exists(mask_path_0) and os.path.exists(mask_path_1) and os.path.exists(mask_path_2):
            return [wsi_name,mask_path_0,mask_path_1,mask_path_2]
    try:
        slide = openslide.open_slide(args.wsi_path)
        dzSlide = DeepZoomGenerator(slide, tile_size=max(slide.level_dimensions[0]), overlap=0)
        max_size_w, max_size_h = dzSlide.level_dimensions[-1]
        w, h = max_size_w // args.ratio, max_size_h // args.ratio
        # 优先级由小到大

        priority = [1,2,3,4,5,6,255,0]
        label_dict = {
            0: {
                'name': "bg",
                'name_list': [],
                'colorRGB': (255,255,255)
            },
            1: {
                'name': "0",
                'name_list': ["0"],
                'colorRGB': (199, 237, 204)
            },
            2: {
                'name': "1",
                'name_list': ["1"],
                'colorRGB': (250, 249, 222)
            },
            3: {
                'name': "2",
                'name_list': ["2"],
                'colorRGB': (255, 242, 226)
            },
            4: {
                'name': "3",
                'name_list': ["3"],
                'colorRGB': (253, 230, 224)
            },
            5: {
                'name': "4",
                'name_list': ["4"],
                'colorRGB': (227, 237, 205)
            },
            6: {
                'name': "5",
                'name_list': ["5"],
                'colorRGB': (220, 226, 241)
            },
            255: {
                'name': "未标记",
                'name_list': ["Area","Tumor"],
                'colorRGB': (0, 0, 0)
            },
        }
        # 0白色背景、2肿瘤（Tumor）、1正常（Other）、3未标记；
        # 优先级有小到大3,1,2,0
        tissueImage,tissueMask = get_tissue(dzSlide, (w, h),args.bg_level, args.RGB_min, args.RGB_max,args.area_min)

        # [masksInfo[tissueType]['mask']]=erode([masksInfo[tissueType]['mask']],kernelSize=(6,6))
        # # 膨胀一下组织区域
        # [tissueMask] = erode([tissueMask], kernelSize=args.erode_kernelSize)
        # [tissueMask] = dilate([tissueMask], kernelSize=args.dilate_kernelSize)

        masks = np.ones((3,h, w), dtype=np.uint8)*255
        if (not str(args.sketch_path) == 'nan') and (not args.sketch_path is None) :
            if "json" in args.sketch_path:
                with open(args.sketch_path) as f:
                    vectorgraph = json.load(f)
                masksInfo = vectorgraphAnalysis(vectorgraph,(w, h), (max_size_w, max_size_h), 1)
                for label in priority:
                    if label == 0:
                        masks[tissueMask == 0] = 0
                    else:
                        for key in masksInfo.keys():
                            if (key in label_dict[label]['name_list']):
                                if label==255:
                                    masks[masksInfo[key]["mask"] == 0] = label
                                else:
                                    masks[masksInfo[key]["mask"] == 1] = label
            # if "MASK" in args.sketch_path:
            #     mask = mask * 0
            #     MASK = np.array(Image.open(args.sketch_path).resize((w, h), Image.NEAREST))
            #     mask[MASK == 255] = 2
        else:
            masks[0,tissueMask == 1] = int(other["isup_grade"][0])+1
            masks[1,tissueMask == 1] = int(other["gleason_score"][1])+1
            masks[2,tissueMask == 1] = int(other["gleason_score"][1])+1
            masks[:,tissueMask == 0] = 0
        Image.fromarray(tissueMask*255).save(tissueMask_path)
        Image.fromarray(tissueImage).save(tissueImage_path)
        for i in range(masks.shape[0]):
            mask = cv2.resize(masks[i], (w, h), interpolation=cv2.INTER_NEAREST)
            mask_path = os.path.join(args.mask_dir, "{}_{}.png".format(wsi_name,str(i)))
            Image.fromarray(mask).save(mask_path)

            color_list=[]
            for k,v in label_dict.items():
                color_list.extend(v['colorRGB'])
            image=colormap(mask, color_list)
            image_path = os.path.join(args.image_dir, "{}_{}.png".format(wsi_name,str(i)))
            image.save(image_path)
            current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(index,current_time, os.path.abspath(mask_path))
        return [wsi_name,mask_path_0,mask_path_1,mask_path_2]
    except Exception as e:
        print(str(index) + ": " + str(wsi_path) + ";" + str(sketch_path))
        print(e)
        return [wsi_name,None]


def run(args):
    print(args)
    os.makedirs(args.mask_dir, exist_ok=True)
    os.makedirs(args.image_dir, exist_ok=True)
    os.makedirs(args.tissueImage_dir, exist_ok=True)
    os.makedirs(args.tissueMask_dir, exist_ok=True)
    data_paths_df = pd.read_csv(args.data_path,index_col=0)

    opts_list = []
    for index, (wsi_name, wsi_path, sketch_path, type,other) in enumerate(data_paths_df.values):
        opts_list.append((index, wsi_name, wsi_path, sketch_path, type,other, args))
    mask_paths = []
    if (args.debug):
        for (index, wsi_name, wsi_path, sketch_path, type,other, args) in opts_list:
            mask_paths.append(process((index, wsi_name, wsi_path, sketch_path, type,other, args)))
            if index>3:
                break
    else:
        with multiprocessing.Pool(processes=args.num_process)as pool:
            with tqdm(total=len(opts_list)) as tbar:
                for ret in pool.imap(process, opts_list):
                    mask_paths.append(ret)
                    tbar.update()
    column_names=["wsi_name","mask_path_0","mask_path_1","mask_path_2"]
    mask_path_df = pd.DataFrame(mask_paths, columns=column_names)
    mask_path_df.to_csv(args.mask_path)


def main():
    logging.basicConfig(level=logging.INFO)

    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()