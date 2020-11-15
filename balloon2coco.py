# 将Detectron 2的balloon数据集格式转化为MS COCO的数据集格式
import json
import time

import numpy as np

UNIQUE_ID = 0


def getUniqueId():
    global UNIQUE_ID
    UNIQUE_ID = UNIQUE_ID + 1
    return UNIQUE_ID


def COCOFile(info, licenses, images, annotations, categories):
    return {
        "info": info,
        "licenses": licenses,
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }


def COCOStructure():
    return COCOInfo(), [], [], [], COCOCategories()


def COCOInfo():
    return {
        "description": "CUSTOM COCO 2017 Dataset",
        "year": 2017,
        "contributor": "https://github.com/PPeanutButter",
        "date_created": time.time().__str__()
    }


def COCOImage(img_unique_id, file_name, width, height):
    return {
        "id": img_unique_id,
        "file_name": file_name,
        "width": width,
        "height": height
    }


def COCOAnnotations(annotation_id, image_id, category_id, x_points, y_points):
    segmentation = []
    for x, y in zip(x_points, y_points):
        segmentation.append(float(x))
        segmentation.append(float(y))
    return {
        "segmentation": [segmentation],
        "id": annotation_id,
        "image_id": image_id,
        "category_id": category_id,
        "area": int(vector_product(segmentation)),
        "iscrowd": 0,
        "bbox": [
            np.min(x_points).__int__(),
            np.min(y_points).__int__(),
            np.abs(np.max(x_points) - np.min(x_points)).__int__(),
            np.abs(np.max(y_points) - np.min(y_points)).__int__()
        ]
    }


def COCOCategories():
    return {
        "supercategory": "paper",
        "id": 1,
        "name": "paper"
    }


# 基于向量积计算不规则四边形的面积
def vector_product(coord):
    coord = np.array(coord).reshape((4, 2))
    temp_det = 0
    for idx_vp in range(3):
        temp = np.array([coord[idx_vp], coord[idx_vp + 1]])
        temp_det += np.linalg.det(temp)
    temp_det += np.linalg.det(np.array([coord[-1], coord[0]]))
    return temp_det * 0.5


if __name__ == "__main__":
    for dataset_name in ["train", "val"]:
        MY_SET_JSON_INDEX_PATH = f"H:/raw/{dataset_name}_set_man.json"
        DES_COCO_SET_JSON_INDEX_PATH = f"H:/raw/coco_{dataset_name}_2017.json"
        with open(MY_SET_JSON_INDEX_PATH, "r", encoding="utf-8") as f:
            DICT = json.load(f)
            info, licenses, images, annotations, categories = COCOStructure()
            for idx, v in enumerate(DICT.values()):
                # 读取文件名
                filename = v["filename"]
                # 读取大小
                width, height = v["WH"]
                image_id = getUniqueId()
                images.append(COCOImage(image_id, filename, width, height))
                # 读取masks
                for _, rect in v["regions"].items():
                    a = rect["shape_attributes"]
                    px = a["all_points_x"]
                    py = a["all_points_y"]
                    annotation_id = getUniqueId()
                    annotations.append(COCOAnnotations(annotation_id, image_id, 1, px, py))
                # 测试用,只看2个
                # if idx / 2 == 1:
                #     break
            with open(DES_COCO_SET_JSON_INDEX_PATH, 'w', encoding='utf-8') as w:
                content = COCOFile(info, licenses, images, annotations, categories)
                w.write(json.dumps(content))
        print(f"Convert {dataset_name} Done")
