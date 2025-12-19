import argparse
import base64
import json
import os
import os.path as osp

import imgviz
import PIL.Image
import numpy

from labelme.logger import logger
from labelme import utils


def main():
    logger.warning(
        "This script is aimed to demonstrate how to convert the "
        "JSON file to a single image dataset."
    )
    logger.warning(
        "It won't handle multiple JSON files to generate a "
        "real-use dataset."
    )

    # json_file是标注完之后生成的json文件的目录。out_dir是输出目录，即数据处理完之后文件保存的路径
    json_file = r"./datasets/before_newBPT"
    out_dir1 = r"./datasets/SegmentationClass"

    # 如果输出的路径不存在，则自动创建这个路径
    if not osp.exists(out_dir1):
        os.mkdir(out_dir1)
        # 将类别名称转换成数值，以便于计算
    # label_name_to_value = {"_background_": 0, "0": 1}
    label_name_to_value = {"_background_": 0, "0": 1, "1": 2, "3": 3}
    for file_name in os.listdir(json_file):
        # 遍历json_file里面所有的文件，并判断这个文件是不是以.json结尾
        if file_name.endswith(".json"):
            path = os.path.join(json_file, file_name)
            if os.path.isfile(path):
                data = json.load(open(path, encoding="utf-8"))

                # 获取json里面的图片数据，也就是二进制数据
                imageData = data.get("imageData")
                # 如果通过data.get获取到的数据为空，就重新读取图片数据
                if not imageData:
                    imagePath = os.path.join(json_file, data["imagePath"])
                    with open(imagePath, "rb") as f:
                        imageData = f.read()
                        imageData = base64.b64encode(imageData).decode("utf-8")
                #  将二进制数据转变成numpy格式的数据
                img = utils.img_b64_to_arr(imageData)

                for shape in sorted(data["shapes"], key=lambda x: x["label"]):
                    label_name = shape["label"]
                    if label_name in label_name_to_value:
                        label_value = label_name_to_value[label_name]
                    else:
                        label_value = len(label_name_to_value)
                        label_name_to_value[label_name] = label_value
                lbl, _ = utils.shapes_to_label(img.shape, data["shapes"], label_name_to_value)

                label_names = [None] * (max(label_name_to_value.values()) + 1)
                for name, value in label_name_to_value.items():
                    label_names[value] = name

                lbl_viz = imgviz.label2rgb(
                    label=lbl, image=imgviz.asgray(img), label_names=label_names, loc="rb"
                )

                logger.info("Saved to: {}".format(out_dir1))
    print("label:", label_name_to_value)


if __name__ == "__main__":
    main()
