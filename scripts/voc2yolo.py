import os
import xml.etree.ElementTree as ET
from shutil import copyfile

# -----------------------------
# 配置区域
# -----------------------------
VOC_DIR = "multi-hand-gesture-dataset"  # 原数据集目录
YOLO_DIR = "gesture_dataset"            # 输出 YOLO 数据集目录

# 类别映射（顺序固定）
classes = [
    "hello",
    "hand up",
    "vectory",
    "i love you",
    "yes",
    "no",
    "ok",
    "okay",
    "call me",
    "thaks"
]

# -----------------------------
# 功能函数
# -----------------------------
def convert_bbox(size, box):
    """VOC bbox -> YOLO bbox"""
    dw = 1.0 / size[0]
    dh = 1.0 / size[1]
    xmin, xmax, ymin, ymax = box
    x = (xmin + xmax) / 2.0 * dw
    y = (ymin + ymax) / 2.0 * dh
    w = (xmax - xmin) * dw
    h = (ymax - ymin) * dh
    return x, y, w, h

def voc_to_yolo(xml_file, yolo_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    with open(yolo_file, 'w') as out_file:
        for obj in root.iter('object'):
            cls_name = obj.find('name').text
            if cls_name not in classes:
                continue
            cls_id = classes.index(cls_name)
            xmlbox = obj.find('bndbox')
            bbox = (
                int(xmlbox.find('xmin').text),
                int(xmlbox.find('xmax').text),
                int(xmlbox.find('ymin').text),
                int(xmlbox.find('ymax').text)
            )
            x, y, w_box, h_box = convert_bbox((w, h), bbox)
            out_file.write(f"{cls_id} {x:.6f} {y:.6f} {w_box:.6f} {h_box:.6f}\n")

# -----------------------------
# 主程序
# -----------------------------
def process_split(split):
    """
    split = "train" 或 "test"
    """
    img_in_dir = os.path.join("/home/lfk/robot_dog/data/multi-hand-gesture-dataset-main", split)
    img_out_dir = os.path.join("/home/lfk/robot_dog/data/dataset1", "images", "train" if split=="train" else "val")
    label_out_dir = os.path.join("/home/lfk/robot_dog/data/dataset1", "labels", "train" if split=="train" else "val")
    os.makedirs(img_out_dir, exist_ok=True)
    os.makedirs(label_out_dir, exist_ok=True)

    for file in os.listdir(img_in_dir):
        if file.endswith(".jpg") or file.endswith(".png"):
            name = os.path.splitext(file)[0]
            img_path = os.path.join(img_in_dir, file)
            xml_path = os.path.join(img_in_dir, name + ".xml")
            yolo_path = os.path.join(label_out_dir, name + ".txt")

            if not os.path.exists(xml_path):
                continue

            # 转换标签
            voc_to_yolo(xml_path, yolo_path)
            # 拷贝图片
            copyfile(img_path, os.path.join(img_out_dir, file))

# -----------------------------
# 执行
# -----------------------------
if __name__ == "__main__":
    process_split("train")
    process_split("test")
    print("VOC -> YOLO 转换完成！")
