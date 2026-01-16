#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import shutil
import random

# ----------------------------
# 配置区域
# ----------------------------
src_root = "data/archive/leapGestRecog"   # 原始数据集路径
dst_root = "data/dataset2"                # 整理后的目标路径
train_ratio = 0.8                         # 训练集比例
random_seed = 42                          # 随机种子

# ----------------------------
# 脚本开始
# ----------------------------
random.seed(random_seed)

train_dir = os.path.join(dst_root, "train")
val_dir = os.path.join(dst_root, "val")
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

# 遍历第一层 00-09
for seq_dir_name in sorted(os.listdir(src_root)):
    seq_dir_path = os.path.join(src_root, seq_dir_name)
    if not os.path.isdir(seq_dir_path):
        continue
    
    # 遍历第二层手势文件夹
    for class_dir_name in sorted(os.listdir(seq_dir_path)):
        class_dir_path = os.path.join(seq_dir_path, class_dir_name)
        if not os.path.isdir(class_dir_path):
            continue
        
        # 用第二层文件夹名作为类别名
        class_name = class_dir_name
        train_class_dir = os.path.join(train_dir, class_name)
        val_class_dir = os.path.join(val_dir, class_name)
        os.makedirs(train_class_dir, exist_ok=True)
        os.makedirs(val_class_dir, exist_ok=True)
        
        # 收集所有图片
        img_files = [os.path.join(class_dir_path, f) for f in os.listdir(class_dir_path)
                     if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        
        # 打乱顺序
        random.shuffle(img_files)
        
        # 划分训练/验证
        n_train = int(len(img_files) * train_ratio)
        train_imgs = img_files[:n_train]
        val_imgs = img_files[n_train:]
        
        # 复制到目标目录
        for src_path in train_imgs:
            dst_path = os.path.join(train_class_dir, os.path.basename(src_path))
            shutil.copy(src_path, dst_path)
        for src_path in val_imgs:
            dst_path = os.path.join(val_class_dir, os.path.basename(src_path))
            shutil.copy(src_path, dst_path)
        
        print(f"类别 {class_name} 完成: train={len(train_imgs)}, val={len(val_imgs)}")

print("数据集整理完成！")
print(f"训练集目录: {train_dir}")
print(f"验证集目录: {val_dir}")
