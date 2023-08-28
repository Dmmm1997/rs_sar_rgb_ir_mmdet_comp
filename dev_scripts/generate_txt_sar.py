import shutil
import os
import numpy as np

'''
然后执行以下脚本， 随机抽取20%样本作为验证集。

'''

train_split_ratio = 0.8

root_dir = "/home/dmmm/Dataset/2023_multi-modals_det_comp/SAR" # RGB通道
annotation_dir = os.path.join(root_dir, "Annotations")
image_dir = os.path.join(root_dir, "Images")

names = os.listdir(image_dir)
np.random.shuffle(names)

# 生成 全量txt文件
total_image_for_train = [name.split(".")[0] for name in names]

with open(os.path.join(root_dir, "total_{}.txt".format(len(total_image_for_train))), "w") as F:
    for item in total_image_for_train:
        F.write("%s\n" % item)

# 生成 部分txt文件 
names_for_train = names[:int(len(names)*train_split_ratio)]
part_image_for_train = [name.split(".")[0] for name in names_for_train]

with open(os.path.join(root_dir, "train_{}.txt".format(len(part_image_for_train))), "w") as F:
    for item in part_image_for_train:
        F.write("%s\n" % item)

# 生成 验证txt文件
names_for_val = names[int(len(names)*train_split_ratio):]
val_image_for_train = [name.split(".")[0] for name in names_for_val]
    
with open(os.path.join(root_dir, "val_{}.txt".format(len(val_image_for_train))), "w") as F:
    for item in val_image_for_train:
        F.write("%s\n" % item)


