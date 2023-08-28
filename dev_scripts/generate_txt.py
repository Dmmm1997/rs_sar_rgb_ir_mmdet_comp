import shutil
import os

'''
把航线1-7作为训练集，剩下的作为测试集合｜同时生成一个全量的txt

首先，把所有图片和标注移动到同一个文件夹下
cp /home/dmmm/Dataset/2023_multi-modals_det_comp/双光/VIData/images/*/*.jpg /home/dmmm/Dataset/2023_multi-modals_det_comp/双光/VIData/images/Image/
cp /home/dmmm/Dataset/2023_multi-modals_det_comp/双光/VIData/Annotation/*/*.xml /home/dmmm/Dataset/2023_multi-modals_det_comp/双光/VIData/Annotation/Annotations/
然后执行以下脚本，生成对应的全量txt文件 train_1to9.txt
                       部分txt文件 train_1to7.txt
                       验证txt文件 val_8to9.txt

'''


root_dir = "/home/dmmm/Dataset/2023_multi-modals_det_comp/双光/IRData" # RGB通道
annotation_dir = os.path.join(root_dir, "Annotation")
image_dir = os.path.join(root_dir, "images")

route_names = os.listdir(image_dir)
route_names.remove("Images")
route_names.sort()

# 生成 全量txt文件 train_1to9.txt
total_image_for_train = []
for i in range(len(route_names)):
    name = route_names[i]
    img_dir_path = os.path.join(image_dir, name)
    filenames = os.listdir(img_dir_path)
    total_image_for_train += [filename.split(".")[0] for filename in filenames]

with open(os.path.join(root_dir, "train_1to9.txt"), "w") as F:
    for item in total_image_for_train:
        F.write("%s\n" % item)

# 生成 部分txt文件 train_1to7.txt
train_list = route_names[:7]
part_image_for_train = []
for i in range(len(train_list)):
    name = train_list[i]
    img_dir_path = os.path.join(image_dir, name)
    filenames = os.listdir(img_dir_path)
    part_image_for_train += [filename.split(".")[0] for filename in filenames]

with open(os.path.join(root_dir, "train_1to7.txt"), "w") as F:
    for item in part_image_for_train:
        F.write("%s\n" % item)

# 生成 验证txt文件 val_8-9.txt
val_list = route_names[7:]
val_image_for_train = []
for i in range(len(val_list)):
    name = val_list[i]
    img_dir_path = os.path.join(image_dir, name)
    filenames = os.listdir(img_dir_path)
    val_image_for_train += [filename.split(".")[0] for filename in filenames]
    
with open(os.path.join(root_dir, "val_8to9.txt"), "w") as F:
    for item in val_image_for_train:
        F.write("%s\n" % item)


