### Chuẩn bị một số thông tin data cho việc huấn luyện mô hình ###

import glob
import config
import math
import numpy as np

train_videos_list = []
train_labels_list = []

val_videos_list = []
val_labels_list = []

folder_name = "rgb_frames"

# Lấy ra videos có số lượng frames là ít nhất
min_video_frames = math.inf 

for video_file_name in glob.glob("rgb_frames/*"):
    frames = glob.glob(video_file_name + "/*")
    length = len(frames)
    if length < min_video_frames:
        min_video_frames = length

### Đọc dữ liệu metadata train.txt, val.txt để lấy ra list các video và label tương ứng

with open("metadata/train.txt", "r", encoding="utf8") as fr:
    for line in fr.readlines():
        ### Biến đổi label về dạng one-hot
        label = np.zeros((1, config.NUM_CLASSES))[0]
        label[int(line.replace("\n", "").split()[1])] = 1
        train_labels_list.append(label)
        train_videos_list.append(folder_name + "/" + line.replace("\n", "").split()[0])

with open("metadata/val.txt", "r", encoding="utf8") as fr:
    for line in fr.readlines():
        ### Biến đổi label về dạng one-hot
        label = np.zeros((1, config.NUM_CLASSES))[0]
        label[int(line.replace("\n", "").split()[1])] = 1
        val_labels_list.append(label)
        val_videos_list.append(folder_name + "/" + line.replace("\n", "").split()[0])

### Có tổng cộng 12 class
num_classes = 12