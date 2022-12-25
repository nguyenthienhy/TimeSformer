import cv2
import numpy as np
import torch
import glob
from sklearn.utils import shuffle

class DataLoader:

  def __init__(self, videos_list, labels_list, num_frames_to_take):
    self.videos_list = videos_list # List đường dẫn đến các thư mục video
    self.labels_list = labels_list # List label tương ứng với các thư mục video
    self.num_frames_to_take = num_frames_to_take # Số lượng frames được sử dụng trong mỗi video

  def __getitem__(self, index):
    frames = []
    frames_list = glob.glob(self.videos_list[index] + "/*") # List các đường dẫn frames trong một video
    frames_list = sorted(frames_list, key=lambda x: int(x.split("/")[-1].split("_")[1].split(".")[0]), reverse=False) # Phải sắp xếp thứ tự các frames từ 1 đến n ...
    frames_list = frames_list[0:self.num_frames_to_take] # Lấy ra số lượng frames trong một video
    for frame_path in frames_list:
      ### Đọc dữ liệu ảnh ###
      frame = cv2.imread(frame_path, cv2.IMREAD_COLOR)
      frame = np.transpose(np.asarray(cv2.resize(frame, (frame.shape[0], frame.shape[1]))), (2, 0, 1))
      frames.append(frame)
      ## Kết thúc đọc dữ liệu ảnh ###
    frames = np.array(frames)
    label = self.labels_list[index]
    return torch.FloatTensor(frames), label

  def __len__(self):
    return len(self.videos_list)

if __name__ == "__main__":
  dataloader = DataLoader(glob.glob("rgb_frames/*"), [0 for _ in glob.glob("rgb_frames/*")], 10)
  print(dataloader.__getitem__(0)[0].shape)