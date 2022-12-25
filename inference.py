from model import TimeSformer
from torch import nn
import torch
import config
import cv2
import numpy as np

min_video_frames = 32 # Thông số này được tính trong getSomeInfor.py

model = torch.nn.Sequential(
    TimeSformer(dim=config.DIM,
        image_size=config.IMAGE_SIZE, 
        patch_size=config.PATCH_SIZE, 
        num_frames=config.NUM_FRAMES, 
        num_classes=config.NUM_CLASSES, 
        depth=config.DEPTH, 
        heads=config.HEADS, 
        dim_head=config.DIM_HEAD, 
        attn_dropout=config.ATTN_DROPOUT, 
        ff_dropout=config.FF_DROPOUT
    ),
    nn.Softmax(dim=1)
).to("cuda:0") # Đưa mô hình vào cuda, do có 1 gpu nên id gpu sẽ là "cuda:0"

epoch_to_load = 1

model.load_state_dict(torch.load(config.MODEL_SAVE_PATH + "/timesformer_%04d.ckpt"%epoch_to_load)) # Load checkpoint

def predict(video_file_name):

    print("Loading " + video_file_name + "...")

    vidcap  = cv2.VideoCapture(video_file_name)
    success, image = vidcap.read()
    considered_frames_counter = 0
    frames = []

    while success:    

        if considered_frames_counter == min_video_frames:
            break

        success, image = vidcap.read()
        
        if considered_frames_counter == min_video_frames:
            cv2.imshow(video_file_name, cv2.resize(image, (224, 224)))
        
        if success and considered_frames_counter % min_video_frames == 0:
            image = np.transpose(np.asarray(cv2.resize(image, (224, 224))), (2, 0, 1))
            frames.append(image)
        
        if success:
            considered_frames_counter += 1

    frames = np.array(frames)
    frames = np.expand_dims(frames, axis=0)
    
    with torch.no_grad():
        frames = torch.FloatTensor(frames).cuda()
        y_pred = model.forward(frames)
        y_pred = y_pred.detach().cpu().numpy()[0]
        return np.argmax(y_pred)

if __name__ == "__main__":
    ### Add some code here ###
    label = predict("videoTests/chess.mov")
    print(label)