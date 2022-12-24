from model import TimeSformer
from dataLoader import DataLoader
from torch import nn
import torch
import config
import numpy as np
import sys
import time
from tqdm import tqdm
from getSomeInfor import *

num_frames_to_take = int(min_video_frames / config.FRAMES_INTERVAL) - 1

### Tạo các dataloader

trainloader = DataLoader(videos_list=train_videos_list, labels_list=train_labels_list, num_frames_to_take=num_frames_to_take)
trainLoader = torch.utils.data.DataLoader(trainloader, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=2, drop_last=True)

valloader = DataLoader(videos_list=val_videos_list, labels_list=val_labels_list, num_frames_to_take=num_frames_to_take)
valLoader = torch.utils.data.DataLoader(valloader, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=2, drop_last=True)

### Khởi tạo model

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
).to("cuda:0")

### Định nghĩa các hyper-parameters cho việc training mô hình ###
epoch = 1
max_epoch = 100
test_step = 1
lr = 0.005
lr_decay = 0.97
weight_decay = 2e-5

loss_fn = torch.nn.MSELoss(reduction='sum') ### Hàm loss này có thể sử dụng CrossEntropy, ...
optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=test_step, gamma=lr_decay)

if __name__ == '__main__':

    while True:

        model.train()
        scheduler.step(epoch-1)
        lr = optim.param_groups[0]['lr']

        ## Training cho một epoch
        for index, (frames, labels) in enumerate(trainLoader, start=1):
            model.zero_grad()
            frames = frames.cuda()
            y_pred = model.forward(frames)
            loss = loss_fn(y_pred, torch.FloatTensor(labels.float()).cuda())
            loss.backward()
            optim.step()
            sys.stderr.write(time.strftime("%m-%d %H:%M:%S") + \
                " [%d] Lr: %5f, Training: %.2f%%, " %(epoch, lr, 100 * (index / trainLoader.__len__())) + \
                " Loss: %.5f \r" %(loss))
            sys.stderr.flush()
        
        sys.stdout.write("\n")

        ## Evaluation cho tập validation
        if epoch % test_step == 0:

            model.eval()

            torch.save(model.state_dict(), config.MODEL_SAVE_PATH + "/timesformer_%04d.ckpt"%epoch)

            count_true = 0
            val_loss = 0

            with torch.no_grad():

                for index, (frames, labels) in enumerate(tqdm(valLoader), start=1):

                    frames = frames.cuda()
                    y_pred = model.forward(frames)
                    loss_val = loss_fn(y_pred, torch.FloatTensor(labels.float()).cuda()).detach().cpu().numpy()
                    val_loss += loss_val
                    y_pred_np = y_pred.detach().cpu().numpy()
                    preds = []
                    labels_true = []

                    ### y_pred có shape [batch, num_labels]
                    ### [[0, 0, 0, ... 1]
                    #    ...............
                    #    [0, 1, 0, ....0]]
                    
                    ### Biến đổi để đưa y_pred, labels từ dạng one-hot về các chỉ số label từ 0 đến 11
                    ### [0, 0, 0, .... 1] => 11
                    ### [1, 0, 0, .....0] => 0

                    for i in range(y_pred_np.shape[0]):
                        pred = np.argmax(y_pred_np[i])
                        label = np.argmax(labels[i].detach().cpu().numpy())
                        preds.append(pred)
                        labels_true.append(label)
                        if pred == label:
                            count_true += 1
            
            print(time.strftime("%Y-%m-%d %H:%M:%S"), "[%d], ValAccuracy %2.2f%%, ValLoss %.5f " %(epoch, count_true / valLoader.__len__(), val_loss))

        if epoch >= max_epoch:
            quit()

        epoch += 1

