import torch
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from model import UNET
from utils import (
    load_checkpoint,
    save_checkpoint,
    get_loaders,
    check_accuracy,
    save_predictions_as_imgs,
)
import time
import matplotlib.pyplot as plt

# data 2d slice new
from function_brats_loading_2d_slices_155_std_t1ce import brats_2d_slices  

# Hyperparameters etc.
LEARNING_RATE = 1e-4

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
NUM_EPOCHS = 30
NUM_WORKERS = 4
IMAGE_HEIGHT = 240  # 1280 originally
IMAGE_WIDTH = 240 # 1918 originally
PIN_MEMORY = True
LOAD_MODEL = False
# TRAIN_IMG_DIR = '/home/kerimduman/Downloads/monai/code/2D_UNET/dataset/train/images'
# TRAIN_MASK_DIR = '/home/kerimduman/Downloads/monai/code/2D_UNET/dataset/train/masks'
# VAL_IMG_DIR = '/home/kerimduman/Downloads/monai/code/2D_UNET/dataset/val/images'
# VAL_MASK_DIR = '/home/kerimduman/Downloads/monai/code/2D_UNET/dataset/val/masks'


def train_fn(loader, model, optimizer, loss_fn, scaler):
    loop = tqdm(loader)

    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=DEVICE)
        targets = targets.float().unsqueeze(1).to(device=DEVICE)

        # forward
        with torch.cuda.amp.autocast():
            predictions = model(data)
            loss = loss_fn(predictions, targets)

        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # update tqdm loop
        loop.set_postfix(loss=loss.item())
    return loss.item()

def main():
    train_transform = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Rotate(limit=35, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.Normalize(
                mean=[0.0],
                std=[1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )

    val_transforms = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Normalize(
                mean=[0.],
                std=[1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
        
    )
    
    # only 1 modality
    model = UNET(in_channels=1, out_channels=1).to(DEVICE)
    # 4 modality
    # model = UNET(in_channels=4, out_channels=1).to(DEVICE)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_loader, val_loader = get_loaders(
        TRAIN_IMG_DIR,
        TRAIN_MASK_DIR,
        VAL_IMG_DIR,
        VAL_MASK_DIR,
        BATCH_SIZE,
        train_transform,
        val_transforms,
        NUM_WORKERS,
        PIN_MEMORY,
    )

    if LOAD_MODEL:
        load_checkpoint(torch.load("my_checkpoint.pth.tar"), model)


    # check_accuracy(val_loader, model, device=DEVICE)
    scaler = torch.cuda.amp.GradScaler()
    totime=0
    lst_loss=[]
    lst_dice=[]
    loss_val=100
    diceS=0
    for epoch in range(NUM_EPOCHS):
        print("\n")
        print(epoch)
        print("\n")
        fit_start = time.process_time()
        tmp_loss=train_fn(train_loader, model, optimizer, loss_fn, scaler)
        
        if tmp_loss < loss_val:
            # save model
            print("better loss is saving")
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer":optimizer.state_dict(),
            }
            afile="my_checkpoint.pth.tar"
            save_checkpoint(checkpoint,filename=afile)
            loss_val=tmp_loss
        # check accuracy
        
        if epoch%10==0:
            #tmp_dice=check_accuracy(val_loader, model, device=DEVICE)
            tmp_dice=0
            if tmp_dice > diceS:
                print("better validation dice saving")
                # save model
                checkpoint = {
                    "state_dict": model.state_dict(),
                    "optimizer":optimizer.state_dict(),
                }
                afile="my_checkpoint_dice.pth.tar"
                save_checkpoint(checkpoint,filename=afile)
                diceS=tmp_dice
     
        # print some examples to a folder
        # save_predictions_as_imgs(
        #     val_loader, model, folder="saved_images/", device=DEVICE
        # )
        fit_end = time.process_time()
        print("\ntrain time is in seconds: ", fit_end - fit_start)
        totime += fit_end - fit_start
        lst_loss.append(tmp_loss)
        lst_dice.append(float(tmp_dice))
        # print("loss value")
        # print(lst_loss)
        # print("dice value")
        # print(lst_dice)
        
        
        
    print("\ntotal time in hour:", totime/3600)
    return lst_loss,lst_dice
    
    
if __name__ == "__main__":
    my_old_path='/home/kerimduman/Downloads/monai/code/biUNET/tc'
    path_dataset='/home/kerimduman/Downloads/monai/code/2D_UNET'
    dataset_name='brats_flair_1_mod'
    modality_mr='flair'
    float_type=np.float32
    # data_2d slicer
    
    # train 2d slices
    # range_limit=range(876)
    brats_2d_slices(range_limit=range(876),modality_mr=modality_mr,
                    file_type='train',
                    path_dataset=path_dataset,
                    dataset_name=dataset_name,
                    delete_all='yes',
                    original_path=my_old_path,
                    float_type=float_type)
    
    
    # val 2d slices
    # range_limit=range(876,876+188)
    brats_2d_slices(range_limit=range(876,876+188),modality_mr=modality_mr,
                    file_type='val',
                    path_dataset=path_dataset,
                    dataset_name=dataset_name,
                    delete_all='no',
                    original_path=my_old_path,
                    float_type=float_type)
    
    
    
    # test 2d slices
    # range_limit=range(876+188,188+188)
    brats_2d_slices(range_limit=range(876+188,876+188+187),modality_mr=modality_mr,
                    file_type='test',
                    path_dataset=path_dataset,
                    dataset_name=dataset_name,
                    delete_all='no',
                    original_path=my_old_path,
                    float_type=float_type)
    
    
    TRAIN_IMG_DIR = path_dataset + '/'+ dataset_name + '/train/images'
    TRAIN_MASK_DIR =path_dataset + '/'+ dataset_name + '/train/masks'
    VAL_IMG_DIR = path_dataset + '/'+ dataset_name + '/val/images'
    VAL_MASK_DIR = path_dataset + '/'+ dataset_name + '/val/masks'
    
    loss_list, loss_dice=main()
    plt.plot(loss_list,label='loss')
    plt.xlabel("n iteration")
    plt.ylabel("loss function value")
    plt.legend(loc='upper left')
    plt.title("loss graph")
    plt.show()
    
    plt.plot(loss_dice,label='dice_Score')
    plt.xlabel("n iteration")
    plt.ylabel("dice score value")
    plt.legend(loc='upper left')
    plt.title("dice score graph")
    plt.show()
    