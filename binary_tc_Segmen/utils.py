import torch
import torchvision
from dataset import CarvanaDataset
from torch.utils.data import DataLoader
import numpy as np

def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])

def get_loaders(
    train_dir,
    train_maskdir,
    val_dir,
    val_maskdir,
    batch_size,
    train_transform,
    val_transform,
    num_workers=4,
    pin_memory=True,
):
    train_ds = CarvanaDataset(
        image_dir=train_dir,
        mask_dir=train_maskdir,
        transform=train_transform,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
    )

    val_ds = CarvanaDataset(
        image_dir=val_dir,
        mask_dir=val_maskdir,
        transform=val_transform,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )

    return train_loader, val_loader

def check_accuracy(loader, model, device="cuda"):
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    model.eval()
    nm=0
    dice_scoret=0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device).unsqueeze(1)
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)
            dice_scoret=0
            for i in range(len(x)):
                    
                tempd= (2 * (preds[i] * y[i]).sum()) / (
                    (preds[i] + y[i]).sum() + 1e-18
                )
                # print(tempd)
                dice_scoret=dice_scoret+tempd
                nm=nm+1
            # dice_scoret=dice_scoret
            dice_score+=dice_scoret
    # print(
    #     f"Got {num_correct}/{num_pixels} with acc {num_correct/num_pixels*100:.2f}"
    # )
    # print(f"Dice score: {dice_score/nm}")
    model.train()
    dice_scr=dice_score/nm
    return dice_scr









# def check_accuracy(loader, model, device="cuda"):
#     num_correct = 0
#     num_pixels = 0
#     dice_score = 0
#     model.eval()

#     with torch.no_grad():
#         for x, y in loader:
#             x = x.to(device)
           
#             # y=y/255
            
#             y = y.to(device).unsqueeze(1)
#             # z= (y > 0).float()
            
            
#             preds = torch.sigmoid(model(x))
#             preds = (preds > 0.5).float()
#             # z= (y > 0.1).float()
#             num_correct += (preds == y).sum()
#             num_pixels += torch.numel(preds)
#             #single dice score calculation
#             # a00=0
#             # for i in range(len(x[:,0,:,:])):
#             #     a0= ((2 * (preds[i,0,:,:] * y[i,0,:,:]).sum() ) / ((preds[i,0,:,:] + y[i,0,:,:]).sum() + 1e-18)).cpu()
#             #     a0=np.round(a0,10)
#             #     print(i)
#             #     print(a0)
#             #     a00 +=a0
#             # print("overall")
#             # print(a00/len(x[:,0,:,:]))
            
#             #total dice score
#             a= (2 * (preds * y).sum() ) / ((preds + y).sum() + 1e-18)  
#             if preds.sum()==0 and y.sum()==0:
#                 a=1
            
#             # print(a)
#             dice_score += a
#             # print(dice_score)

#     print(
#         f"Got {num_correct}/{num_pixels} with acc {num_correct/num_pixels*100:.2f}"
#     )
#     print(f"Dice score: {dice_score/len(loader)}")
#     model.train()
#     dice_scr=dice_score/len(loader)
#     return dice_scr.cpu()

# def save_predictions_as_imgs(
#     loader, model, folder="saved_images/", device="cuda"
# ):
#     model.eval()
#     for idx, (x, y) in enumerate(loader):
#         ix=x
        
#         # if idx==0:
#         #     mu=1
#         # elif idx==1:
#         #     mu=17
#         # elif idx==2:
#         #     mu=33
#         # else:
#         #     mu=49
#         x = x.to(device=device)
#         with torch.no_grad():
#             preds = torch.sigmoid(model(x))
#             preds = (preds > 0.5).float()
#         for i in range(len(ix[:,0,:,:])):
#             imu=i+mu
#             torchvision.utils.save_image(
#                 preds[i,0,:,:], f"{folder}/pred_{imu}.png"
#             )
#             torchvision.utils.save_image(
#                 ix[i,0,:,:], f"{folder}/real_{imu}.png"
#             )
#             torchvision.utils.save_image(y.unsqueeze(1)[i,0,:,:], f"{folder}{imu}.png")

def save_predictions_as_imgs(
    loader, model, folder="saved_images/", device="cuda"
):
    model.eval()
    batch_size=0
    for idx, (x, y) in enumerate(loader):
        ix=x*255.0
        x = x.to(device=device)
        with torch.no_grad():
            preds = torch.sigmoid(model(x))
            # preds = (preds > 0.5).float()
        for i in range(len(x)):
            imu=i+batch_size
            # torchvision.utils.save_image(
            #     preds[i,0,:,:], f"{folder}/pred_{imu}.png"
            # )
            # torchvision.utils.save_image(y.unsqueeze(1)[i,0,:,:], f"{folder}/mask_{imu}.png")
            # torchvision.utils.save_image(ix[i,:,:,:], f"{folder}/real_{imu}.png")
            
            np.save(f"{folder}/pred_{imu}", preds[i,:,:].cpu().float())
            np.save(f"{folder}/mask_{imu}", y[i,:,:].cpu().float())
            np.save(f"{folder}/real_{imu}",ix[i,:,:,:])
        batch_size +=len(x)
        
    model.train()
    
    
# def save_predictions_as_imgs(
#     loader, model, folder="saved_images/", device="cuda"
# ):
#     model.eval()
#     for idx, (x, y) in enumerate(loader):
#         ix=x*255.0
#         x = x.to(device=device)
#         with torch.no_grad():
#             preds = torch.sigmoid(model(x))
#             preds = (preds > 0.5).float()
#         torchvision.utils.save_image(
#             preds, f"{folder}/pred_{idx}.png"
#         )
#         torchvision.utils.save_image(y.unsqueeze(1), f"{folder}/mask_{idx}.png")
#         torchvision.utils.save_image(ix, f"{folder}/real_{idx}.png")
        
        
#     model.train()

# def save_predictions_as_imgs(
#     loader, model, folder="saved_images/", device="cuda"
# ):
#     model.eval()
#     for idx, (x, y) in enumerate(loader):
#         x = x.to(device=device)
#         with torch.no_grad():
#             preds = torch.sigmoid(model(x))
#             preds = (preds > 0.5).float()
#         torchvision.utils.save_image(
#             preds, f"{folder}/pred_{idx}.png"
#         )
#         torchvision.utils.save_image(y.unsqueeze(1), f"{folder}/mask_{idx}.png")

#     model.train()











