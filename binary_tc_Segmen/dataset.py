import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
from natsort import natsorted

class CarvanaDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir)
        self.images.sort()
        self.images=natsorted(self.images)
        
    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.images[index].replace("image", "mask"))
        
        # image = np.array(Image.open(img_path))
        image=np.load(img_path)
        image=np.float32(image)

        # image=image[:,:,1]
        
        # image=np.stack([image[:,:,1]],axis=2)
        #np stack ile combination yap        
        
        # image = np.array(Image.open(img_path).convert("L")) #no need to convert greyscale here
        
        # mask = np.array(Image.open(mask_path), dtype=np.float32)
        mask = np.load(mask_path)
        mask=np.float32(mask)
        # change to tumor core
        mask[mask==2]=0
        mask[mask==3]=1
        
        #tumor core
        # mask=mask[:,:,3]+mas1k[:,:,1]
        #enhancing tumor
        # mask=mask[:,:,3]
        #whole tumor
        # mask=mask[:,:,1]+mask[:,:,2]+mask[:,:,3]

        
        # mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)
        # mask[mask == 255.0] = 1.0
        
        # mask[mask != 0] = 1.0
        
        # mask[mask !=1] = 0.0

        
        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]

        return image, mask

