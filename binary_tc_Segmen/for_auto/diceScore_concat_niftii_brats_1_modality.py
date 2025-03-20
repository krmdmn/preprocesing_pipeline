import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import glob
from natsort import natsorted

import nibabel as nib

image_dir_pred ='/home/kerimduman/Downloads/monai/code/biUNET/tc/saved_images//pred*'
image_dir_mask ='/home/kerimduman/Downloads/monai/code/biUNET/tc/saved_images/mask*'
image_dir_real ='/home/kerimduman/Downloads/monai/code/biUNET/tc/saved_images/real*'

real_lst = sorted(glob.glob(image_dir_real))
real_lst=natsorted(real_lst)
predict_lst = sorted(glob.glob(image_dir_pred))
predict_lst=natsorted(predict_lst)
mask_lst = sorted(glob.glob(image_dir_mask))
mask_lst=natsorted(mask_lst)

#brats data load
import numpy as np
import nibabel as nib
import glob
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
from tifffile import imsave

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

from tqdm import tqdm

t1_list = sorted(glob.glob('/home/kerimduman/Downloads/monai/brats21/TrainingData/*/*t1.nii.gz'))
t2_list = sorted(glob.glob('/home/kerimduman/Downloads/monai/brats21/TrainingData/*/*t2.nii.gz'))
t1ce_list = sorted(glob.glob('/home/kerimduman/Downloads/monai/brats21/TrainingData/*/*t1ce.nii.gz'))
flair_list = sorted(glob.glob('/home/kerimduman/Downloads/monai/brats21/TrainingData/*/*flair.nii.gz'))
mask_list = sorted(glob.glob('/home/kerimduman/Downloads/monai/brats21/TrainingData/*/*seg.nii.gz'))

import random
random.seed(41)
# my_list = [0,1,2,3,4,5]

random.shuffle(t1_list)

random.seed(41)
random.shuffle(t2_list)

random.seed(41)
random.shuffle(t1ce_list)

random.seed(41)
random.shuffle(flair_list)

random.seed(41)
random.shuffle(mask_list)


img=876+188

temp_mask=nib.load(mask_list[img]).get_fdata()
temp_mask=temp_mask.astype(np.uint8)
temp_mask[temp_mask==4] = 3

affine=nib.load(mask_list[img]).affine











slc=155
img=876

diceT=[]

#val or testing all files
# for img in range(876,876+188):

if True:
    
  
    for k in tqdm(range(int(len(mask_lst)/slc))):
        
        
        predC=[]
        flag=0
        for i in range(k*slc,k*slc+slc):
    
            #enhancing tumor
            # mask1=mask[:,:,3]
            # pred1=pred[:,:,3]
            # mask1[mask1 != 0] = 1.0
            pred=np.array(Image.open(predict_lst[i]))
            pred=pred[:,:,2]
            pred[pred!=0]=1
            
            mask=np.array(Image.open(mask_lst[i]))
            mask=mask[:,:,1]
            
            
            # realx=np.array(Image.open(real_lst[i]))
            # realx=realx[:,:,1]
         
            
            
            
            
            # real=np.array(Image.open(real_lst[i]))
            
            #enhancing tumor
            # mask1=mask[:,:,3]
            # pred1=pred[:,:,3]
            
            mask1=mask.copy()
            pred1=pred.copy()
            # mask1[mask1 != 0] = 1.0
            # pred1[pred1 != 0] = 1.0
            
            pred1=np.expand_dims(pred1, axis=2)
           
            if flag==0:
                predC=pred1
           
            if flag!=0:
               
                predC=np.append(predC,pred1,axis=2)
            flag=1
            
        # fileN=testPaths[sayac][0].split('/')
        # os.mkdir(f"{folder}/{fileN[5]}")
       
       
       
        # #pred
        # normal_array=predC
        # converted_array = np.array(normal_array, dtype=np.float32) # You need to replace normal array by yours
        # # converted_array=np.fliplr(np.flipud(converted_array))
        # # affine = np.eye(4)
        # nifti_file = nib.Nifti1Image(converted_array, affine)
       
        # nib.save(nifti_file, f"nifti/pred.nii")
        
       
        maskC=[]
        
        flag=0
        for i in range(k*slc,k*slc+slc):
            # pred=np.array(Image.open(predict_lst[i]))
            mask=np.array(Image.open(mask_lst[i]))
            mask=mask[:,:,2]
            mask[mask!=0]=1
            #enhancing tumor
            # mask1=mask[:,:,3]
            # pred1=pred[:,:,3]
            
            # mask1=mask[:,:,3]
            # pred1=pred[:,:,3]
            # mask1[mask1 != 0] = 1.0
            # pred1[pred1 != 0] = 1.0
            
            
            # pred1[pred1 != 0] = 1.0
            mask1=mask.copy()
            mask1=np.expand_dims(mask1, axis=2)

            if flag==0:
                maskC=mask1
           
            if flag!=0:
               
                maskC=np.append(maskC,mask1,axis=2)
            flag=1
        
        # #mask
        # normal_array=maskC
        # converted_array = np.array(normal_array, dtype=np.float32) # You need to replace normal array by yours
        # # converted_array=np.fliplr(np.flipud(converted_array))
        # # affine = np.eye(4)
        # nifti_file = nib.Nifti1Image(converted_array, affine)
       
        # nib.save(nifti_file, f"nifti/mask.nii")
        
        
        
        
        
        
        
        

        
        dice_score_et = (2 * (predC * maskC).sum() ) / ((predC + maskC).sum() + 1e-18)  
        
        if predC.sum()==0 and maskC.sum()==0:
            dice_score_et=1
        
        print(dice_score_et)
        diceT.append(dice_score_et)
print("tc mean")
print(np.mean(diceT))



    
   
    
   


