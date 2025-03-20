#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 17 13:16:07 2022

@author: kerimduman
"""

#2d 4 modality with combinations, and 3 slices as an input.


# https://youtu.be/oB35sV1npVI
"""
Use this code to get your BRATS 2020 dataset ready for semantic segmentation. 
Code can be divided into a few parts....

#Combine 
#Changing mask pixel values (labels) from 4 to 3 (as the original labels are 0, 1, 2, 4)
#Visualize


https://pypi.org/project/nibabel/

All BraTS multimodal scans are available as NIfTI files (.nii.gz) -> commonly used medical imaging format to store brain imagin data obtained using MRI and describe different MRI settings

T1: T1-weighted, native image, sagittal or axial 2D acquisitions, with 1–6 mm slice thickness.
T1c: T1-weighted, contrast-enhanced (Gadolinium) image, with 3D acquisition and 1 mm isotropic voxel size for most patients.
T2: T2-weighted image, axial 2D acquisition, with 2–6 mm slice thickness.
FLAIR: T2-weighted FLAIR image, axial, coronal, or sagittal 2D acquisitions, 2–6 mm slice thickness.

#Note: Segmented file name in Folder 355 has a weird name. Rename it to match others.
"""

import os
import numpy as np
import nibabel as nib
import glob
# from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
from tifffile import imsave

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

from tqdm import tqdm
##########################



t1_list = sorted(glob.glob("C:/Users/krm/Documents/pipeline_auto/output/STGL001/T1.nii.gz"))
t2_list = sorted(glob.glob("C:/Users/krm/Documents/pipeline_auto/output/STGL001/T2.nii.gz"))
t1ce_list = sorted(glob.glob("C:/Users/krm/Documents/pipeline_auto/output/STGL001/T1ce.nii.gz"))
flair_list = sorted(glob.glob("C:/Users/krm/Documents/pipeline_auto/output/STGL001/FLAIR.nii.gz"))
mask_list = sorted(glob.glob("C:/Users/krm/Documents/pipeline_auto/output/STGL001/GTV.nii.gz"))



# You can check the current working directory to verify the change
current_directory = os.getcwd()
print("Current Working Directory:", current_directory)

def storm_2d_slices(range_limit=range(1),modality_mr='all',file_type='train',
                    path_dataset='C:/Users/krm/Documents/pipeline_auto/output/STGL001',
                    dataset_name='dataset2',delete_all='yes',
                    original_path='C:/Users/krm/Documents/pipeline_auto/output/STGL001',
                    float_type=np.float32): 
    if False:
        my_old_path='/home/kerimduman/Downloads/monai/code/biUNET/tc'
        os.chdir(my_old_path)
    
    old_directory = original_path
    
    new_directory_path = path_dataset

    # Change the current working directory to the specified path
    os.chdir(new_directory_path)

    # You can check the current working directory to verify the change
    current_directory = os.getcwd()
    print("Current Working Directory:", current_directory)
    
    # 
    import shutil
    directory='./{}'.format(dataset_name)
    if delete_all=='yes':
    
            
        if os.path.exists(directory):
            shutil.rmtree(directory)
    
        os.mkdir(directory)
    
    else:
            
        if not os.path.exists(directory):
            os.mkdir(directory)
    
    
    
    file=file_type
    
    # 
    import shutil
    directory='./{}'.format(dataset_name)+'/'+file_type
    if os.path.exists(directory):
        shutil.rmtree(directory)

    os.mkdir(directory)
    
    # 
    import shutil
    directory='./{}'.format(dataset_name)+'/'+'{}/images'.format(file_type)
    if os.path.exists(directory):
        shutil.rmtree(directory)

    os.mkdir(directory)
    
    # 
    import shutil
    directory='./{}'.format(dataset_name)+'/'+'{}/masks'.format(file_type)
    if os.path.exists(directory):
        shutil.rmtree(directory)

    os.mkdir(directory)
    

    
    
    #2d slices for 4modalities
    
    
    # for img in tqdm(range(len(t2_list))):   #Using t1_list as all lists are of same size
    # for img in tqdm(range(876)):   #Using t1_list as all lists are of same size
    for img in tqdm(range_limit):   #Using t1_list as all lists are of same size
    # for img in tqdm(range(876+188,876+188+188)):   #Using t1_list as all lists are of same size
        # print("Now preparing image and masks number: ", img)
        # img=random.randint(0, len(t2_list))
        temp_modality=0
        
        if modality_mr=='all':
            temp_image_t2=nib.load(t2_list[img]).get_fdata()
            # temp_image_t2x=scaler.fit_transform(temp_image_t2.reshape(-1, temp_image_t2.shape[-1])).reshape(temp_image_t2.shape)
            a=temp_image_t2
            mean = a.mean()
            std = a.std()
            b = (a - mean)/(std + 1e-8)
            np.min(b), np.max(b)
            temp_image_t2=b
            temp_modality=temp_image_t2
           
        if modality_mr=='all':
            temp_image_t1ce=nib.load(t1ce_list[img]).get_fdata()
            # temp_image_t1ce=scaler.fit_transform(temp_image_t1ce.reshape(-1, temp_image_t1ce.shape[-1])).reshape(temp_image_t1ce.shape)
            a=temp_image_t1ce
            mean = a.mean()
            std = a.std()
            b = (a - mean)/(std + 1e-8)
            np.min(b), np.max(b)
            temp_image_t1ce=b
            temp_modality=temp_image_t1ce
        
       
        
        if modality_mr=='all':
            temp_image_flair=nib.load(flair_list[img]).get_fdata()
            # temp_image_flair=scaler.fit_transform(temp_image_flair.reshape(-1, temp_image_flair.shape[-1])).reshape(temp_image_flair.shape)
            a=temp_image_flair
            mean = a.mean()
            std = a.std()
            b = (a - mean)/(std + 1e-8)
            np.min(b), np.max(b)
            temp_image_flair=b
            temp_modality=temp_image_flair
    
    
    
        if modality_mr=='all':
            temp_image_t1=nib.load(t1_list[img]).get_fdata()
            # temp_image_t1=scaler.fit_transform(temp_image_t1.reshape(-1, temp_image_t1.shape[-1])).reshape(temp_image_t1.shape)
            a=temp_image_t1
            mean = a.mean()
            std = a.std()
            b = (a - mean)/(std + 1e-8)
            np.min(b), np.max(b)
            temp_image_t1=b
            temp_modality=temp_image_t1
        
        
        
        
        temp_mask=nib.load(mask_list[img]).get_fdata()
        temp_mask=temp_mask.astype(np.uint8)
        temp_mask[temp_mask==4] = 3  #Reassign mask values 4 to 3
        # print(np.unique(temp_mask))
        
        # change to tumor core
        # temp_mask[temp_mask==2]=0
        # temp_mask[temp_mask==3]=1
        
        # #change to enhancing tumor
        # temp_mask[temp_mask==1]=0
        # temp_mask[temp_mask==2]=0
        # temp_mask[temp_mask==3]=1
        
        # change to whole tumor
        # temp_mask[temp_mask==1]=1
        # temp_mask[temp_mask==2]=1
        # temp_mask[temp_mask==3]=1
        
        # #change to necrotic
        # temp_mask[temp_mask==2]=0
        # temp_mask[temp_mask==3]=0
        
        
        
        temp_combined_images = np.stack([temp_image_flair, temp_image_t1ce, temp_image_t2, temp_image_t1], axis=3)
        # temp_combined_images=temp_modality.astype(float_type)
    
        #Crop to a size to be divisible by 64 so we can later extract 64x64x64 patches. 
        #cropping x, y, and z
        # temp_combined_images=temp_combined_images[25:225, 25:225, :]
        # temp_mask = temp_mask[25:225, 25:225, :]
        
        # plt.imshow(temp_combined_images[:,:,66,0])
        # plt.show()
        # plt.imshow(temp_mask[:,:,66])
        # plt.show()
        for index in range(temp_mask.shape[-1]):
        # for index in range(50,75):
            # print(index)
            # val, counts = np.unique(temp_mask[:,:,index], return_counts=True)
            
            if True:
            # if (counts.sum()-counts[0]) != 0 :  #At least 1 pixel useful volume with labels that are not 0
            # if (1 - (counts[0]/counts.sum())) > 0.01:  #At least 1% useful volume with labels that are not 0
                # print("Save Me")
                # temp_mask2= to_categorical(temp_mask[:,:,index], num_classes=4)
                 
                   # np.save('/home/kerimduman/Downloads/monai/code/2D_UNET/dataset2/train/images/image_'+str(img)+'_'+str(index)+'.npy', temp_combined_images[:,:,index])
                   # np.save('/home/kerimduman/Downloads/monai/code/2D_UNET/dataset2/train/masks/mask_'+str(img)+'_'+str(index)+'.npy', temp_mask[:,:,index])
                  
                    np.save('./{}'.format(dataset_name)+'/'+'{}/images/'.format(file)+'image_'+str(img)+'_'+str(index)+'.npy', temp_combined_images[:,:,index])
                    np.save('./{}'.format(dataset_name)+'/'+'{}/masks/mask_'.format(file)+str(img)+'_'+str(index)+'.npy', temp_mask[:,:,index])
                  
                   # np.save('/home/kerimduman/Downloads/monai/code/2D_UNET/dataset/test/images/image_'+str(img)+'_'+str(index)+'.npy', temp_combined_images[:,:,index])
                   # np.save('/home/kerimduman/Downloads/monai/code/2D_UNET/dataset/test/masks/mask_'+str(img)+'_'+str(index)+'.npy', temp_mask[:,:,index])
    
            # else:
                # print("I am useless")   
    # get back to the original fıle_path 
    os.chdir(old_directory)
    
    pass

def main():
    print(0)
    


# def main(my_old_path='/home/kerimduman/Downloads/monai/code/biUNET/tc'):
    
    # # my_old_path='/home/kerimduman/Downloads/monai/code/biUNET/tc'
    
    # # train 2d slices
    # # range_limit=range(876)
    # brats_2d_slices(range_limit=range(2),modality_mr='t1ce',
    #                 file_type='train',
    #                 path_dataset='/home/kerimduman/Downloads/monai/code/2D_UNET',
    #                 dataset_name='brats_t2',
    #                 delete_all='yes',
    #                 original_path=my_old_path)
    
    
    # # val 2d slices
    # # range_limit=range(876,188)
    # brats_2d_slices(range_limit=range(2,3),modality_mr='t1ce',
    #                 file_type='val',
    #                 path_dataset='/home/kerimduman/Downloads/monai/code/2D_UNET',
    #                 dataset_name='brats_t2',
    #                 delete_all='no',
    #                 original_path=my_old_path)
    
    
    
    # # test 2d slices
    # # range_limit=range(876+188,188+188)
    # brats_2d_slices(range_limit=range(3,4),modality_mr='t1ce',
    #                 file_type='test',
    #                 path_dataset='/home/kerimduman/Downloads/monai/code/2D_UNET',
    #                 dataset_name='brats_t2',
    #                 delete_all='no',
    #                 original_path=my_old_path)
    
    



# Check if this script is the main module and then call main
if __name__ == "__main__":
    my_old_path='C:/Users/krm/Documents/pipeline_auto/output'
    path_dataset='C:/Users/krm/Documents/pipeline_auto/output'
    dataset_name='storm_Data'
    modality_mr='all'
    # data_2d slicer
    # range_limit=range(876)
    storm_2d_slices(range_limit=range(1),modality_mr=modality_mr,
                    file_type='test',
                    path_dataset=path_dataset,
                    dataset_name=dataset_name,
                    delete_all='yes',
                    original_path=my_old_path)
    
    

    
    main()






   
