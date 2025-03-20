# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 10:53:23 2023

@author: krm
"""

import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import glob
from natsort import natsorted

import nibabel as nib

def dice_calculation():
    image_dir_pred =r"C:/Users/krm/Documents/pipeline_auto/output/storm_Data/test/saved_images/pred*"
    image_dir_mask =r"C:/Users/krm/Documents/pipeline_auto/output/storm_Data/test/saved_images/mask*"
    image_dir_real =r"C:/Users/krm/Documents/pipeline_auto/output/storm_Data/test/saved_images/real*"
    
    real_lst = sorted(glob.glob(image_dir_real))
    real_lst=natsorted(real_lst)
    predict_lst = sorted(glob.glob(image_dir_pred))
    predict_lst=natsorted(predict_lst)
    mask_lst = sorted(glob.glob(image_dir_mask))
    mask_lst=natsorted(mask_lst)
    
    #brats data load
    import numpy as np
    from numpy import asarray
    import nibabel as nib
    # import glob
    # from tensorflow.keras.utils import to_categorical
    import matplotlib.pyplot as plt
    from tifffile import imsave
    
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    
    from tqdm import tqdm
    
    
    t1_list = sorted(glob.glob("D:/radiomics/spaarc_python/the last version - sripped and registered/*/*T1.nii.gz"))
    t2_list = sorted(glob.glob("D:/radiomics/spaarc_python/the last version - sripped and registered/*/*T2.nii.gz"))
    # t1ce_list = sorted(glob.glob("D:/radiomics/spaarc_python/the last version - sripped and registered/*/*T1ce.nii"))
    flair_list = sorted(glob.glob("D:/radiomics/spaarc_python/the last version - sripped and registered/*/*FLAIR.nii.gz"))
    mask_list = sorted(glob.glob("D:/radiomics/spaarc_python/the last version - sripped and registered/*/*mask.nii"))
    
    t1ce_list=sorted(glob.glob("C:/Users/krm/Documents/pipeline_auto/output/STGL001/T1ce*"))
    # import random
    # random.seed(41)
    # # my_list = [0,1,2,3,4,5]
    
    # random.shuffle(t1_list)
    
    # random.seed(41)
    # random.shuffle(t2_list)
    
    # random.seed(41)
    # random.shuffle(t1ce_list)
    
    # random.seed(41)
    # random.shuffle(flair_list)
    
    # random.seed(41)
    # random.shuffle(mask_list)
    
    
    img=0
    
    temp_mask=nib.load(mask_list[img]).get_fdata()
    temp_mask=temp_mask.astype(np.uint8)
    # temp_mask[temp_mask==4] = 3
    
    affine=nib.load(t2_list[img]).affine
    
    
    
    for root, dirs, files in os.walk("D:/radiomics/spaarc_python/the last version - sripped and registered/", True):
       break
    
    dirs.sort()
    
    
    
    
    
    
    import shutil
    directory='C:/Users/krm/Documents/pipeline_auto/output/storm_Data/test/nifti'
    if os.path.exists(directory):
        shutil.rmtree(directory)
    
    os.mkdir(directory)
    
    #val or testing all files
    # for img in range(len(mask_list)):
    diceT=[]
    if True:
        
        offset=0
        for k in range(1): #int(len(mask_list)
        # for k in range(5):
    
            
            img=k                           
            
            affinex=nib.load(t1ce_list[img]).affine
            header = nib.load(t1ce_list[img]).header                             
            temp_mask=nib.load(t1ce_list[img]).get_fdata()
            slc=temp_mask.shape[-1]
    
    
            predC=[]
            flag=0
            for i in range(offset,offset+slc):
        
                #enhancing tumor
                # mask1=mask[:,:,3]
                # pred1=pred[:,:,3]
                # mask1[mask1 != 0] = 1.0
                pred=np.load(predict_lst[i])[0,:,:]
                # pred=pred[:,:,1]+pred[:,:,3]
                pred=Image.fromarray(pred)
    
                    
                pred = pred.resize((temp_mask.shape[1],temp_mask.shape[0]))            
                pred=asarray(pred).copy()  
                # plt.imshow(pred[:,:])
                pred[pred<0.5]=0
                # plt.imshow(pred[:,:])
                pred[pred!=0]=1.0
                # plt.imshow(pred[:,:])
                
                # pred=pred[:,:,2]
                
                
                
    
                # # pred=np.argmax(pred,axis=2)
                # mask=Image.open(mask_lst[i])
                
                # mask = mask.resize((temp_mask.shape[0],temp_mask.shape[1]))            
    
                # mask=asarray(mask)                         
                
                # mask=np.argmax(mask,axis=2)
                
                # # real=np.array(Image.open(real_lst[i]))
                
                # #enhancing tumor
                
                
                
                # mask1=mask.copy()
                         
                # # mask1[mask1<=5]=0
                # mask1[mask1!=0]=1.0
                # mask1[mask1 != 0] = 1.0
                # pred1[pred1 != 0] = 1.0
                
                pred1=np.expand_dims(pred, axis=2)
               
                if flag==0:
                    predC=pred1
               
                if flag!=0:
                   
                    predC=np.append(predC,pred1,axis=2)
                flag=1
                
            # fileN=testPaths[sayac][0].split('/')
            # os.mkdir(f"{folder}/{fileN[5]}")
           
           
            os.mkdir('C:/Users/krm/Documents/pipeline_auto/output/storm_Data/test/nifti'+'/'+dirs[img])
    
            #pred
            normal_array=predC
            # converted_array = np.array(normal_array, dtype=np.float32) # You need to replace normal array by yours
            # converted_array=np.fliplr(np.flipud(converted_array))
            # affinex = np.eye(4)
           
            
            
            # mask_RAS=converted_array
            # nifti_segmentation_mask=mask_RAS
            # nifti_segmentation_mask=np.fliplr(nifti_segmentation_mask)
            # nifti_segmentation_mask=np.flipud(nifti_segmentation_mask)
            # nifti_segmentation_mask=np.flipud(nifti_segmentation_mask)
            # nifti_segmentation_mask=np.flipud(nifti_segmentation_mask)

            nifti_file = nib.Nifti1Image(normal_array, affinex, header=header)
            nib.save(nifti_file, f"C:/Users/krm/Documents/pipeline_auto/output/storm_Data/test/nifti/{dirs[img]}/pred_GTV.nii.gz")
            
           
            # maskC=[]
            
            # flag=0
            # for i in range(offset,offset+slc):
            #     # pred=np.array(Image.open(predict_lst[i]))
                
            #     mask=np.array(Image.open(mask_lst[i]))
            #     # mask=mask[:,:,1]+mask[:,:,3]
    
            #     mask=Image.fromarray(mask)
                
            #     mask = mask.resize((temp_mask.shape[1],temp_mask.shape[0]))
    
                    
            #     mask=asarray(mask)
         
                             
                
            #     # mask=np.swapaxes(mask,0,1)
            #     mask[mask!=0]=1.0
            #     mask=mask[:,:,2]
    
    
            #     #
            #     # pred=Image.fromarray(pred)
    
                    
            #     # pred = pred.resize((temp_mask.shape[0],temp_mask.shape[1]))            
            #     # pred=asarray(pred)            
            #     # # mask1[mask1<=5]=0
            #     # pred[pred!=0]=1.0
                
                
                
            #     # mask=np.argmax(mask,axis=2)
            #     #enhancing tumor
            #     # mask1=mask[:,:,3]
            #     # pred1=pred[:,:,3]
                
            #     # mask1=mask[:,:,3]
            #     # pred1=pred[:,:,3]
            #     # mask1[mask1 != 0] = 1.0
            #     # pred1[pred1 != 0] = 1.0
                
            #     # mask=np.argmax(mask,axis=2)
                
            #     # pred1[pred1 != 0] = 1.0
            #     mask1=mask
    
            #     mask1=np.expand_dims(mask, axis=2)
               
            #     if flag==0:
            #         maskC=mask1
               
            #     if flag!=0:
                   
            #         maskC=np.append(maskC,mask1,axis=2)
            #     flag=1
            
            # #mask
            # normal_array=maskC
            # converted_array = np.array(normal_array, dtype=np.float32) # You need to replace normal array by yours
            # # converted_array=np.fliplr(np.flipud(converted_array))
            # # affine = np.eye(4)
            # nifti_file = nib.Nifti1Image(converted_array, affine)
           
            # nib.save(nifti_file, f"nifti/{dirs[img]}/mask_resized.nii")
            ##original niftii
            mask=nib.load(mask_list[img]).get_fdata()
            maskNifti=mask
            nifti_file = nib.Nifti1Image(maskNifti, affine)
               
            # nib.save(nifti_file, f"nifti/{dirs[img]}/mask_original.nii")
            dice_score_et = (2 * (predC * maskNifti).sum() ) / ((predC + maskNifti).sum() + 1e-18)  
            print(dice_score_et)
            diceT.append(dice_score_et)
            offset=offset+slc
    print('mean dice score')
    print(np.mean(diceT)) 
    mean_dice_score = np.mean(diceT)
    return mean_dice_score

# import pandas as pd
# df = pd.DataFrame(diceT, columns=["f1"])
# df = df[df.f1 >= 0.2]    
# mean_Dice = df.mean().values   
# #sort ny values ascending
# a=df.sort_values(by=['f1'],ascending=True)

# ######################

# import pandas as pd
# df = pd.DataFrame(diceT, columns=["f1"])

# dosya_liste=[]
# for i in range(len(t1ce_list)):
    
#     dosya=t1ce_list[i].split('/')[7]
#     dosya_liste.append(dosya)

# df2 = pd.DataFrame(dosya_liste, columns=["f2"])

# concat=pd.concat([df, df2], axis=1)

# a=concat.sort_values(by=['f1'],ascending=True)

# a.to_csv('list_DSC_binary',index=False)


 
# ###dice score resize version
# dice_score_et = (2 * (predC * maskC).sum() ) / ((predC + maskC).sum() + 1e-18)  
# print(dice_score_et)
# ###original nifti
# mask=nib.load(mask_list[img]).get_fdata()
# maskNifti=mask
# nifti_file = nib.Nifti1Image(maskNifti, affine)
   
# nib.save(nifti_file, f"nifti/{dirs[img]}/mask_original.nii")
# dice_score_et = (2 * (predC * maskNifti).sum() ) / ((predC + maskNifti).sum() + 1e-18)  
# print(dice_score_et)

def main():
    mean_dice_scorE=dice_calculation()
    print(mean_dice_scorE)
        


if __name__ == "__main__":
    main()
   
    
   


