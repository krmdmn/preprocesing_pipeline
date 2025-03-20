# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 15:15:56 2023

@author: krm
"""

import matplotlib.pyplot as plt
import nibabel as nib
from rt_utils import RTStructBuilder

def rt_struct():
    # Create new RT Struct. Requires the DICOM series path for the RT Struct.
    rtstruct = RTStructBuilder.create_new(
        dicom_series_path=r"C:/Users/krm/Documents/pipeline_auto/STGL001/MR T1-post")
    
    # Your code to generate the segmentation mask
    nii = nib.load(r"C:/Users/krm/Documents/pipeline_auto/output/STGL001/pred_GTV.nii.gz").get_fdata()
    
    import numpy as np
    nifti_segmentation_mask=nii
    nifti_segmentation_mask = np.rot90(np.rot90(np.rot90(nifti_segmentation_mask)))
    nifti_segmentation_mask=np.fliplr(nifti_segmentation_mask)
    
    
    numpy_segmentation_mask = nifti_segmentation_mask
    
    # Add the 3D mask as an ROI.
    numpy_segmentation_mask = numpy_segmentation_mask.astype(bool)
    rtstruct.add_roi(mask=numpy_segmentation_mask,color=[255, 0, 255], name='Pred')
    
    # Save the resulting RT Struct
    rtstruct.save("C:/Users/krm/Documents/pipeline_auto/output/STGL001")
    
    ####
    
    # # Load existing RT Struct. Requires the series path and existing RT Struct path
    # rtstruct = RTStructBuilder.create_from(
    #     dicom_series_path=r"D:\stgl_rt_dicom\dicom_Stgl\MR T1-post",
    #     rt_struct_path=r"D:\stgl_rt_dicom\dicom_Stgl\rt_origin\struct_set_2020-07-18_07-54-10.dcm"
    # )
    
    # # Add ROI. This is the same as the above example.
    # rtstruct.add_roi(
    #     mask=MASK_FROM_ML_MODEL,
    #     color=[255, 0, 255],
    #     name="RT-Utils ROI!"
    # )
    
    # rtstruct.save('new-rt-struct')
