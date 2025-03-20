# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 08:38:15 2023
# pip install tqdm
# pip install albumentations
# pip install natsort
# pip install rt_utils

@author: krm
"""

import sys
import os
import shutil
from multiprocessing import freeze_support
from contextlib import contextmanager

@contextmanager
def temporary_sys_path(path):
    """
    A context manager to temporarily add a directory to sys.path.
    The directory is removed after the context exits.
    """
    added = False
    if path not in sys.path:
        sys.path.append(path)
        added = True
    try:
        yield
    finally:
        if added and path in sys.path:
            sys.path.remove(path)

def run_storm_2d_slices():
    """
    Adds the necessary path, imports and runs the 2D slicing function.
    """
    path = "C:/Users/krm/Documents/pipeline_auto/binary_tc_Segmen/for_auto"
    with temporary_sys_path(path):
        from function_storm_loading_2d_slices_155_std_t1ce import storm_2d_slices

        my_old_path = 'C:/Users/krm/Documents/pipeline_auto/output'
        path_dataset = 'C:/Users/krm/Documents/pipeline_auto/output'
        dataset_name = 'storm_Data'
        modality_mr = 'all'

        storm_2d_slices(
            range_limit=range(1),
            modality_mr=modality_mr,
            file_type='test',
            path_dataset=path_dataset,
            dataset_name=dataset_name,
            delete_all='yes',
            original_path=my_old_path
        )

def run_test_auto():
    """
    Temporarily adds the path for 'test_auto' module and executes its main function.
    """
    path = "C:/Users/krm/Documents/pipeline_auto/binary_tc_Segmen"
    with temporary_sys_path(path):
        import test_auto
        test_auto.main()

def run_dice_calculation():
    """
    Temporarily adds the path for dice score calculation and runs the calculation.
    """
    path = "C:/Users/krm/Documents/pipeline_auto/binary_tc_Segmen"
    with temporary_sys_path(path):
        from diceScore_auto import dice_calculation
        dice_calculation()

def move_and_cleanup():
    """
    Moves the predicted file to the original folder and deletes the residue directory.
    """
    source_path = "C:/Users/krm/Documents/pipeline_auto/output/storm_Data/test/nifti/STGL001/pred_GTV.nii.gz"
    destination_path = "C:/Users/krm/Documents/pipeline_auto/output/STGL001/pred_GTV.nii.gz"
    directory_to_delete = "C:/Users/krm/Documents/pipeline_auto/output/storm_Data/"

    shutil.move(source_path, destination_path)

    if os.path.exists(directory_to_delete):
        shutil.rmtree(directory_to_delete)

def run_rt_struct():
    """
    Temporarily adds the main pipeline path, imports and runs the rt_struct function.
    """
    path = "C:/Users/krm/Documents/pipeline_auto"
    with temporary_sys_path(path):
        from rt_struct_auto import rt_struct
        rt_struct()

def main():
    run_storm_2d_slices()
    run_test_auto()
    run_dice_calculation()
    move_and_cleanup()
    run_rt_struct()
    print('finished')

if __name__ == '__main__':
    freeze_support()  # Needed for multiprocessing on Windows
    main()
