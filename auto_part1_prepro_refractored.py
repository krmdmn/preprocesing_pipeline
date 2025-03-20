"""
Full Auto Pipeline: DICOM to NIfTI, Registration, and Brain Extraction

This pipeline performs the following steps:
1. Convert DICOM RT structure (with contours) to NIfTI using dcmrtstruct2nii.
2. Convert additional DICOM series (T1-pre, T2-ax, T2-flair) to NIfTI using dicom2nifti.
3. Perform rigid registration using ANTs, aligning T1 to T1ce and applying the same transforms to T2 and FLAIR.
4. Clean up the output directory by removing unwanted files.
5. Rename files for consistency.
6. Run brain extraction (HD-BET) on the registered images.

Requirements:
- Python 3.9
- pip install antspyx dicom2nifti dcmrtstruct2nii ants HD_BET
"""

import os
import shutil
import glob

# --- Step 1: DICOM RT Struct to NIfTI Conversion ---
from dcmrtstruct2nii import dcmrtstruct2nii, list_rt_structs

def convert_rtstruct_to_nifti(rt_struct_path: str, dicom_dir: str, output_dir: str) -> None:
    """
    Convert a DICOM RT structure file to a NIfTI mask.
    
    Parameters:
        rt_struct_path (str): Path to the DICOM RT structure file.
        dicom_dir (str): Directory containing the DICOM images.
        output_dir (str): Output directory for the converted NIfTI file.
    """
    print(f"Converting RT structure from {rt_struct_path} ...")
    dcmrtstruct2nii(rt_struct_path, dicom_dir, output_dir)
    print("RT structure conversion complete.\n")

# --- Step 2: DICOM Series Conversion to NIfTI ---
import dicom2nifti

def convert_dicom_directory(input_dir: str, output_dir: str, reorient: bool = True, compression: bool = True) -> None:
    """
    Convert an entire DICOM directory to NIfTI format.
    
    Parameters:
        input_dir (str): The input directory containing DICOM files.
        output_dir (str): The output directory where the NIfTI file will be saved.
        reorient (bool): Whether to reorient images (default True).
        compression (bool): Whether to compress the output (default True).
    """
    print(f"Converting DICOM series from {input_dir} ...")
    dicom2nifti.convert_directory(input_dir, output_dir, compression=compression, reorient=reorient)
    print("DICOM conversion complete.\n")

# --- Step 3: Rigid Registration Using ANTs ---
import ants

def perform_rigid_registration(t1ce_path: str, t1_path: str, t2_path: str, flair_path: str, output_dir: str) -> None:
    """
    Perform rigid registration of T1 (moving) to T1ce (fixed) and apply the transform to T2 and FLAIR.
    
    Parameters:
        t1ce_path (str): File path to the T1ce image (used as fixed image).
        t1_path (str): File path to the T1 image (used as moving image).
        t2_path (str): File path to the T2 image.
        flair_path (str): File path to the FLAIR image.
        output_dir (str): Output directory to save the registered images.
    """
    print("Loading images for registration...")
    fixed = ants.image_read(t1ce_path)
    moving = ants.image_read(t1_path)
    
    # Perform rigid registration
    print("Performing rigid registration...")
    registration = ants.registration(fixed=fixed, moving=moving, type_of_transform='Rigid')
    
    # Save registered T1 image
    registered_t1 = registration['warpedmovout']
    t1_save_path = os.path.join(output_dir, 'T1.nii.gz')
    ants.image_write(registered_t1, t1_save_path)
    print(f"Registered T1 image saved to: {t1_save_path}")
    
    # Get transformation parameters
    fwdtransforms = registration['fwdtransforms']
    
    # Apply the same transformation to T2 and FLAIR images
    for modality, input_path, out_name in zip(
        ['T2', 'FLAIR'],
        [t2_path, flair_path],
        ['T2.nii.gz', 'FLAIR.nii.gz']
    ):
        print(f"Applying transform to {modality} image...")
        moving_img = ants.image_read(input_path)
        registered_img = ants.apply_transforms(fixed, moving_img, fwdtransforms)
        save_path = os.path.join(output_dir, out_name)
        ants.image_write(registered_img, save_path)
        print(f"Registered {modality} image saved to: {save_path}")
    
    print("Rigid registration complete.\n")

# --- Step 4: Clean-Up Directory ---
def clean_directory(directory: str, files_to_keep: list) -> None:
    """
    Delete files and folders in a directory that are not in the 'files_to_keep' list.
    
    Parameters:
        directory (str): The directory to clean.
        files_to_keep (list): List of filenames to retain.
    """
    print(f"Cleaning directory: {directory}")
    for file in os.listdir(directory):
        if file not in files_to_keep:
            file_path = os.path.join(directory, file)
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
                    print(f"Deleted file: {file_path}")
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
                    print(f"Deleted directory and its contents: {file_path}")
            except Exception as e:
                print(f"Error deleting {file_path}: {e}")
    print("Directory clean-up complete.\n")

# --- Step 5: Rename Files ---
def rename_files_in_directory(directory: str, rename_mapping: dict) -> None:
    """
    Rename files in a directory according to a provided mapping.
    
    If the target file already exists, it will be removed first.
    
    Parameters:
        directory (str): The directory containing the files.
        rename_mapping (dict): A dictionary mapping old filenames to new filenames.
    """
    print(f"Renaming files in directory: {directory}")
    for old_name, new_name in rename_mapping.items():
        old_file = os.path.join(directory, old_name)
        new_file = os.path.join(directory, new_name)
        if os.path.isfile(old_file):
            # Remove target file if it already exists
            if os.path.exists(new_file):
                os.remove(new_file)
                print(f"Removed existing file: {new_file}")
            os.rename(old_file, new_file)
            print(f"Renamed '{old_name}' to '{new_name}'")
        else:
            print(f"File '{old_name}' not found in directory.")
    print("File renaming complete.\n")


# --- Step 6: Run HD-BET Brain Extraction ---
from HD_BET.run import run_hd_bet

def run_hd_bet_on_images(base_output_path: str, folder_name: str) -> None:
    """
    Run HD-BET brain extraction on a set of registered images.
    
    Parameters:
        base_output_path (str): Base directory where the converted images are located.
        folder_name (str): Name of the folder for the current case.
    """
    # Build full folder path
    folder_path = os.path.join(base_output_path, folder_name)
    
    # Find image files using glob (adjust patterns if needed)
    t1_path = os.path.join(folder_path, 'T1.nii.gz')
    t2_path = os.path.join(folder_path, 'T2.nii.gz')
    t1ce_path = os.path.join(folder_path, 'T1ce.nii.gz')
    flair_path = os.path.join(folder_path, 'FLAIR.nii.gz')
    
    # Define output names with folder prefix if desired
    output_paths = {
        'T1': os.path.join(folder_path, f'{folder_name}_T1.nii.gz'),
        'T2': os.path.join(folder_path, f'{folder_name}_T2.nii.gz'),
        'T1ce': os.path.join(folder_path, f'{folder_name}_T1ce.nii.gz'),
        'FLAIR': os.path.join(folder_path, f'{folder_name}_FLAIR.nii.gz')
    }
    
    print("Running HD-BET brain extraction on T1...")
    run_hd_bet(t1_path, output_paths['T1'], keep_mask=False, bet=True, overwrite=True)
    print("Running HD-BET brain extraction on T2...")
    run_hd_bet(t2_path, output_paths['T2'], keep_mask=False, bet=True, overwrite=True)
    print("Running HD-BET brain extraction on T1ce...")
    run_hd_bet(t1ce_path, output_paths['T1ce'], keep_mask=False, bet=True, overwrite=True)
    print("Running HD-BET brain extraction on FLAIR...")
    run_hd_bet(flair_path, output_paths['FLAIR'], keep_mask=False, bet=True, overwrite=True)
    print("HD-BET brain extraction complete.\n")

# --- Main Pipeline Execution ---
def main():
    # Define base paths (adjust these paths as needed)
    base_dicom_path = r"C:/Users/krm/Documents/pipeline_auto"
    case_folder = "STGL001"  # Example case folder name
    
    # Define paths for different modalities and outputs
    rt_struct_path = os.path.join(base_dicom_path, case_folder, "MR T1-post", "struct_set_2020-07-18_07-54-10.dcm")
    t1post_dicom = os.path.join(base_dicom_path, case_folder, "MR T1-post")
    t1pre_dicom  = os.path.join(base_dicom_path, case_folder, "MR T1-pre")
    t2ax_dicom   = os.path.join(base_dicom_path, case_folder, "MR T2-ax")
    t2flair_dicom= os.path.join(base_dicom_path, case_folder, "MR T2-flair")
    
    output_base = os.path.join(base_dicom_path, "output")
    
    # For naming output folders, we use the case folder name.
    output_case_folder = os.path.join(output_base, case_folder)
    os.makedirs(output_case_folder, exist_ok=True)
    
    # Step 1: Convert RT structure to NIfTI (for contour extraction)
    convert_rtstruct_to_nifti(rt_struct_path, t1post_dicom, output_case_folder)
    
    # Step 2: Convert DICOM series to NIfTI for T1-pre, T2-ax, and T2-flair
    convert_dicom_directory(t1pre_dicom, os.path.join(output_case_folder), reorient=True, compression=True)
    convert_dicom_directory(t2ax_dicom, os.path.join(output_case_folder), reorient=True, compression=True)
    convert_dicom_directory(t2flair_dicom, os.path.join(output_case_folder), reorient=True, compression=True)
    
    # Define file paths for registration (adjust file names as generated by conversion)
    # Note: It is assumed that the T1ce image is generated from the RT struct conversion.
    t1ce_file = os.path.join(output_case_folder, "image.nii.gz")
    t1_file   = os.path.join(output_case_folder, "8_t1-pre.nii.gz")
    t2_file   = os.path.join(output_case_folder, "4_t2-ax.nii.gz")
    flair_file= os.path.join(output_case_folder, "5_t2-flair.nii.gz")
    
    # Step 3: Perform rigid registration and save the registered images
    perform_rigid_registration(t1ce_file, t1_file, t2_file, flair_file, output_case_folder)
    
    # Step 4: Clean-up the output directory by removing unwanted files.
    files_to_keep = ["T1.nii.gz", "image.nii.gz", "T2.nii.gz", "FLAIR.nii.gz", 
                      "mask_GTV-post.nii.gz", "mask_CTV-post.nii.gz"]
    clean_directory(output_case_folder, files_to_keep)
    
    # Step 5: Rename files for consistency.
    rename_mapping = {
        "image.nii.gz": "T1ce.nii.gz",
        "mask_GTV-post.nii.gz": "GTV.nii.gz",
        "mask_CTV-post.nii.gz": "CTV.nii.gz"
    }
    rename_files_in_directory(output_case_folder, rename_mapping)
    
    # Step 6: Run HD-BET for brain extraction on the registered images
    run_hd_bet_on_images(output_base, case_folder)
    
    # Optional: Final clean-up/renaming if needed after HD-BET
    # For example, if HD-BET outputs files with names that need further renaming:
    final_rename_mapping = {
        f"{case_folder}_T1.nii.gz": "T1.nii.gz",
        f"{case_folder}_T1ce.nii.gz": "T1ce.nii.gz",
        f"{case_folder}_T2.nii.gz": "T2.nii.gz",
        f"{case_folder}_FLAIR.nii.gz": "FLAIR.nii.gz"
    }
    rename_files_in_directory(output_case_folder, final_rename_mapping)
    
    # Confirm final files in the directory
    print("\nFinal contents of the output folder:")
    for f in os.listdir(output_case_folder):
        print(f)

if __name__ == '__main__':
    main()
