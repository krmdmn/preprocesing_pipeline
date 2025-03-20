Part 1 From DICOM to NIftI converting, registration, skull stripping and 
Part 2 autosegmentation (GTV/TC), convert NIftI to RTSTRUCT

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

For HD_BET
Isensee F, Schell M, Tursunova I, Brugnara G, Bonekamp D, Neuberger U, Wick A,
Schlemmer HP, Heiland S, Wick W, Bendszus M, Maier-Hein KH, Kickingereder P.
Automated brain extraction of multi-sequence MRI using artificial neural
networks. Hum Brain Mapp. 2019; 1–13. https://doi.org/10.1002/hbm.24750

For dcmrtstruct2nii
Thomas Phil, Thomas Albrecht, Skylar Gay, & Mathis Ersted Rasmussen. (2023). Sikerdebaard/dcmrtstruct2nii: dcmrtstruct2nii v5 (Version v5). Zenodo. https://doi.org/10.5281/zenodo.4037864T


