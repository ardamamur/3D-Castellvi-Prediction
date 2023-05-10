import numpy as np
from BIDS import BIDS_Family

def get_cutout(bids_family: BIDS_Family, roi_object_idx: list[int], return_seg = False):

    ct_nii = bids_family["ct"][0].open_nii()
    seg_nii = bids_family["msk_seg-vertsac"][0].open_nii()

    #separate arrays to manipulate the ct_nii_array 
    vert_arr = seg_nii.get_array()
    ct_arr = ct_nii.get_array()

    #get all indices of voxels classified as belonging to the relevant object(s)
    roi_vox_idx = np.where((vert_arr[:,:,:,None] == roi_object_idx).any(axis = 3))

    ct_nii.set_array_(ct_arr[roi_vox_idx[0].min():roi_vox_idx[0].max(), roi_vox_idx[1].min():roi_vox_idx[1].max(), roi_vox_idx[2].min():roi_vox_idx[2].max()])

    #let's not forget to return a properly oriented and scaled version of the nii
    ct_nii.rescale_and_reorient_(axcodes_to=('P', 'I', 'R'), voxel_spacing = (1,1,1))

    if return_seg:
        seg_nii.set_array_(vert_arr[roi_vox_idx[0].min():roi_vox_idx[0].max(), roi_vox_idx[1].min():roi_vox_idx[1].max(), roi_vox_idx[2].min():roi_vox_idx[2].max()])
        seg_nii.rescale_and_reorient_(axcodes_to=('P', 'I', 'R'), voxel_spacing = (1,1,1))


    return ct_nii, seg_nii