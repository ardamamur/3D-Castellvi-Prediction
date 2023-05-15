import logging
import numpy as np
import pandas as pd
from BIDS import BIDS_Global_info, BIDS_Family
from BIDS.snapshot2D import create_snapshot,Snapshot_Frame,Visualization_Type,Image_Modes
from pathlib import Path

class DataHandler:
    def __init__(self, master_list: str, dataset:list, data_types:list, image_types:list) -> None:
        """
        Initialize a new object of BIDS toolbox for the given dataset ( VerSe19 and VerSe20)
        """
        self.bids = BIDS_Global_info(dataset, data_types, additional_key = image_types, verbose=True)
        self.master_df = pd.read_excel(master_list)
        self.subject_list = self.master_df['Full_Id'].values.tolist()
    
    def _get_len(self):
        """
        Returns:
            length of the dataset
        """
        return len(self.bids.subjects) 
        
    def _get_families(self):
        """
        IDS splits data samples roughly into:
            - Subject: different patients
            - Sessions: one patient can have multiple scans

        You use enumerate_subjects to iterate over different, unique subjects.
        Then, you can use queries to apply various filters. If you use flatten=True, 
        that means you filter inividual files, and not a group/family of files.
        
        Everyone needs a family! 
        Files that are generated from others should belong to a family. 
        We automatically find related files and cluster them into a dictionary.

        Returns:
            Length of the families in the dataset
        """

        bids_families = []
        fam_count = {}
        for subject_name, subject_container in self.bids.enumerate_subjects(sort = True):

            query = subject_container.new_query(flatten=False) #<- flatten=False means we search for family
            #For the project, we need a ct scan, a segmentation and the centroid data. So let's filter for that
            query.filter('format','ct')
            query.filter('seg','vert')

            i = 0

            #now we can loop over families and gather some information
            for bids_family in query.loop_dict(sort=True):
                bids_families.append(bids_family)
                i = i + 1

            
            fam_count[query.subject.name] = i

        print("Total families in the dataset:", len(bids_families))
        return fam_count
    
    def _get_subjects_with_multiple_families(self,families):
        """
        Args:
            Dict : subjects as keys and related family numbers as values
        Return:
            List of subjects that has multiple families
        """
        keys_with_value_greater_than_1 = []
        for key, value in families.items():
            if value > 1:
                keys_with_value_greater_than_1.append(key)

        print("Subjects with multiple families:", keys_with_value_greater_than_1)
        return keys_with_value_greater_than_1
     
    def _get_subject_family(self, subject:str):
        """
        Return:
            First family of the subject
    
        TODO : Make enable for all families that subjects have
        """
        subject_container = self.bids.subjects[subject]
        query = subject_container.new_query(flatten = False)
        query.filter('format','ct')
        query.filter('seg','vert')
        family = next(query.loop_dict())
        return family

            
    def _get_missing_subjects(self):
        print("Number of missing subjects: {}".format(self.master_df["Full_Id"].isnull().sum()))
        null_indexes = self.master_df.index[self.master_df["Full_Id"].isnull()].tolist()
        return null_indexes

    def _get_missing_sacrum(self):
        print("Number of missing sacrum_seg: {}".format(self.master_df["Sacrum Seg"].isnull().sum()))
        null_indexes = self.master_df.index[self.master_df["Sacrum Seg"].isnull()].tolist()
        return null_indexes

    def _drop_missing_entries(self)->None:
        """
        Args:
            pd.DataFrame of ground truth labels ( master file)
        Returns:
            pd.DataFrame without missin entries.
        """
        if self._get_missing_subjects() == self._get_missing_sacrum():
            # Cause there is clearly a castellvi anomaly there, 
            # but the S1 is not fully visible in the scan 
            print("All missing subjects has no information about sacrum segmentation")
            self.master_df = self.master_df.dropna(subset=['Full_Id'])
            self.subject_list = self.master_df['Full_Id'].values.tolist()
        else:
            print('missing subjects: ' , self._get_missing_subjects())
            raise Exception('There are some mising subjects that has sacrum segmentation. Check them again')


    def _get_roi_object_idx(self, roi_parts:list):
        """
        Args:
            desired vertebra parts in a list
        Return:
            List of ids
        """
        v_idx2name = {
            1: "C1",     2: "C2",     3: "C3",     4: "C4",     5: "C5",     6: "C6",     7: "C7", 
            8: "T1",     9: "T2",    10: "T3",    11: "T4",    12: "T5",    13: "T6",    14: "T7",    15: "T8",    16: "T9",    17: "T10",   18: "T11",   19: "T12", 28: "T13",
            20: "L1",    21: "L2",    22: "L3",    23: "L4",    24: "L5",    25: "L6",    
            26: "S1",    29: "S2",    30: "S3",    31: "S4",    32: "S5",    33: "S6",
            27: "Cocc"
        }
        idx = [key for key, value in v_idx2name.items() if value in roi_parts]
        return sorted(idx)
    
    def pad_with_borders(self, img, pad_size):
        padding = [(s - i)//2 for s, i in zip(pad_size, img.shape)]
        return np.pad(img, [(p, p) for p in padding], mode='edge')

    def pad_3d_image(self, img, target_shape):

        print(img.shape)
        # Calculate padding sizes only if original dimension is less than target
        pad_x = max(0, target_shape[0] - img.shape[0]) if img.shape[0] < target_shape[0] else 0
        pad_y = max(0, target_shape[1] - img.shape[1]) if img.shape[1] < target_shape[1] else 0
        pad_z = max(0, target_shape[2] - img.shape[2]) if img.shape[2] < target_shape[2] else 0

        # Divide the padding evenly to both ends of the dimensions
        pad_x_before, pad_x_after = pad_x // 2, pad_x - pad_x // 2
        pad_y_before, pad_y_after = pad_y // 2, pad_y - pad_y // 2
        pad_z_before, pad_z_after = pad_z // 2, pad_z - pad_z // 2

        # Create padding specification
        padding = ((pad_x_before, pad_x_after), (pad_y_before, pad_y_after), (pad_z_before, pad_z_after))

        # Pad using np.pad
        img_padded = np.pad(img, padding, mode='constant', constant_values=0)

        return img_padded


    def resize_image(self, img, target_shape):
        # TODO Implement this function
        pass



    def _get_cutout(self, family:BIDS_Family, roi_object_idx: list[int], return_seg=False, pad=False, pad_size=(135, 181, 126), crop=False, crop_size=((135, 181, 126))):
        """
        Args:
            BIDS Family, roi_object_idx (id of the desired vertebras)
        Returns:
            Cutout for the given subject.
            Cutouts generally contains the following parts of last lumbar vertebra and sacrum:
                L4, L5, L6 (last lumbar vertebra) and S1 

            
            This version of the function first calculates the indices of the ROI, then extends them by 10 in each direction to get a slightly larger cutout. 
            It then applies the resize_image function to ensure that the cutout is not larger than the target size.
            The value 10 is arbitrary and you might need to adjust it to ensure that you're getting a large enough region around your ROI.
        """
        ct_nii = family["ct"][0].open_nii()
        seg_nii = family["msk_seg-vertsac"][0].open_nii()

       

        #separate arrays to manipulate the ct_nii_array 
        vert_arr = seg_nii.get_array()
        ct_arr = ct_nii.get_array()
        print('full ct shape:', ct_arr.shape)

        #get all indices of voxels classified as belonging to the relevant object(s)
        roi_vox_idx = np.where((vert_arr[:,:,:,None] == roi_object_idx).any(axis = 3))


        # Calculate the min and max indices for each dimension
        # TODO : Ask this approach to Hendrick
        min_idx = np.maximum(np.array([idx.min() for idx in roi_vox_idx]) - np.array([50, 50, 50]), 0)  # Ensure indices are not negative
        max_idx = np.array([idx.max() for idx in roi_vox_idx]) + np.array([50, 50, 50])



        ct_nii.set_array_(ct_arr[roi_vox_idx[0].min():roi_vox_idx[0].max(), roi_vox_idx[1].min():roi_vox_idx[1].max(), roi_vox_idx[2].min():roi_vox_idx[2].max()])

        #let's not forget to return a properly oriented and scaled version of the nii
        ct_nii.rescale_and_reorient_(axcodes_to=('P', 'I', 'R'), voxel_spacing = (1,1,1))

        seg_nii.set_array_(vert_arr[roi_vox_idx[0].min():roi_vox_idx[0].max(), roi_vox_idx[1].min():roi_vox_idx[1].max(), roi_vox_idx[2].min():roi_vox_idx[2].max()])
        seg_nii.rescale_and_reorient_(axcodes_to=('P', 'I', 'R'), voxel_spacing = (1,1,1))

        if return_seg:
            return seg_nii
        else:
            return ct_nii


    
    def _get_subject_name(self, subject:str):
        
        subject_name = None
        for sub in self.subject_list:
            if subject in sub:
                subject_name = sub
                return subject_name, True
        if subject_name == None :
            return subject_name, False
    
    def _is_multi_family(self, subject, families):
        if subject in families:
            return True
        else:
            return False

    def _get_max_shape(self, multi_family_subjects):
        """
        Args:
            subject name
        Return:
            the maximum shape within the subjects

        Be sure to drop missing subjects before running this function
        """

        max_shape_ct = np.array((0,0,0))
        max_shape_seg = np.array((0,0,0))
        for subject in self.bids.subjects:
            if not self._is_multi_family(subject, families=multi_family_subjects):
                sub_name, exists = self._get_subject_name(subject=subject)
                if exists:
                    print(sub_name)
                    last_l = self.master_df.loc[self.master_df['Full_Id'] == sub_name, 'Last_L'].values
                    roi_object_idx = self._get_roi_object_idx(roi_parts=[last_l, 'S1'])
                    family = self._get_subject_family(subject=subject)
                    seg_nii = self._get_cutout(family=family, roi_object_idx=roi_object_idx, return_seg=True)
                    ct_nii = self._get_cutout(family=family, roi_object_idx=roi_object_idx, return_seg=False)
                    max_shape_ct = np.maximum(max_shape_ct, ct_nii.get_array().shape)
                    print(max_shape_ct)
                    max_shape_seg = np.maximum(max_shape_seg, seg_nii.get_array().shape)
                    print(max_shape_seg)

        return tuple(max_shape_ct), tuple(max_shape_seg)
    

    def _get_resize_shape_V2(self, multi_family_subjects, is_max=True):
        """
        Args:
            subject name
        Return:
            the maximum shape within the subjects

        Be sure to drop missing subjects before running this function
        """

        resize_shape_ct = np.array((0,0,0))
        resize_shape_seg = np.array((0,0,0))
        for subject in self.bids.subjects:
            if not self._is_multi_family(subject, families=multi_family_subjects):
                sub_name, exists = self._get_subject_name(subject=subject)
                if exists:
                    print(sub_name)
                    last_l = self.master_df.loc[self.master_df['Full_Id'] == sub_name, 'Last_L'].values
                    roi_object_idx = self._get_roi_object_idx(roi_parts=[last_l, 'S1'])
                    family = self._get_subject_family(subject=subject)
                    seg_nii = self._get_cutout(family=family, roi_object_idx=roi_object_idx, return_seg=True)
                    ct_nii = self._get_cutout(family=family, roi_object_idx=roi_object_idx, return_seg=False)

                    ct_shape = np.array(ct_nii.get_array().shape)
                    seg_shape = np.array(seg_nii.get_array().shape)

                    # Update maximum shape if current image shape in each dimension is larger
                    for dim in range(3):
                        if is_max:
                            resize_shape_ct[dim] = max(resize_shape_ct[dim], ct_shape[dim])
                            resize_shape_seg[dim] = max(resize_shape_seg[dim], seg_shape[dim])
                        else:
                            resize_shape_ct[dim] = min(resize_shape_ct[dim], ct_shape[dim])
                            resize_shape_seg[dim] = min(resize_shape_seg[dim], seg_shape[dim])                           

                    print(resize_shape_ct)
                    print(resize_shape_seg)

        return tuple(resize_shape_ct), tuple(resize_shape_seg)




    def _get_subject_samples(self):
        '''
        This function extract subject names for training dataset creation
        '''
        bids_subjects = []
        master_subjects = []
        self._drop_missing_entries()
        families = self._get_families()
        multi_family_subjects = self._get_subjects_with_multiple_families(families)
        for subject in self.bids.subjects:
            if not self._is_multi_family(subject, families=multi_family_subjects):
                sub_name, exists = self._get_subject_name(subject=subject)
                if exists:
                    bids_subjects.append(subject)
                    master_subjects.append(sub_name)
        return bids_subjects, master_subjects
    


def main():
    dataset = ['/data1/practical-sose23/dataset-verse19',  '/data1/practical-sose23/dataset-verse20']
    data_types = ['rawdata',"derivatives"]
    image_types = ["ct", "subreg", "cortex"]
    master_list = '../dataset/VerSe_masterlist.xlsx'
    processor = DataHandler(master_list=master_list ,dataset=dataset, data_types=data_types, image_types=image_types)
    processor._drop_missing_entries()
    families = processor._get_families()
    #print(families)
    # multi_family_subjects = processor._get_subjects_with_multiple_families(families)
    # max_ct, max_seg = processor._get_resize_shape_V2(multi_family_subjects, is_max=True)
    # min_ct, min_seg = processor._get_resize_shape_V2(multi_family_subjects, is_max=False)
    # print('max_ct_shape:', max_ct)
    # print('max_seg_shape:', max_seg)
    # print('min_ct_shape:', min_ct)
    # print('min_seg_shape:', min_seg)
    bids_subjects, master_subjects = processor._get_subject_samples()
    #print(len(bids_subjects))
    #print(len(master_subjects))
    
    family =processor._get_subject_family(subject=bids_subjects[0])
    last_l =processor.master_df.loc[processor.master_df['Full_Id'] == master_subjects[0], 'Last_L'].values
    roi_object_idx = processor._get_roi_object_idx(roi_parts=[last_l, 'S1'])
    img = processor._get_cutout(family=family, roi_object_idx=roi_object_idx, return_seg=True, pad=False, pad_size=(135, 204, 139))
    #img2 = processor._get_cutout(family=family, roi_object_idx=roi_object_idx, return_seg=True, pad=True, pad_size=(135, 204, 139))
    print('cutout shape:', img.shape)
    #print('pad:', img2.shape)
    #print('expected:',(135, 204, 139))
if __name__ == "__main__":
    main()