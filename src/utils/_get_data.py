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
    

    def _get_cutout(self, family:BIDS_Family, roi_object_idx: list[int], return_seg=False):
        """
        Args:
            BIDS Family, roi_object_idx (id of the desired vertebras)
        Returns:
            Cutout for the given subject.
            Cutouts generally contains the following parts of last lumbar vertebra and sacrum:
                L4, L5, L6 (last lumbar vertebra) and S1 
        """
        ct_nii = family["ct"][0].open_nii()
        seg_nii = family["msk_seg-vertsac"][0].open_nii()

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

        max_shape_ct = (0,0,0)
        max_shape_seg = (0,0,0)
        for subject in self.bids.subjects:
            if not self._is_multi_family(subject, families=multi_family_subjects):
                sub_name, exists = self._get_subject_name(subject=subject)
                if exists:
                    print(sub_name)
                    last_l = self.master_df.loc[self.master_df['Full_Id'] == sub_name, 'Last_L'].values
                    #print(last_l)
                    roi_object_idx = self._get_roi_object_idx(roi_parts=[last_l, 'S1'])
                    family = self._get_subject_family(subject=subject)
                    seg_nii = self._get_cutout(family=family, roi_object_idx=roi_object_idx, return_seg=True)
                    ct_nii = self._get_cutout(family=family, roi_object_idx=roi_object_idx, return_seg=False)
                    if ct_nii.get_array().shape > max_shape_ct:
                        max_shape_ct = ct_nii.get_array().shape
                    if seg_nii.get_array().shape > max_shape_seg:
                        max_shape_seg = seg_nii.get_array().shape

        return max_shape_ct, max_shape_seg

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
    master_list = '../../dataset/VerSe_masterlist.xlsx'
    processor = DataHandler(master_list=master_list ,dataset=dataset, data_types=data_types, image_types=image_types)
    processor._drop_missing_entries()
    families = processor._get_families()
    #print(families)
    multi_family_subjects = processor._get_subjects_with_multiple_families(families)
    max_ct, max_seg = processor._get_max_shape(multi_family_subjects)
    print('max_ct_shape:', max_ct)
    print('max_seg_shape:', max_seg)


if __name__ == "__main__":
    main()