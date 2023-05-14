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
    
    def _get_subjects_with_multiple_families(families:dict):
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

        print("SUbjects with multiple families:", keys_with_value_greater_than_1)
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
        null_indexes = self.master_df.index[self.master_df["Sacrum_Seg"].isnull()].tolist()
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
            missing_subjects = self._get_missing_subjects()
            self.master_df = self.master_df.drop(missing_subjects, inplace=True)
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
        idx = []
        for i in v_idx2name:
            if i in roi_parts:
                idx.append(v_idx2name[i])
        return idx.sort()
    

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


        return ct_nii, seg_nii
        

def main():
    dataset = ['/data1/practical-sose23/dataset-verse19',  '/data1/practical-sose23/dataset-verse20']
    data_types = ['rawdata',"derivatives"]
    image_types = ["ct", "subreg", "cortex"]
    master_list = '../../dataset/VerSe_masterlist.xlsx'
    processor = DataHandler(master_list=master_list ,dataset=dataset, data_types=data_types, image_types=image_types)
    
    print(processor._get_len())

if __name__ == "__main__":
    main()