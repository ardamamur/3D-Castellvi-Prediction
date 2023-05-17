import logging
import numpy as np
import pandas as pd
import BIDS
from BIDS import BIDS_Global_info, BIDS_Family, NII
from BIDS.snapshot2D import create_snapshot,Snapshot_Frame,Visualization_Type,Image_Modes
from pathlib import Path
import types
from tqdm import tqdm
import yaml
from pqdm.threads import pqdm

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
    
    def _compute_slice(self, seg_arr:np.ndarray, ctd:BIDS.Centroids, max_shape:tuple | None):
        """
        Args: 
            Segmentation Array, Max Shape, BIDS Centroids file
        Returns:
            Cutout slices that can be applied to the segmentation file itself or the original CT of the same family to contain at least 
            the lowest L vertebra and the image part up to the centroid of the superior vertebra and down to the S1 centroid.
            Slices are increased to max_shape dimension-wise to include as much real image data as possible.
        """

        lowest_L_idx = 25 if 25 in ctd else 24 if 24 in ctd else 23 if 23 in ctd else None
        assert(lowest_L_idx != None)

        lowest_L_mask = np.where(seg_arr == lowest_L_idx)

        #for the anterior and posterior bounds as well as the left and right bounds we just use the indices of the lowest L which we get from the mask
        ap_slice = slice(lowest_L_mask[0].min(), lowest_L_mask[0].max())
        lr_slice = slice(lowest_L_mask[2].min(), lowest_L_mask[2].max())

        #for the superior and inferior bound we use the centroids of the previous L vertebra and the S1 vertebra if available 
        prev_L_idx = lowest_L_idx - 1
        sac_idx = 26

        is_slice = slice(lowest_L_mask[1].min() if not prev_L_idx in ctd.centroids.keys() else int(ctd.centroids[prev_L_idx][1]),
                        lowest_L_mask[1].max() if not sac_idx in ctd.centroids.keys() else int(ctd.centroids[sac_idx][1]))

        #Now we adapt the slices to achieve the target_shape
        if max_shape:
            ap_increase = max(max_shape[0] - (ap_slice.stop - ap_slice.start), 0)
            lr_increase = max(max_shape[2] - (lr_slice.stop - lr_slice.start), 0)
            is_increase = max(max_shape[1] - (is_slice.stop - is_slice.start), 0)

            ap_slice = slice(ap_slice.start - int(ap_increase / 2), ap_slice.stop + (ap_increase - int(ap_increase / 2)))
            lr_slice = slice(lr_slice.start - int(lr_increase / 2), lr_slice.stop + (lr_increase - int(lr_increase / 2)))
            is_slice = slice(is_slice.start - int(is_increase / 2), is_slice.stop + (is_increase - int(is_increase / 2)))

        if max_shape:
            assert(ap_slice.stop-ap_slice.start == max_shape[0])
            assert(lr_slice.stop-lr_slice.start == max_shape[2])
            assert(is_slice.stop-is_slice.start == max_shape[1])

        return ap_slice, lr_slice, is_slice


    def _get_cutout(self, family:BIDS_Family, return_seg=False, max_shape=(135, 181, 126)) -> NII:
        """
        Args:
            BIDS Family, return_seg (instead of ct), max_shape
        Returns:
            Cutout for the given subject.
            Cutouts generally contains the full lowest vertebra and are always padded to max_shape
    
        """
        seg_nii = family["msk_seg-vertsac"][0].open_nii()
        ctd = family["ctd_seg-vertsac"][0].open_cdt()
        ctd.zoom = seg_nii.zoom

        if not return_seg:
            ct_nii = family["ct"][0].open_nii()

        seg_nii.reorient_(axcodes_to=('P', 'I', 'R'), verbose = False)
        ctd.reorient_(axcodes_to=('P', 'I', 'R'), _shape = seg_nii.shape, verbose = False)

        if not return_seg:
            ct_nii.reorient_(axcodes_to=('P', 'I', 'R'), verbose = False)

        
        #naive pre-cropping around centroid to decrease size that needs to be resampled
        # lowest_L_idx = 25 if 25 in ctd else 24 if 24 in ctd else 23 if 23 in ctd else None
        # assert(lowest_L_idx != None)
        # lowest_L_ctd = ctd[lowest_L_idx]
        
        # #save 10 cm in each direction from centroid, accounting for zoom
        # slices = [slice(int((lowest_L_ctd[i] - 100)/seg_nii.zoom[i]) , int((lowest_L_ctd[i] + 100)/seg_nii.zoom[i])) for i in range(3)]
        # seg_nii.set_array_(seg_nii.get_array()[slices[0], slices[1], slices[2]])

        # if not return_seg:
        #     ct_nii.set_array(ct_nii.get_array()[slices[0], slices[1], slices[2]])

        seg_nii.rescale_(voxel_spacing = (1,1,1))
        ctd.rescale_(voxel_spacing = (1,1,1))

        if not return_seg:
            ct_nii.rescale_(voxel_spacing = (1,1,1))


        seg_arr = seg_nii.get_array()

        ap_slice, lr_slice, is_slice = self._compute_slice(seg_arr = seg_arr, ctd = ctd, max_shape = max_shape)

        if return_seg:
            seg_cutout = seg_arr[ap_slice, is_slice, lr_slice]
            #We still need to pad in case the slice exceeded the image bounds i.e. no image data is available for the entire range
            seg_cutout = np.pad(seg_cutout, [(0, max_shape[i] - seg_cutout.shape[i]) for i in range(len(seg_cutout.shape))],"constant")
            return seg_nii.set_array(seg_cutout)
        else:
            ct_arr = ct_nii.get_array()
            ct_cutout = ct_arr[ap_slice, is_slice, lr_slice]
            #We still need to pad in case the slice exceeded the image bounds i.e. no image data is available for the entire range
            ct_cutout = np.pad(ct_cutout, [(0, max_shape[i] - ct_cutout.shape[i]) for i in range(len(ct_cutout.shape))],"constant")
            return ct_nii.set_array(ct_cutout)

    
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
        
    def _max_shape_job(self, family):
        """
        Args:
            list of families to handle.
        Returns:
            The maximum shape of the cutouts over the list
        """

        max_shape = [0,0,0]

        seg_nii = family["msk_seg-vertsac"][0].open_nii()
        ctd = family["ctd_seg-vertsac"][0].open_cdt()
        ctd.zoom = seg_nii.zoom

        seg_arr = seg_nii.reorient(axcodes_to=('P', 'I', 'R'), verbose = False).get_array()
        ctd = ctd.reorient(axcodes_to=('P', 'I', 'R'), _shape = seg_nii.shape, verbose = False)

        seg_arr = seg_nii.get_array()
            
        lowest_L_idx = 25 if 25 in ctd else 24 if 24 in ctd else 23 if 23 in ctd else None
        assert(lowest_L_idx != None)

        lowest_L_mask = np.where(seg_arr == lowest_L_idx)

        ap_size = (lowest_L_mask[0].max() - lowest_L_mask[0].min()) * seg_nii.zoom[0]
        lr_size = (lowest_L_mask[2].max() - lowest_L_mask[2].min()) * seg_nii.zoom[2]

        prev_L_idx = lowest_L_idx - 1
        sac_idx = 26

        is_size = 0 if not (prev_L_idx in ctd.centroids.keys() and sac_idx in ctd.centroids.keys()) else (ctd.centroids[sac_idx][1] - ctd.centroids[prev_L_idx][1]) * seg_nii.zoom[1]

        return max(max_shape, [ap_size, is_size, lr_size])

        

    def _get_max_shape(self, multi_family_subjects, n_jobs = 8):
        """
        Args:
            subject name
        Return:
            the maximum shape within the subjects

        Be sure to drop missing subjects before running this function
        """

        families = []

        for subject in tqdm(self.bids.subjects):
            if not self._is_multi_family(subject, families=multi_family_subjects):
                sub_name, exists = self._get_subject_name(subject=subject)
                if exists:
                    family = self._get_subject_family(subject=subject)
                    families.append(family)
                    
        max_shapes = pqdm(families, self._max_shape_job, n_jobs = n_jobs)
        max_shapes = [[0,0,0] if type(shape) != list else shape for shape in max_shapes]
        max_shapes = np.array(max_shapes)

        return max_shapes.max(axis = 0)
    

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
    


def read_config(config_file):
    with open(config_file, 'r') as file:
        config_dict = yaml.safe_load(file)
    config_dict['save_folder'] = "../../experiments/{}/".format(config_dict['model'])
    config = types.SimpleNamespace(**config_dict)
    return config

def main():
    WORKING_DIR = "/data1/practical-sose23/"
    dataset = [WORKING_DIR  + 'dataset-verse19',  WORKING_DIR + 'dataset-verse20']
    data_types = ['rawdata',"derivatives"]
    image_types = ["ct", "subreg", "cortex"]
    master_list = WORKING_DIR + 'castellvi/3D-Castellvi-Prediction/src/dataset/VerSe_masterlist.xlsx'
    processor = DataHandler(master_list=master_list ,dataset=dataset, data_types=data_types, image_types=image_types)
    processor._drop_missing_entries()
    families = processor._get_families()
    print(families)
    #multi_family_subjects = processor._get_subjects_with_multiple_families(families)
    #max_shape = processor._get_max_shape(multi_family_subjects)
    #print(max_shape)
    # max_ct, max_seg = processor._get_resize_shape_V2(multi_family_subjects, is_max=True)
    # min_ct, min_seg = processor._get_resize_shape_V2(multi_family_subjects, is_max=False)
    # print('max_ct_shape:', max_ct)
    # print('max_seg_shape:', max_seg)
    # print('min_ct_shape:', min_ct)
    # print('min_seg_shape:', min_seg)
    # bids_subjects, master_subjects = processor._get_subject_samples()
    # bids_families = [processor._get_subject_family(subject) for subject in bids_subjects]
    # for family in tqdm(bids_families):
    #     processor._get_cutout(family = family, return_seg = False, max_shape = (128,86,136))
    

    
    
if __name__ == "__main__":
    main()