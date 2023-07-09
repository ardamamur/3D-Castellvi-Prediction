import logging
import numpy as np
import pandas as pd
import BIDS
from BIDS import BIDS_Global_info
from BIDS.centroids import load_centroids
import types
from tqdm import tqdm
import yaml
from pqdm.processes import pqdm
import logging
import os
import random
import nibabel as nib
import sys

from functools import partial

class DataHandler:
    def __init__(self, master_list: str, dataset:list, data_types:list, image_types:list) -> None:
        """
        Initialize a new object of BIDS toolbox for the given dataset ( VerSe19 and VerSe20)
        """
        self.bids = BIDS_Global_info(dataset, data_types, additional_key = image_types, verbose=True)
        self.verse_records = []
        self.tri_records = []
        master_df = pd.read_excel(master_list)
        self.master_records = master_df.dropna(subset = "Full_Id").to_dict('records')


        for record in self.master_records:
            sub, dataset, split, ce = self._split_full_id(record["Full_Id"])

            castellvi = str(record["Castellvi"])
            dataset_split = str(record['Split'])
            flip = record['Flip']        
            last_l = record["Last_L"]
            side = record["2a/3a Side"]

            if dataset == "verse":
                verse_record = self._create_verse_record(sub, split, castellvi, last_l, side, dataset_split, flip)
                self.verse_records.append(verse_record)
            elif dataset == "tri":
                tri_record = self._create_tri_record(sub, ce, castellvi, last_l, split, side, dataset_split, flip)
                self.tri_records.append(tri_record)

    

    def _split_full_id(self, full_id: str):
        decomposed_id = full_id.split("_")
        sub = decomposed_id[0]
        if sub[:4] == "sub-":
            #clean Full_Id prefixes
            sub = sub[4:]
        if sub[:5] == "verse":
            dataset = "verse"
            split = None if len(decomposed_id) == 1 else decomposed_id[1][6:]
            ce = None
        elif sub[:3] == "tri":
            dataset = "tri"
            split = None
            ce = None if len(decomposed_id) == 1 else decomposed_id[1][3:]
        else:
            raise Exception("Unrecognised dataset {}".format(full_id))
        
        return sub, dataset, split, ce

    def _create_verse_record(self, sub, split, castellvi, last_l, side, dataset_split, flip):
        record = {}

        record["dataset"] = "verse"

        record["subject"] = sub
        record["split"] = split
        record["dataset_split"] = dataset_split
        record["flip"] = flip

        raw_file = None
        seg_file = None
        ctd_file = None

        subject_container = self.bids.subjects[sub]
        query = subject_container.new_query(flatten = True)
        
        for file in query.loop_list():

            split_filter = True if split == None else file.do_filter("split", split)

            if file.get_parent() == "rawdata" and split_filter:
                raw_file = file
            elif file.do_filter("seg", "vertsac") and file.do_filter("format", "msk") and split_filter:
                seg_file = file
            if file.do_filter("format", "ctd") and file.do_filter("seg", "vertsac") and split_filter:
                ctd_file = file

        assert(raw_file != None)
        assert(seg_file != None)
        assert(ctd_file != None)

        record["raw_file"] = raw_file
        record["seg_file"] = seg_file
        record["ctd_file"] = ctd_file

        record["castellvi"] = castellvi
        record["last_l"] = last_l
        record["side"] = side

        return record

    def _create_tri_record(self, sub, ce, castellvi, last_l, split, side, dataset_split, flip):
        record = {}
        record["dataset"] = "tri"

        record["subject"] = sub
        record["ce"] = ce

        raw_file = None
        seg_file = None
        ctd_file = None

        subject_container = self.bids.subjects[sub]
        query = subject_container.new_query(flatten = True)

        for file in query.loop_list():

            if file.get_parent() == "rawdata" and file.do_filter("ce", ce):
                raw_file = file
            elif file.do_filter("seg", "vertsac") and file.do_filter("format", "msk") and file.do_filter("ce", ce):
                seg_file = file
            if file.do_filter("format", "ctd") and file.do_filter("seg", "vertsac") and file.do_filter("ce", ce):
                ctd_file = file

        assert(raw_file != None)
        assert(seg_file != None)
        assert(ctd_file != None)

        record["raw_file"] = raw_file
        record["seg_file"] = seg_file
        record["ctd_file"] = ctd_file
        record['split'] = split


        record["castellvi"] = castellvi
        record["last_l"] = last_l
        record["side"] = side
        record["dataset_split"] = dataset_split
        record["flip"] = flip

        return record
        
    
    def _compute_slice(self, seg_arr:np.ndarray, ctd:BIDS.Centroids, max_shape:tuple | None, zoom:tuple):
        """
        Args: 
            Segmentation Array, Max Shape, BIDS Centroids file
        Returns:
            Cutout slices that can be applied to the segmentation file itself or the original CT of the same family to contain at least 
            the lowest L vertebra and the image part up to the centroid of the superior vertebra and down to the S1 centroid.
            Slices are increased to max_shape dimension-wise to include as much real image data as possible.
        """

        #adjust the max_shape to work with general zooms
        max_shape = tuple([int(max_shape[i] / zoom[i]) for i in range(3)]) if max_shape else None

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

        #Replace negative values with 0
        ap_slice = slice(max(ap_slice.start, 0), max(ap_slice.stop, 0))
        lr_slice = slice(max(lr_slice.start, 0), max(lr_slice.stop, 0))
        is_slice = slice(max(is_slice.start, 0), max(is_slice.stop, 0))

        return ap_slice, lr_slice, is_slice


    def _get_cutout(self, record, return_seg, max_shape, save_dir = "/home/daniel/Documents/Uni/practical-sose23/castellvi/3D-Castellvi-Prediction/data", skip_existing = True):
        """
        Args:
            BIDS Family, return_seg (instead of ct), max_shape
        Returns:
            Cutout for the given subject.
            Cutouts generally contains the full lowest vertebra and are always padded to max_shape
    
        """
        assert(save_dir != None)

        filepath_seg = save_dir + "/cutouts/shape_{}_{}_{}/sub-{}_castellvi-cutout_{}_iso".format(max_shape[0], max_shape[1], max_shape[2], record["subject"], "seg")
        filepath_ct = save_dir + "/cutouts/shape_{}_{}_{}/sub-{}_castellvi-cutout_{}_iso".format(max_shape[0], max_shape[1], max_shape[2], record["subject"], "ct")

        if os.path.isfile(filepath_seg + ".npy") and os.path.isfile(filepath_ct + ".npy") and skip_existing:
            if return_seg:
                return(np.load(file=filepath_seg + ".npy"))
            else:
                return(np.load(file=filepath_ct + ".npy"))

        else:
            self._prepare_cutout(record, max_shape, save_dir = save_dir, skip_existing = skip_existing)
        
        if return_seg:
            return(np.load(file=filepath_seg + ".npy"))
        else:
            return(np.load(file=filepath_ct + ".npy"))
        
    def get_ct_seg_ctd_cutout(self, record, max_shape=(128, 86, 136), save_dir = "/home/daniel/Documents/Uni/practical-sose23/castellvi/3D-Castellvi-Prediction/data", skip_existing = True):
        """
        Args:
            BIDS Family, max_shape
        Returns:
            Cutout for the given subject.
            Cutouts generally contains the full lowest vertebra and are always padded to max_shape
    
        """
        assert save_dir != None

        filepath_ctd = save_dir + "/cutouts/shape_{}_{}_{}/sub-{}_castellvi-cutout_{}_iso".format(max_shape[0], max_shape[1], max_shape[2], record["subject"], "ctd")
        filepath_seg = save_dir + "/cutouts/shape_{}_{}_{}/sub-{}_castellvi-cutout_{}_iso".format(max_shape[0], max_shape[1], max_shape[2], record["subject"], "seg")
        filepath_ct = save_dir + "/cutouts/shape_{}_{}_{}/sub-{}_castellvi-cutout_{}_iso".format(max_shape[0], max_shape[1], max_shape[2], record["subject"], "ct")
        return self._np_to_nifti(np.load(file=filepath_ct + ".npy")), self._np_to_nifti(np.load(file=filepath_seg + ".npy")), load_centroids(filepath_ctd + ".json", verbose = False)

    def _prepare_cutouts(self, save_dir, max_shape=(128, 86, 136), n_jobs = 8, skip_existing = True):

        assert(save_dir != None)
        total_records = self.verse_records + self.tri_records
        seg_fun = partial(self._prepare_cutout, max_shape = max_shape, save_dir = save_dir, skip_existing = skip_existing)
        random.shuffle(total_records)
        
        #We use pqdm to apply seg_fun to all records in parallel
        results = pqdm(total_records, seg_fun, n_jobs=n_jobs)
        for result in results:
            print(result)

    def _np_to_nifti(self, np_arr):
        #Assumes that the array is in (P, I, R) orientation
        #define affine matrix to change (PIL) coordinates to (RAS) coordinates
        affine = np.array([[ 0,  0,  1,  0],
            [-1,  0,  0,  0],
            [ 0, -1,  0,  0],
            [ 0,  0,  0,  1]])
        return(BIDS.NII(nib.Nifti1Image(np_arr, affine)))
        
        

    def _prepare_cutout(self, record, max_shape, save_dir = "/home/daniel/Documents/Uni/practical-sose23/castellvi/3D-Castellvi-Prediction/data", skip_existing = True):
        """
        Args:
            record, max_shape
        """

        #Check for existing cutouts
        assert(save_dir != None)

        logging.basicConfig(filename=save_dir + "/cutouts/shape_{}_{}_{}/cutout_log.txt".format(max_shape[0], max_shape[1], max_shape[2]), level=logging.INFO)

        dir_path = save_dir + "/cutouts/shape_{}_{}_{}".format(max_shape[0], max_shape[1], max_shape[2])
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        filepath_seg = save_dir + "/cutouts/shape_{}_{}_{}/sub-{}_castellvi-cutout_{}_iso".format(max_shape[0], max_shape[1], max_shape[2], record["subject"], "seg")
        filepath_ct = save_dir + "/cutouts/shape_{}_{}_{}/sub-{}_castellvi-cutout_{}_iso".format(max_shape[0], max_shape[1], max_shape[2], record["subject"], "ct")
        filepath_ctd = save_dir + "/cutouts/shape_{}_{}_{}/sub-{}_castellvi-cutout_{}_iso".format(max_shape[0], max_shape[1], max_shape[2], record["subject"], "ctd")
        if os.path.isfile(filepath_seg + ".npy") and os.path.isfile(filepath_ct + ".npy") and os.path.isfile(filepath_ctd + ".json") and os.path.isfile(filepath_ct + ".nii.gz") and os.path.isfile(filepath_seg + ".nii.gz") and skip_existing:
            logging.info("Cutout for subject {} already exists".format(record["subject"]))
            return record["subject"]
        
        #redirect stdout to silence the function
        sys.stdout = open(os.devnull, "w")
        
        #Open files
        seg_nii = record["seg_file"].open_nii()
        ctd = record["ctd_file"].open_cdt()
            
        #Reorient to standard orientation
        seg_nii.reorient_(axcodes_to=('P', 'I', 'R'), verbose = False)
        ctd.reorient_(axcodes_to=('P', 'I', 'R'), _shape = seg_nii.shape, verbose = False)
        ctd.zoom = seg_nii.zoom

        #detach arrays to manipulate them
        seg_arr = seg_nii.get_array()

        #Compute cutouts
        ap_slice, lr_slice, is_slice = self._compute_slice(seg_arr = seg_arr, ctd = ctd, max_shape = max_shape, zoom = seg_nii.zoom)
        seg_cutout = seg_arr[ap_slice, is_slice, lr_slice]

        #Reattach to use nifty rescale function
        seg_nii.set_array_(seg_cutout)

        #Rescale to isotropic voxel spacing
        seg_nii.rescale_(voxel_spacing = (1,1,1))

        #Detach again to manipulate arrays
        seg_cutout = seg_nii.get_array()

        #Now we pad to final size
        seg_cutout = np.pad(seg_cutout, [(0, max_shape[i] - seg_cutout.shape[i]) for i in range(len(seg_cutout.shape))],"constant")
        
        #Save cutouts
        np.save(filepath_seg, arr = seg_cutout)
        self._np_to_nifti(seg_cutout).save(filepath_seg + ".nii.gz")

        #Now we do the same for the CT 
        ct_nii = record["raw_file"].open_nii()
        ct_nii.reorient_(axcodes_to=('P', 'I', 'R'), verbose = False)
        ct_arr = ct_nii.get_array()
        ct_cutout = ct_arr[ap_slice, is_slice, lr_slice]
        ct_nii.set_array_(ct_cutout)
        ct_nii.rescale_(voxel_spacing = (1,1,1))
        ct_cutout = ct_nii.get_array()
        ct_cutout = np.pad(ct_cutout, [(0, max_shape[i] - ct_cutout.shape[i]) for i in range(len(ct_cutout.shape))],"constant")
        np.save(filepath_ct, arr = ct_cutout)
        self._np_to_nifti(ct_cutout).save(filepath_ct + ".nii.gz")

        #Change ctd to match new shape
        centroids = ctd.centroids
        cutout_centroids = {key: (centroids[key][0] - ap_slice.start, centroids[key][1] - is_slice.start, centroids[key][2] - lr_slice.start) for key in centroids.keys() if key in seg_cutout}
        #Drop all centroids with any negative value
        cutout_centroids = {key: cutout_centroids[key] for key in cutout_centroids.keys() if not any([cutout_centroids[key][i] < 0 for i in range(len(cutout_centroids[key]))])}
        ctd.centroids = cutout_centroids
        ctd.rescale_(voxel_spacing = (1,1,1))

        #Save centroids
        ctd.save(filepath_ctd + ".json")

        #Revert stdout
        sys.stdout = sys.__stdout__

        logging.info("Cutout for subject {} saved".format(record["subject"]))

        return record["subject"]

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
    
    def _get_array(self, img):
        img_arr = img.get_array()
        return img_arr.astype(np.float32)

def save_list(path, list):
    # Open the file in write mode ('w')
    with open(path, 'w') as f:
        for item in list:
            # Write each item on a new line
            f.write("%s\n" % item)


def read_config(config_file):
    with open(config_file, 'r') as file:
        config_dict = yaml.safe_load(file)
    config_dict['save_folder'] = "../../experiments/{}/".format(config_dict['model'])
    config = types.SimpleNamespace(**config_dict)
    return config

def main():
    WORKING_DIR = "/home/daniel/Documents/Uni/practical-sose23/castellvi/3D-Castellvi-Prediction/"
    dataset = [WORKING_DIR  + 'data/dataset-verse19',  WORKING_DIR + 'data/dataset-verse20', WORKING_DIR + "data/dataset-tri", WORKING_DIR + "data/dataset-tri"]
    data_types = ['rawdata',"derivatives"]
    image_types = ["ct", "subreg"]
    master_list = '/home/daniel/Documents/Uni/practical-sose23/castellvi/3D-Castellvi-Prediction/src/dataset/Castellvi_list_v2.xlsx'
    processor = DataHandler(master_list=master_list ,dataset=dataset, data_types=data_types, image_types=image_types)
    #processor._get_cutout(sample,return_seg = False, max_shape=(128, 86, 136),save_dir = WORKING_DIR + "data", skip_existing=True)
    processor._prepare_cutouts(save_dir = WORKING_DIR + "data", n_jobs=8, skip_existing=True)


    
    
if __name__ == "__main__":
     main()