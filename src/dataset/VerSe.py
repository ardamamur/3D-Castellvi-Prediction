import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from utils._prepare_data import DataHandler
from monai.transforms import Compose, CenterSpatialCrop, RandRotate, Rand3DElastic, RandAffine
from scipy import ndimage


class VerSe(Dataset):
    def __init__(self, opt, processor:DataHandler, records, training=True) -> None:

        '''
        This class is used to create a dataset for the VerSe dataset. It is used by the VerSeDataModule class. Also, it is used to create the training, validation and test datasets.

        Args:
            processor (DataHandler): DataHandler object
            records (list): list of records to use for this dataset
            opt (dict): dictionary containing the options :
                opt.castellvi_classes (list): list of castellvi classes to use
                opt.pad_size (tuple): size to pad the image to
                opt.use_seg (bool): whether to return the segmentation mask
                opt.use_binary_classes (bool): whether to use binary classes
            training (bool): whether this is a training dataset
        
        Returns:
            None
        '''
        self.opt = opt
        self.processor = processor
        self.pad_size = (128,86,136) #pad_size
        self.use_seg = opt.use_seg #use_seg
        self.use_bin_seg = opt.use_bin_seg
        self.use_zero_out = opt.use_zero_out
        self.training = training
        self.classification_type = opt.classification_type 
        self.transformations = self.get_transformations()
        self.test_transformations = self.get_test_transformations()
        self.records = records
        self.use_bin_seg = opt.use_bin_seg
        self.use_zero_out = opt.use_zero_out


    def __len__(self):
        '''
        Args:
            None
        Returns:
            int: length of the dataset
        '''
        return len(self.records)


    def __getitem__(self, index):
        '''        
        Returns the data point at the given index. If we are using the right side, then we flip the image for non-zero labels.
        Args:
            index (int): index of the data point to return
        Returns:
            dict: dictionary containing the data point and the label

        '''

        record = self.records[index] 

        img = self.processor._get_cutout(record, return_seg=self.use_seg, max_shape=self.pad_size) 



        ###Apply zeroing out and binarizing
        if self.use_seg:
            if self.use_zero_out:
                l_idx = 25 if 25 in img else 24 if 24 in img else 23
                l_mask = img == l_idx #create a mask for values belonging to lowest L
                sac_mask = img == 26 #Sacrum is always denoted by value of 26
                lsac_mask = (l_mask + sac_mask) != 0
                lsac_mask = ndimage.binary_dilation(lsac_mask, iterations=2)
                img = img * lsac_mask

            if self.use_bin_seg:
                bin_mask = img != 0
                img = bin_mask.astype(float)
            

        elif self.use_zero_out:
            #We need the segmentation mask to create the boolean zero-out mask, TODO: Use seg-subreg mask in future for better details
            seg = self.processor._get_cutout(record, return_seg=self.use_seg, max_shape=self.pad_size) 
            l_idx = 25 if 25 in seg else 24 if 24 in seg else 23
            l_mask = seg == l_idx #create a mask for values belonging to lowest L
            sac_mask = seg == 26 #Sacrum is always denoted by value of 26
            lsac_mask = (l_mask + sac_mask) != 0
            lsac_mask = ndimage.binary_dilation(lsac_mask, iterations=2)
            img = img * lsac_mask

        
        if record["flip"]:
            # Flip 2b and 3b labels and 0 cases
            #print("subject_name:", record["subject"])
            img = np.flip(img, axis=2).copy() # Flip the image along the z-axis. In other words, flip the image horizontally.

        
        img = img[np.newaxis, ...]         
        # Get the label
        labels = self._get_label_based_on_conditions(record)

        inputs = self.transformations(img) if self.training else self.test_transformations(img) # Only apply transformations if training
        
        #print('tatget:', record["subject"], 'flip:', record['flip'],  'label:', labels)
        return {"target": inputs, "class": labels}


    def _get_label_based_on_conditions(self, record):
        # This method is created to clean up the label generation code in getitem.
        # Replace this with the appropriate conditions and methods you have for generating labels

        if self.classification_type == "binary":
            return self._get_binary_label(record)
        elif self.classification_type == "right_side":
            return self._get_castellvi_right_side_label(record)
        elif self.classification_type == "multi_class":
            return self._get_castellvi_multi_labels(record)
        elif self.classification_type == "right_side_binary":
            return self._get_right_side_binary_label(record)
        else:
            raise ValueError("Invalid classification type")

    def _get_binary_label(self, record):

        if str(record["castellvi"]) != '0':
            return 1
        else:
            return 0
        
    def _get_castellvi_right_side_label(self, record):

        """
        Returns the label for the right side of the image. If the image is flipped, then the label is flipped as well.
        Args:
            record (dict): record to get the label for
        Returns:
            int: label for the right side of the image

            # Mapping original labels to the ones used in training
            original_labels = [0, 2, 3]
            training_labels = [0, 1, 2]

        """

        castellvi = str(record["castellvi"])
        side = str(record['side'])
        flip = str(record['flip'])

        
        if castellvi == '2b' or (castellvi == '2a' and side == 'R'):
            return 1
        
        elif castellvi == '3b' or (castellvi == '3a' and side == 'R'):
            return 2
        
        elif castellvi == '4':
            if side == 'R':
                return 2
            else:
                return 1  
        else:
            return 0
        
    def _get_right_side_binary_label(self, record):
        """
        Returns a binary label for right side classification. 0 if the right side label is 0 and 1 in all other cases.
        Args:
            record (dict): record to get the label for
        """
        return 0 if self._get_castellvi_right_side_label(record) == 0 else 1

    def _get_castellvi_multi_labels(self, record):

        """
        Returns the label for the right side of the image. If the image is flipped, then the label is flipped as well.
        Args:
            record (dict): record to get the label for
            flip (bool): whether the image is flipped
        Returns:
            int: label for the right side of the image
        """
    
        castellvi = str(record["castellvi"]) # Get the castellvi class
        castellvi_class, castellvi_subclass = None, None # Initialize the class and subclass

        if len(castellvi) > 1:
            # if the class is not 0 or 4 
            castellvi_class, castellvi_subclass = castellvi[0], castellvi[1] # Get the class and subclass
            if castellvi_subclass.upper() == "A": # If the subclass is A, then set it to 0
               castellvi_subclass = 0 # Set the subclass to 0
            else: # Otherwise, set it to 1
                castellvi_subclass = castellvi_class # Set the subclass to 1
        else: # If the class is 0 or 4, then set the class to 0 and the subclass to 0
            castellvi_class = castellvi
            castellvi_subclass = 0
        # Return the class and subclass
        return [int(castellvi_class), int(castellvi_subclass)]


    def get_transformations(self):
        # TODO do that like hyper parameeter optimization
        # randmom translation and rotation
        # random scaling to some degree

        # Rand3DElastic transform in MONAI is used for data augmentation, particularly for volumetric medical images. It applies random 3D elastic deformations to images, which simulates biological variability and is often used to increase the diversity of your training dataset.

        # Here are the key parameters for Rand3DElastic:

        # prob: Probability of applying the transform. Default is 0.1.
        # sigma_range: Range of Gaussian blurring of the displacement field, where a higher value indicates more blurring. The blur is used to smooth the random displacement fields which generate the elastic deformations. This is a tuple of two float values indicating the range.
        # magnitude_range: Range of deformation magnitudes. This is a tuple of two float values indicating the range. The magnitude of the deformations applied is a key factor affecting the degree of deformation.
        # spacing: This parameter controls the spacing between the control points which are used to define the elastic deformation field. The deformation field is a grid of vectors that dictate how the image is deformed. Lower spacing values will create more control points and generally more complex deformations.
        # pad_mode: This mode determines how the edges of the input are treated during the transform. Options include "zeros" (default), "border", and "reflection".
        # mode: This mode determines how the input array is interpolated. Options include "bilinear", "nearest", and "bicubic".
        # rotate_range: Rotation range in radians. Can be a single float, or a tuple of floats specifying a different range for each axis.
        # shear_range: Shear range in radians. Can be a single float, or a tuple of floats specifying a different range for each axis.
        # translate_range: Translation range in voxels. Can be a single float, or a tuple of floats specifying a different range for each axis.
        # scale_range: Scaling range. Can be a single float, or a tuple of floats specifying a different range for each axis. A scaling factor of 1.0 does not change the size, less than 1.0 reduces the size, and greater than 1.0 increases the size.
        # The Rand3DElastic transform works by first creating a grid of vectors which represent the displacement of each pixel in the image. This grid is then blurred by a Gaussian filter (controlled by sigma_range), to generate a smoothly varying displacement field. This displacement field is then scaled by the deformation magnitude (magnitude_range), and applied to the image. This whole process results in an "elastic" deformation of the image.

        # An important thing to note when using Rand3DElastic is that the extent of the deformations should be carefully controlled. Extreme deformations may result in unrealistic images that may negatively impact the performance of your model. It's recommended to use domain knowledge (in your case, knowledge about spinal CT images) to set the parameters appropriately.

        # transformations = Compose([CenterSpatialCrop(roi_size=[128,86,136]),
        #                             # random translation
        #                            RandRotate(range_x = 0.2, range_y = 0.2, range_z = 0.2, prob = 0.5)
        #                           ])
        
        if self.opt.elastic_transform:

            transformations = Compose([
                                        CenterSpatialCrop(roi_size=[128,86,136]),
                                        Rand3DElastic(
                                            prob=0.5,
                                            sigma_range=(5, 8),
                                            magnitude_range=(100, 200),
                                            rotate_range=np.deg2rad(self.opt.rotate_range),  # Rotation range
                                            shear_range=self.opt.shear_range,  # Shear range
                                            translate_range=self.opt.translate_range,  # Translation range
                                            scale_range=(float(self.opt.scale_range[0]), float(self.opt.scale_range[1])), # Scaling range
                                            spatial_size=[128, 86, 136],
                                        )
                                    ])
        else:

            transformations = Compose([CenterSpatialCrop(roi_size=[128,86,136])],
                                      RandAffine(translate_range=self.opt.translate_range, 
                                                rotate_range=np.deg2rad(self.opt.rotate_range),
                                                scale_range=(float(self.opt.scale_range[0]),float(self.opt.scale_range[1])),
                                                prob=self.opt.aug_prob)
                                    )
            

        return transformations
    

    def get_test_transformations(self):
        transformations = Compose([CenterSpatialCrop(roi_size=[128,86,136])])
        return transformations
    

        

