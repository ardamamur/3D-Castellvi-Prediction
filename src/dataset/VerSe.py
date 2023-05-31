import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from utils._prepare_data import DataHandler
from monai.transforms import Compose, CenterSpatialCrop, RandRotate


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

        self.processor = processor
        self.pad_size = (128,86,136) #pad_size
        self.use_seg = opt.use_seg #use_seg
        self.training = training
        self.classification_type = opt.classification_type 
        self.transformations = self.get_transformations()
        self.test_transformations = self.get_test_transformations()
        self.records = records


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
        if record["flip"]:
            print("subject_name:", record["subject"])
            img = np.flip(img, axis=2).copy() # Flip the image along the z-axis. In other words, flip the image horizontally.

        
        img = img[np.newaxis, ...]         
        # Get the label
        labels = self._get_label_based_on_conditions(record)

        inputs = self.transformations(img) if self.training else self.test_transformations(img) # Only apply transformations if training

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

        if castellvi == '2b' or ((castellvi == '2a' and side == 'R')):
            return 1
        
        elif castellvi == '3b' or (castellvi == '3a' and side == 'R'):
            return 2
        
        else:
            return 0

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

        transformations = Compose([CenterSpatialCrop(roi_size=[128,86,136]),
                                   RandRotate(range_x = 0.2, range_y = 0.2, range_z = 0.2, prob = 0.5)
                                  ])
        return transformations
    

    def get_test_transformations(self):
        transformations = Compose([CenterSpatialCrop(roi_size=[128,86,136])])
        return transformations
    

        

