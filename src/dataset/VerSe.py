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
        self.binary = opt.binary_classification
        self.categories = opt.castellvi_classes
        self.castellvi_dict = {category: i for i, category in enumerate(self.categories)} 
        self.transformations = self.get_transformations()
        self.test_transformations = self.get_test_transformations()
        self.right_side = opt.right_side
        if self.right_side:
            self.no_side = ['0', '1a', '1b', '4']
        self.records = records


    def __len__(self):
        '''
        Returns the length of the dataset. If we are using the right side, then we double the length of the dataset only for non-zero labels.
        Args:
            None
        Returns:
            int: length of the dataset
        '''
        

        if self.right_side:
            zero_labels = len([rec for rec in self.records if str(rec["castellvi"]) in self.no_side])
            non_zero_labels = len(self.records) - zero_labels
            return 2 * non_zero_labels + zero_labels
        else:
            return len(self.records)


    def __getitem__(self, index):
        '''
        if self.right_side is set to True, every record contributes two images to the dataset - the original and a flipped version. 
        In other words, the dataset's length is twice the number of records. This is implemented by the __len__() method. 
        In the __getitem__(self, index) method, we use record_index = index // 2 to map the index back to a record_index for our actual records. 
        Integer division (//) by 2 ensures that for two consecutive indices in our dataset, we refer to the same record. 
        For instance, index=0 and index=1 both refer to record_index=0, index=2 and index=3 refer to record_index=1, 
        and so on. However, we don't want to provide the exact same image for two consecutive indices. That's why we use the flip variable. 
        We want one of these images to be the original and one to be flipped. 
        The expression (index % 2 == 1) is True for odd indices and False for even ones. 
        This means, for the same record, we return the original image when the dataset index is even and the flipped image when the dataset index is odd.
        Now, we add another condition. We only flip an image if its label is not zero. We achieve this with the condition if str(record["castellvi"]) != '0' and flip:. 
        Here, flip is already determined based on the index being odd or even, and the additional condition str(record["castellvi"]) != '0' ensures that only non-zero labeled images are flipped. 
        This way, zero-labeled images are never flipped, but each non-zero labeled image appears twice in the dataset: once in its original orientation and once flipped.
        Please note that this will only work if self.right_side is set to True. 
        If you want to flip non-zero labeled images even when self.right_side is set to False, 
        then you would need to adjust the __len__() and __getitem__(self, index) methods accordingly, 
        because currently, self.right_side being False implies that each record only contributes one image to the dataset (the original, unflipped one).

        
        Returns the data point at the given index. If we are using the right side, then we flip the image for non-zero labels.
        Args:
            index (int): index of the data point to return
        Returns:
            dict: dictionary containing the data point and the label

        '''

        record_index = index // 2 if self.right_side else index
        record = self.records[record_index] 

        img = self.processor._get_cutout(record, return_seg=self.use_seg, max_shape=self.pad_size) 
        img = img[np.newaxis, ...] 

        # Flip the image if we are using the right side and the index is odd. Because we are using the right side, we flip the image but if the label is 0, we don't flip it.
        flip = (index % 2 == 1) if self.right_side else False


        if flip:
            if str(record["castellvi"]) not in self.no_side: # Only flip if label is not 0
                img = np.flip(img, axis=2).copy() # Flip the image along the z-axis. In other words, flip the image horizontally.


        # Get the label
        labels = self._get_label_based_on_conditions(record, flip)

        inputs = self.transformations(img) if self.training else self.test_transformations(img) # Only apply transformations if training

        return {"target": inputs, "class": labels}


    def _get_label_based_on_conditions(self, record, flip):
        # This method is created to clean up the label generation code in getitem.
        # Replace this with the appropriate conditions and methods you have for generating labels

        if self.binary:
            return self._get_binary_label(record)
        elif self.right_side:
            return self._get_castellvi_right_side_label(record, flip=flip)
        else:
            return self._get_castellvi_multi_labels(record)

    def _get_binary_label(self, record):

        if str(record["castellvi"]) != '0':
            return 1
        else:
            return 0
    
    def _get_castellvi_label(self, record):

        castellvi = str(record["castellvi"])
        one_hot = np.zeros(len(self.categories))    
        one_hot[self.castellvi_dict[castellvi]] = 1
        return one_hot


    def _get_castellvi_right_side_label(self, record, flip):

        """
        Returns the label for the right side of the image. If the image is flipped, then the label is flipped as well.
        Args:
            record (dict): record to get the label for
            flip (bool): whether the image is flipped
        Returns:
            int: label for the right side of the image

            # Mapping original labels to the ones used in training
            original_labels = [0, 2, 3]
            training_labels = [0, 1, 2]

        """

        castellvi = str(record["castellvi"])
        side = str(record['side'])
        
        if castellvi in self.no_side:
            return 0

        if flip:
            side = 'L' if side == 'R' else 'R'

        if castellvi == '2b' or ((castellvi == '2a' and side == 'R')):
            return 1
        elif castellvi == '3b' or (castellvi == '3a' and side == 'R'):
            return 2
        
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
    

        

