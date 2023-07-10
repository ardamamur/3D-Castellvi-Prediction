import torch
from torch.utils.data import Dataset
from utils._prepare_data import DataHandler
import numpy as np
from scipy import interpolate
from BIDS import Centroids
from scipy.spatial import ConvexHull

class Splines(Dataset):
    def __init__(self, processor:DataHandler):
        self.processor = processor
        self.records = self.processor.verse_records + self.processor.tri_records

    def __len__(self):
        '''
        Args:
            None
        Returns:
            int: length of the dataset
        '''
        return len(self.records)

    def __getitem__(self, idx):
        record = self.records[idx]

        # Get centroid file from processor
        _, _, centroid = self.processor.get_ct_seg_ctd_cutout(record)

        #TODO: Apply transformation of rigid registration of L4 to centroids

        # Get spline interpolation
        try: 
            spline_points, spline_1stderivative = self.fit_spline(centroid, max_dim=128)
        except:
            print(f"Failed to fit spline for {record['subject']}")
            raise Exception

        # Set spline to start at 0,0,0
        spline_points = spline_points - spline_points[0]

        # Flip spline horizontally (3rd dim) if necessary
        if record['flip']:
            spline_points[:, 2] = -spline_points[:, 2]
        
        # Get label
        label = _get_castellvi_right_side_label(record)

        # Convert to tensor
        spline_points = torch.tensor(spline_points, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.long)

        return spline_points, label



    def fit_spline(self, centroids: Centroids, max_dim: int, smoothness: int=10) -> tuple[np.ndarray, np.ndarray]:
        """Makes a spline interpolation through the pointset and calculates the first derivative of the curve.

        Args:
            centroids: Given Centroids
            max_dim: int(max(img.shape))

        Returns:
            spline_points: np.array, spline_1stderivative: np.array
        """
        centroids_coords = list(centroids.sort().values())
        centroids_coords = np.asarray(centroids_coords)
        x_sample = centroids_coords[:, 0]
        y_sample = centroids_coords[:, 1]
        z_sample = centroids_coords[:, 2]
        tck, u = interpolate.splprep([x_sample,y_sample,z_sample], k=2, s=smoothness)
        u_fine = np.linspace(0, 1, max_dim)
        x_fine, y_fine, z_fine = interpolate.splev(u_fine, tck)
        xp_fine, yp_fine, zp_fine = interpolate.splev(u_fine, tck, der=1)
        #xp_fine = np.diff(x_fine)/np.diff(np.linspace(0, max(y_sample)-min(y_sample), max_dim))
        #xp_fine = np.concatenate((xp_fine, np.asarray([xp_fine[-1]])))
        #yp_fine = np.diff(y_fine)/np.diff(np.linspace(0, max(y_sample)-min(y_sample), max_dim))
        #yp_fine = np.concatenate((yp_fine, np.asarray([yp_fine[-1]])))
        #zp_fine = np.diff(z_fine)/np.diff(np.linspace(0, max(y_sample)-min(y_sample), max_dim))
        #zp_fine = np.concatenate((zp_fine, np.asarray([zp_fine[-1]])))#attach the last element missing after diff
        #if centroids.shape[0]>3:
        #     min_tilt_point, max_tilt_point = get_max_tilt(np.asarray([x_fine, y_fine, z_fine]), np.asarray([xp_fine, yp_fine, zp_fine]))
        return np.asarray(list(zip(x_fine, y_fine, z_fine))), np.asarray(list(zip(xp_fine, yp_fine, zp_fine)))

class ConvexHullDataset(Dataset):
    def __init__(self, processor:DataHandler):
        self.processor = processor
        self.records = processor.verse_records + processor.tri_records

    def __len__(self):
        '''
        Args:
            None
        Returns:
            int: length of the dataset
        '''
        return len(self.records)
    
    def __getitem__(self, idx):
        record = self.records[idx]

        # Get centroid file from processor
        _, seg_nii, _ = self.processor.get_ct_seg_ctd_cutout(record)

        seg = seg_nii.get_array()

        last_L = 25 if 25 in seg else 24 if 24 in seg else 23 if 23 in seg else None
        assert(last_L is not None, "Last L not found for subject {}".format(record['subject']))

        sac = 26 if 26 in seg else None
        assert(sac is not None, "Sacrum not found for subject {}".format(record['subject']))

        last_L_mask = seg == last_L

        #Convert mask to point set
        last_L_points = np.argwhere(last_L_mask)
        sac_points = np.argwhere(seg == sac)

        # Get convex hull
        last_L_hull = ConvexHull(last_L_points)
        sac_hull = ConvexHull(sac_points)

        # Get label
        label = _get_castellvi_right_side_label(record)

        # Return convex hulls as dict and label
        return {'last_L': last_L_hull, 'sac': sac_hull}, label



def _get_castellvi_right_side_label(record):

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