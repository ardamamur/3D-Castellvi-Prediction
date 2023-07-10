from utils._prepare_data import DataHandler
from dataset.Splines import ConvexHullDataset

WORKING_DIR = "/home/daniel/Documents/Uni/practical-sose23/castellvi/3D-Castellvi-Prediction/"

dataset = [WORKING_DIR  + 'data/dataset-verse19',  WORKING_DIR + 'data/dataset-verse20', WORKING_DIR + 'data/dataset-tri']
data_types = ['rawdata',"derivatives"]
image_types = ["ct"]
master_list = WORKING_DIR + 'src/dataset/Castellvi_list_v3.xlsx'
processor = DataHandler(master_list=master_list ,dataset=dataset, data_types=data_types, image_types=image_types)

dataset = ConvexHullDataset(processor)
print(dataset[0])
