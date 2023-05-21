import sys

sys.path.append('/u/home/ank/3D-Castellvi-Prediction/bids')

import os

import torch
import argparse
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
from utils._prepare_data import DataHandler
from utils._get_model import *
from utils._prepare_data import read_config
from modules.ResNetModule import ResNetLightning
from modules.VerSeDataModule import VerSeDataModule
from modules.DenseNetModule import *
from dataset.VerSe import *

def load_model(path, params):
    checkpoint = torch.load(path)
    model = DenseNet(opt=params, num_classes=2, data_size=(128,86,136), data_channel=1)
    model.load_state_dict(checkpoint['state_dict'])  # Load weights from the 'state_dict' key
    return model


def get_predictions(model):
    softmax = nn.Softmax(dim=1)
    predictions = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    masterlist_v2 = pd.read_excel('/data1/practical-sose23/castellvi/team_repo/3D-Castellvi-Prediction/src/dataset/VerSe_masterlist_V2.xlsx', index_col=0) 
    masterlist_v2
    validation_img = []
    i = 0
    processor = DataHandler(master_list=params.master_list,
                            dataset=params.data_root,
                            data_types=params.data_types,
                            image_types=params.img_types
                        )

    processor._drop_missing_entries()
    bids_subjects, master_subjects = processor._get_subject_samples()
    bids_families = [processor._get_subject_family(subject) for subject in bids_subjects]

    val_img = []
    dict_castellvi = {}
    dict_predict = {}
    dict_conf_matrix = {'0' : [0,0] , '1a': [0,0], '1b':  [0,0], '2a' : [0,0] , '2b':  [0,0], '3a':  [0,0], '3b': [0,0], '4': [0,0]}

    for index in range(len(bids_subjects)):
        df = masterlist_v2.loc[masterlist_v2['Full_Id'] == master_subjects[index]]
        if df['Split'].values == 'test':
            bids_family = bids_families[index]
            dict_castellvi[master_subjects[index]] = str(df['Castellvi'].values[0])
            img = processor._get_cutout(family = bids_family, return_seg = False, max_shape = (128,86,136))
            img = img[np.newaxis, ...]
            img = img[np.newaxis, ...]
            img = torch.from_numpy(img)
            img = img.float() 
            img = img.to(device)

            with torch.no_grad():
                output = model(img)
                pred_x = softmax(output)  # , dim=1)
                _, pred_cls = torch.max(pred_x, 1)
                #print("pred:", pred_cls)
                pred_cls_str = str(pred_cls.item())
                dict_predict[master_subjects[index]] = pred_cls_str


    for key in dict_castellvi.keys():
        print(type(dict_castellvi[key]), type(dict_predict[key]))
        if dict_castellvi[key] != '0' and dict_predict[key] != '1':
            dict_conf_matrix[str(dict_castellvi[key])][1] += 1
        elif dict_castellvi[key] != '0' and dict_predict[key] == '1':
            dict_conf_matrix[str(dict_castellvi[key])][0] += 1
        elif dict_castellvi[key] == '0' and str(dict_predict[key]) != '0':
            dict_conf_matrix[str(dict_castellvi[key])][1] += 1
        elif dict_castellvi[key] == '0' and str(dict_predict[key]) == '0':
            dict_conf_matrix[str(dict_castellvi[key])][0] += 1

    print(dict_conf_matrix)




def main(params):
    ckpt_path = '/data1/practical-sose23/castellvi/team_repo/3D-Castellvi-Prediction/experiments/baseline_models/densenet/best_models/version_2/densenet-epoch=199-val_loss=1.85.ckpt'
    model = load_model(path=ckpt_path, params=params)
    get_predictions(model)

if __name__ == '__main__':

    # Get Settings
    parser = argparse.ArgumentParser()
    parser.add_argument('--settings', type=str, default='/data1/practical-sose23/castellvi/team_repo/3D-Castellvi-Prediction/settings.yaml', help='Path to the configuration file')
    args = parser.parse_args()
    params = read_config(args.settings)
    print(params)
    main(params=params)
    
