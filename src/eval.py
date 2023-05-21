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
from src.modules.DenseNetModule import *

def load_model(path, params):
    checkpoint = torch.load(path)
    model = DenseNet(opt=params, num_classes=2, data_size=(128,86,136), data_channel=1)
    model.load_state_dict(checkpoint['state_dict'])  # Load weights from the 'state_dict' key
    return model


def get_predictions(model, image_paths, use_seg:bool=False):
    softmax = nn.Softmax(dim=1)
    predictions = []
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model.to(device)
    # model.eval()
    for path in image_paths:
        # Load and preprocess the image
        # TODO : if use_seg == False -> check for if "masked" not in path

        img = []
        cutouts = os.listdir(path)
        for i in cutouts:
            if use_seg:
                if "label" in i and "masked" in i:
                    print(i)
                    img = np.load(i)
                else:
                    pass
            else:
                print(i)
                if "label" in i and "masked" not in i:
                    print(i)
                    img = np.load(i)
                else:
                    pass
            

        # img = img[np.newaxis, ...]
        # img = img.to(device)

        # # Perform inference
        # with torch.no_grad():
        #     output = model(img)
        #     pred_x = softmax(output)  # , dim=1)
        #     _, pred_cls = torch.max(pred_x, 1)
        #     print(output:" , output.data)
        #     print("pred:", pred_cls)

    #return predictions

def get_test_list(test_list):
    with open(test_list, 'r') as file:
        lines = file.readlines()
        lines = [line.strip() for line in lines]  # Optional: Remove leading/trailing whitespace

    return lines


def get_img_list(test_list):
    img_paths = []
    test_subs = get_test_list(test_list)
    master_list = pd.read_excel("/data1/practical-sose23/castellvi/team_repo/3D-Castellvi-Prediction/src/dataset/VerSe_masterlist_year.xlsx")
    for i in test_subs:
        year = master_list.loc[master_list['Full_Id'] == i]['year'].values[0]
        img_path = "/data1/practical-sose23/dataset-verse" + year + "/derivatives/" + i + "/cutout"
        img_paths.append(img_path)
    return img_paths 


def main(params):
    img_paths = get_img_list(params.test_data_path)
    print(img_paths)
    ckpt_path = "/data1/practical-sose23/castellvi/team_repo/3D-Castellvi-Prediction/experiments/baseline_models/densenet/best_models/densenet-epoch=99-val_loss=2.72.ckpt"
    model = load_model(path=ckpt_path, params=params)
    get_predictions(model, img_paths, use_seg=False)

if __name__ == '__main__':

    # Get Settings
    parser = argparse.ArgumentParser()
    parser.add_argument('--settings', type=str, default='/data1/practical-sose23/castellvi/team_repo/3D-Castellvi-Prediction/settings.yaml', help='Path to the configuration file')
    args = parser.parse_args()
    params = read_config(args.settings)
    print(params)
    main(params=params)
