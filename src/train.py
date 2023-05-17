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

def main(params):
    torch.manual_seed(params.manual_seed)
    # Initialize your data module
    processor = DataHandler(master_list=params.master_list,
                            dataset=params.data_root,
                            data_types=params.data_types,
                            image_types=params.img_types
                        )

    processor._drop_missing_entries()
    bids_subjects, master_subjects = processor._get_subject_samples()
    bids_families = [processor._get_subject_family(subject) for subject in bids_subjects]
    subjects = (bids_families, master_subjects)

    verse_data_module = VerSeDataModule(processor,
                                        subjects=subjects,
                                        castellvi_classes=params.castellvi_classes,
                                        pad_size=(128,86,136),
                                        use_seg=params.use_seg,
                                        use_binary_classes=params.binary_classification, 
                                        batch_size=params.batch_size,
                                        test_data_path=params.test_data_path)

    is_baseline = False
    if params.model == 'resnet':
        model = ResNetLightning(params)
        is_baseline = True
    else:
        raise Exception('Not Implemented')
    
    if is_baseline:
        experiment = params.experiments + '/baseline_models/' + params.model
    else:
        raise Exception('Not Implemented')

    # Initialize Tensorboard
    logs = experiment + '/logs/'
    logger = TensorBoardLogger(logs)

    # Define checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath=experiment + '/best_models/',
        filename=params.model + '-{epoch:02d}-{val_loss:.2f}',
        save_top_k=5,
        mode='min',
    )

    # Initialize a trainer
    trainer = pl.Trainer(accelerator="gpu", max_epochs=params.n_epochs, check_val_every_n_epoch=1, devices=params.n_devices, callbacks=[checkpoint_callback], logger=logger)

    # Train the model ⚡
    model = model.cuda()
    # Pass your data module to the trainer
    trainer.fit(model, verse_data_module)



if __name__ == '__main__':

    if torch.cuda.is_available():
        print('Running on GPU #', os.environ["CUDA_VISIBLE_DEVICES"])
    else:
        print('Running on CPU')

    # Get Settings
    parser = argparse.ArgumentParser()
    parser.add_argument('--settings', type=str, default='/u/home/mamar/3D-Castellvi-Prediction/settings.yaml', help='Path to the configuration file')
    args = parser.parse_args()
    params = read_config(args.settings)
    print(params)

    main(params=params)

