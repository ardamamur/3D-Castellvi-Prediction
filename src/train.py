import os
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
from utils._prepare_data import DataHandler
from utils._get_model import *
from utils.settings import parse_opts
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

    verse_data_module = VerSeDataModule(processor,
                                        subjects=(bids_subjects, master_subjects),
                                        castellvi_classes=params.castellvi_classes,
                                        pad_size=(128,86,136),
                                        use_seg=params.use_seg,
                                        use_binary_classes=params.binary_classification, 
                                        batch_size=params.batch_size)

    # Setup the data module
    verse_data_module.setup(phase='train')
    # Create the DataLoader instances
    train_dataloader = verse_data_module.train_dataloader()
    val_dataloader = verse_data_module.val_dataloader()


    is_baseline = False
    if params.model == 'resnet':
        model = ResNetLightning(params)
        is_baseline = True
    else:
        raise Exception('Not Implemented')
    
    if is_baseline:
        experiment = '/data1/practical-sose23/castellvi/team_repo/3D-Castellvi-Prediction/experiments/baseline_models' + params.model
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
    trainer.fit(model, train_dataloader=train_dataloader, val_dataloaders=val_dataloader)



if __name__ == '__main__':
    # settings
    params = parse_opts()
    os.environ["CUDA_VISIBLE_DEVICES"] = params.gpu_id # Use specific gpu -> default 3
    main(params=params)
    # getting model
