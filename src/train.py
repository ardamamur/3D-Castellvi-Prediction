import os
import torch
import argparse
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from tensorboard import program
from utils._prepare_data import DataHandler, read_config
from modules.ResNetModule import ResNetLightning
from modules.VerSeDataModule import VerSeDataModule
from modules.DenseNetModule import DenseNet
from modules.DenseNetModule_v2 import DenseNetV2

def get_model_class(model_name:str):
    model_classes = {
        "resnet": ResNetLightning,
        "densenet": DenseNet,
        "densenet_multi_mlp": DenseNetV2,

        # Add other models here as they become available
    }
    return model_classes.get(model_name)

def main(params):
    torch.manual_seed(params.manual_seed)    
    processor = DataHandler(master_list=params.master_list,
                            dataset=params.data_root,
                            data_types=params.data_types,
                            image_types=params.img_types)
    
    verse_data_module = VerSeDataModule(opt=params, processor=processor)
    # Model selection
    ModelClass = get_model_class(params.model)
    if ModelClass is None:
        raise Exception(f"Model '{params.model}' not implemented")
    # Instantiate model
    model = ModelClass(opt=params,
                       num_classes=params.num_classes,
                       data_size=(128,86,136),
                       data_channel=1
                       ).cuda()

    # TODO: Update experiment name
    experiment = params.experiments + 'baseline_models/'  + params.model
    logger = TensorBoardLogger(experiment, default_hp_metric=False)
    
    # TODO: Add early stopping
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath=f'{experiment}/best_models/version_{logger.version}',
        filename=params.model + '-{epoch:02d}-{val_loss:.2f}',
        save_top_k=5,
        mode='min',
    )

    # Create trainer
    trainer = pl.Trainer(accelerator="gpu",
                         max_epochs=params.n_epochs,
                         check_val_every_n_epoch=1,
                         devices=params.n_devices,
                         log_every_n_steps=min(32, params.batch_size),
                         callbacks=[checkpoint_callback],
                         logger=logger)
    # Start tensorboard
    try:
        tb = start_tensorboard(params.port, experiment+"/lightning_logs") 
    except Exception as e:
        print(f"Could not start tensor board, got error {e}")

    # Start training
    torch.autograd.set_detect_anomaly(True)
    trainer.fit(model, verse_data_module)

def start_tensorboard(port, tracking_address: str):
    tb = program.TensorBoard()
    tb.configure(argv=[None, "--logdir", tracking_address, "--port", str(port)])
    url = tb.launch()
    print(f"Tensorflow listening on {url}")
    return tb



if __name__ == '__main__':

    os.environ["CUDA_VISIBLE_DEVICES"] = '3'

    if torch.cuda.is_available():
        print('Running on GPU #', os.environ["CUDA_VISIBLE_DEVICES"])
    else:
        print('Running on CPU')

    # Get Settings from config file
    parser = argparse.ArgumentParser()
    parser.add_argument('--settings', type=str, default='/data1/practical-sose23/castellvi/3D-Castellvi-Prediction/conf.yaml', help='Path to the configuration file')
    args = parser.parse_args()
    params = read_config(args.settings)
    print(params)

    main(params=params)

