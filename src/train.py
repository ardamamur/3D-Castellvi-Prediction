import os
import torch
import argparse
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from tensorboard import program
from utils._prepare_data import DataHandler, read_config
from modules.ResNetModule import ResNet
from modules.VerSeDataModule import VerSeDataModule
from modules.DenseNetModule import DenseNet
from modules.DenseNetModule_v2 import DenseNetV2

def get_model_class(model_name:str):
    model_classes = {
        "resnet": ResNet,
        "pretrained_resnet": ResNet,
        "densenet": DenseNet,
        "densenet_multi_mlp": DenseNetV2,

        # Add other models here as they become available
    }
    return model_classes.get(model_name)

def run_cross_validation(params, current_fold):
    
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
    experiment = params.experiments + '/baseline_models/'  + params.model
    logger = TensorBoardLogger(experiment, default_hp_metric=False)
    # TODO: Add early stopping
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath=f'{experiment}/best_models/version_{logger.version}',
        filename=params.model + '-{epoch:02d}-{val_loss:.2f}' + '-' + str(current_fold),
        save_top_k = 1,
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
    

    return model,trainer



def main(params):
    torch.manual_seed(params.manual_seed)    
    processor = DataHandler(master_list=params.master_list,
                            dataset=params.data_root,
                            data_types=params.data_types,
                            image_types=params.img_types)
    
    verse_data_module = VerSeDataModule(opt=params, processor=processor)

    if not params.cross_validation:
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
        experiment = params.experiments + '/baseline_models/'  + params.model
        logger = TensorBoardLogger(experiment, default_hp_metric=False)
        
        # TODO: Add early stopping
        checkpoint_callback = ModelCheckpoint(
            monitor='val_loss',
            dirpath=f'{experiment}/best_models/version_{logger.version}',
            filename=params.model + '-{epoch:02d}-{val_loss:.2f}',
            save_top_k=1,
            mode='min',
        )

        # Create trainer
        trainer = pl.Trainer(accelerator="gpu",
                            max_epochs=params.n_epochs,
                            check_val_every_n_epoch=1,
                            devices=params.n_devices,
                            log_every_n_steps=min(32, params.batch_size),
                            callbacks=[checkpoint_callback],
                            accumulate_grad_batches=params.accumulate_grad_batches,
                            logger=logger)
        # Start tensorboard
        try:
            tb = start_tensorboard(params.port, experiment+"/lightning_logs") 
        except Exception as e:
            print(f"Could not start tensor board, got error {e}")

        # Start training
        torch.autograd.set_detect_anomaly(True)
        trainer.fit(model, verse_data_module)

    else:
        print("Cross validation")
        experiment = params.experiments + '/baseline_models/'  + params.model
        try:
            tb = start_tensorboard(params.port, experiment + '/lightning_logs' ) 
        except Exception as e:
            print(f"Could not start tensor board, got error {e}")
        
        train_losses = []
        val_losses = []

        
        num_folds = 5
        fold_models = []
        
        for k in range(num_folds):
            print('--------------------------------------')
            print(k)
            print('--------------------------------------')

            
            verse_data_module = VerSeDataModule(opt=params, processor=processor)
            verse_data_module.set_current_fold( fold_index = k)

            model, trainer = run_cross_validation(params , k)
            
            # Fit the model for the current fold

            trainer.fit(model, verse_data_module)


            # Perform training and validation evaluation for the current fold
            train_result = trainer.validate(model, datamodule=verse_data_module, verbose=False)
            val_result = trainer.validate(model, datamodule=verse_data_module, verbose=False)

            train_loss = train_result[0]['val_loss']
            val_loss = val_result[0]['val_loss']
            
            # Append the trained model and evaluation results to the list
            fold_models.append((model, train_loss, val_loss))
            
            # Store the evaluation results for this fold
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            
        avg_train_loss = sum(train_losses) / num_folds
        avg_val_loss = sum(val_losses) / num_folds


        print(f"Average Train Loss: {avg_train_loss:.4f}")
        print(f"Average Validation Loss: {avg_val_loss:.4f}")


def start_tensorboard(port, tracking_address: str):
    tb = program.TensorBoard()
    tb.configure(argv=[None, "--logdir", tracking_address, "--port", str(port)])
    url = tb.launch()
    print(f"Tensorflow listening on {url}")
    return tb



if __name__ == '__main__':

    if torch.cuda.is_available():
        print('Running on GPU #' + str(torch.cuda.current_device()))
    else:
        print('Running on CPU')

    
    parser = argparse.ArgumentParser(description='Training settings')
    parser.add_argument('--data_root', nargs='+', default=['/home/daniel/Documents/Uni/practical-sose23/castellvi/3D-Castellvi-Prediction/data/dataset-verse19',
                                                           '/home/daniel/Documents/Uni/practical-sose23/castellvi/3D-Castellvi-Prediction/data/dataset-verse20', 
                                                           '/home/daniel/Documents/Uni/practical-sose23/castellvi/3D-Castellvi-Prediction/data/dataset-tri'])
    parser.add_argument('--data_types', nargs='+', default=['rawdata', 'derivatives'])
    parser.add_argument('--img_types', nargs='+', default=['ct', 'subreg', 'cortex'])
    parser.add_argument('--master_list', default='/home/daniel/Documents/Uni/practical-sose23/castellvi/3D-Castellvi-Prediction/src/dataset/Castellvi_list_v2.xlsx')
    parser.add_argument('--classification_type', default='right_side')
    parser.add_argument('--castellvi_classes', nargs='+', default=['1a', '1b', '2a', '2b', '3a', '3b', '4', '0'])
    parser.add_argument('--model', default='densenet')
    parser.add_argument('--phase', default='train')
    parser.add_argument('--scheduler', default='ReduceLROnPlateau')
    parser.add_argument('--optimizer', default='AdamW')
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--total_iterations', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--accumulate_grad_batches', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--save_intervals', type=int, default=10)
    parser.add_argument('--n_epochs', type=int, default=100)
    parser.add_argument('--resume_path', default='')
    parser.add_argument('--experiments', default='/home/daniel/Documents/Uni/practical-sose23/castellvi/3D-Castellvi-Prediction/experiments')
    parser.add_argument('--gpu_id', default='0')
    parser.add_argument('--n_devices', type=int, default=1)
    parser.add_argument('--manual_seed', type=int, default=1)
    parser.add_argument('--num_classes', type=int, default=3)
    parser.add_argument('--port', type=int, default=2023)
    parser.add_argument('--model_type', type=str, default='')


    parser.add_argument('--rotate_range', type=int, default=10)
    parser.add_argument('--shear_range', type=float, default=0.2)
    parser.add_argument('--translate_range', type=float, default=0.15)
    parser.add_argument('--scale_range', nargs='+', default=[0.9, 1.1])
    parser.add_argument('--aug_prob', type=float, default=0.5)


    parser.add_argument('--use_seg', action='store_true')
    parser.add_argument('--no_cuda', action='store_true')
    parser.add_argument('--weighted_sample', action='store_true')
    parser.add_argument('--weighted_loss', action='store_true')
    parser.add_argument('--flip_all', action='store_true')
    parser.add_argument('--cross_validation', action='store_true')
    parser.add_argument('--use_bin_seg', action='store_true')
    parser.add_argument('--use_zero_out', action='store_true')
    parser.add_argument('--gradual_freezing', action='store_true')
    parser.add_argument('--elastic_transform', action='store_true')
    params = parser.parse_args()


    """#Get Settings from config file
    parser = argparse.ArgumentParser()
    parser.add_argument('--settings', type=str, default='/data1/practical-sose23/castellvi/3D-Castellvi-Prediction/settings.yaml', help='Path to the configuration file')
    args = parser.parse_args()
    params = read_config(args.settings)
    print(params)"""

    main(params=params)

