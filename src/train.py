import os, sys
import torch
import argparse
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from tensorboard import program
from utils._prepare_data import DataHandler
from modules.ResNetModule import ResNet
from modules.VerSeDataModule import VerSeDataModule
from modules.DenseNetModule import DenseNet
from modules.DenseNetModule_v2 import DenseNetV2
from utils.environment_settings import env_settings

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
                       data_size=(96,78,78) if (params.classification_type == "right_side" or params.classification_type == "right_side_binary") else (128,86,136),
                       data_channel=2 if params.use_seg_and_raw else 1
                       ).cuda()
    
    # TODO: Update experiment name
    experiment = params.experiments + '/baseline_models/'  + params.model
    logger = TensorBoardLogger(experiment, default_hp_metric=False)
    # TODO: Add early stopping
    checkpoint_callback = ModelCheckpoint(
        monitor='val_mcc',
        dirpath=f'{experiment}/best_models/version_{logger.version}',
        filename=params.model + '-{epoch:02d}-{val_mcc:.2f}' + '-' + str(current_fold),
        save_top_k = 3,
        mode='max',
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
                        data_size=(96,78,78) if (params.classification_type == "right_side" or params.classification_type == "right_side_binary") else (128,86,136),
                        data_channel=2 if params.use_seg_and_raw else 1
                        ).cuda()

        # TODO: Update experiment name
        experiment = params.experiments + '/baseline_models/'  + params.model
        logger = TensorBoardLogger(experiment, default_hp_metric=False)
        
        # TODO: Add early stopping
        monitor = params.val_metric
        if params.val_metric == "val_loss":
            mode = 'min'
            filename = params.model + '-{epoch:02d}-{val_loss:.2f}'
        elif params.val_metric == "val_mcc":
            filename = params.model + '-{epoch:02d}-{val_mcc:.2f}'
            mode = 'max'
        else:
            raise Exception(f"Metric '{params.val_metric}' not implemented")

        checkpoint_callback = ModelCheckpoint(
            monitor=monitor,
            dirpath=f'{experiment}/best_models/version_{logger.version}',
            filename=filename,
            save_top_k=1,
            mode=mode,
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

    if env_settings.CUDA_VISIBLE_DEVICES is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(env_settings.CUDA_VISIBLE_DEVICES)

    if env_settings.BIDS_PATH is not None:
        sys.path.append(env_settings.BIDS_PATH)

    if torch.cuda.is_available():
        print('Running on GPU #' + str(torch.cuda.current_device()))
    else:
        print('Running on CPU')

    
    parser = argparse.ArgumentParser(description='Training settings')
    parser.add_argument('--data_root', nargs='+', default=[str(os.path.join(env_settings.DATA, 'dataset-verse19')),
                                                           str(os.path.join(env_settings.DATA, 'dataset-verse20')),
                                                           str(os.path.join(env_settings.DATA, 'dataset-tri'))],
                                                           help='Path to the data root'
                                                           )
    parser.add_argument('--data_types', nargs='+', default=['rawdata', 'derivatives'], help='Data types to use (rawdata, derivatives)')
    parser.add_argument('--img_types', nargs='+', default=['ct', 'subreg', 'cortex'], help='Image types to use (ct, subreg, cortex)')
    parser.add_argument('--master_list', default= str(os.path.join(env_settings.ROOT, 'src/dataset/Castellvi_list_Final_Split.xlsx')), help='Path to the master list')
    parser.add_argument('--classification_type', default='right_side', help='Classification type (right_side, right_side_binary, both_side)')
    parser.add_argument('--castellvi_classes', nargs='+', default=['1a', '1b', '2a', '2b', '3a', '3b', '4', '0'])
    parser.add_argument('--model', default='densenet', help='Model to use (densenet and/or resnet)')
    parser.add_argument('--phase', default='train', help='Phase to run (train, test, cross_validation)')
    parser.add_argument('--scheduler', default='ReduceLROnPlateau', help='Scheduler to use (ReduceLROnPlateau, CosineAnnealingLR, CosineAnnealingWarmRestarts)')
    parser.add_argument('--optimizer', default='AdamW', help='Optimizer to use (AdamW, SGD)')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0001, help='Weight decay')
    parser.add_argument('--total_iterations', type=int, default=100, help='Total number of iterations')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--accumulate_grad_batches', type=int, default=1, help='Accumulate gradients')
    parser.add_argument('--num_workers', type=int, default=0, help='Number of workers')
    parser.add_argument('--save_intervals', type=int, default=10, help='Save model every n epochs')
    parser.add_argument('--n_epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--resume_path', default='', help='Path to the checkpoint to resume from')
    parser.add_argument('--experiments', default=env_settings.EXPERIMENTS, help='Path to the experiments folder')
    parser.add_argument('--gpu_id', default='0', help='GPU ID')
    parser.add_argument('--n_devices', type=int, default=1, help='Number of GPUs to use')
    parser.add_argument('--manual_seed', type=int, default=1, help='Random seed')
    parser.add_argument('--num_classes', type=int, default=3, help='Number of classes')
    parser.add_argument('--port', type=int, default=1923, help='Port for tensorboard')
    parser.add_argument('--val_metric', type=str, default='val_mcc', help='Validation metric to use (val_loss, val_acc, val_mcc)')


    parser.add_argument('--rotate_range', type=int, default=10)
    parser.add_argument('--shear_range', type=float, default=0.2)
    parser.add_argument('--translate_range', type=float, default=0.15)
    parser.add_argument('--scale_range', nargs='+', default=[0.9, 1.1])
    parser.add_argument('--aug_prob', type=float, default=0.5)
    parser.add_argument('--elastic_transform', action='store_true')
    parser.add_argument('--sigma_range', nargs='+', default=[5, 8])
    parser.add_argument('--magnitude_range', nargs='+', default=[100, 200])


    parser.add_argument('--use_seg', action='store_true', help='Use segmentation')
    parser.add_argument('--use_bin_seg', action='store_true', help='Use binary segmentation')
    parser.add_argument('--use_seg_and_raw', action='store_true', help='Use segmentation and raw data')
    parser.add_argument('--no_cuda', action='store_true', help='Do not use cuda')
    parser.add_argument('--weighted_sample', action='store_true', help='Use weighted sampling')
    parser.add_argument('--weighted_loss', action='store_true', help='Use weighted loss')
    parser.add_argument('--flip_all', action='store_true', help='Flip all images')
    parser.add_argument('--cross_validation', action='store_true', help='Cross validation')
    parser.add_argument('--use_zero_out', action='store_true', help='Use zero out')
    parser.add_argument('--gradual_freezing', action='store_true', help='Gradual freezing')
    parser.add_argument('--dropout_prob', type=float, default=0.0, help='Dropout probability')
    params = parser.parse_args()


    """#Get Settings from config file
    parser = argparse.ArgumentParser()
    parser.add_argument('--settings', type=str, default='/data1/practical-sose23/castellvi/3D-Castellvi-Prediction/settings.yaml', help='Path to the configuration file')
    args = parser.parse_args()
    params = read_config(args.settings)
    print(params)"""

    main(params=params)

