
import argparse

def parse_opts():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_root',
        default=['/data1/practical-sose23/dataset-verse19',  '/data1/practical-sose23/dataset-verse20'],
        type=list,
        help='List of root directory paths of datasets')
    
    parser.add_argument(
        '--data_types',
        default=['rawdata', 'derivatives'],
        type=list,
        help='List of data types (BIDS format)')
    
    parser.add_argument(
        '--img_types',
        default=["ct", "subreg", "cortex"],
        type=list,
        help='List of image types (e.g. ct, subreg, cortex)')
    

    parser.add_argument(
        '--master_list',
        default='../dataset/VerSe_masterlist.xlsx',
        type=str,
        help='Path of master_list file ')

    parser.add_argument(
        '--binary_classification',
        default=True,
        type=bool,
        help="Binary classification task"
    )

    parser.add_argument(
        '--castellvi_classes',
        default=['1a', '1b', '2a', '2b', '3a', '3b', '4', '0'],
        type=list,
        help="Castellvi Classes"
    )

    parser.add_argument(
        '--use_seg',
        default=False,
        type=bool,
        help="Use segmentation images for training"
    )

    parser.add_argument(
        '--phase', 
        default='train', 
        type=str, 
        help='Phase of train or test')
    
    parser.add_argument(
        '--learning_rate',  # set to 0.001 when finetune
        default=1e-3,
        type=float,
        help=
        'Initial learning rate (divided by 10 while training by lr scheduler)')
    
    parser.add_argument(
        '--weight_decay',  # set to 0.001 when finetune
        default=1e-4,
        type=float,
        help=
        'Initial weight decay')
    
    parser.add_argument(
        '--total_iterations', 
        default=100,
        type=int,
        help=
        'max epoch number')
    

    parser.add_argument(
        '--batch_size',
        default=1,
        type=int,
        help='Batch Size')
    
    parser.add_argument(
        '--save_intervals',
        default=10,
        type=int,
        help='Interation for saving model')

    parser.add_argument(
        '--n_epochs',
        default=200,
        type=int,
        help='Number of total epochs to run')
    
    parser.add_argument(
        '--resume_path',
        default='',
        type=str,
        help=
        'Path for resume model.'
    )

    parser.add_argument(
        '--pretrain_path',
        default='pretrain/resnet_50.pth',
        type=str,
        help=
        'Path for pretrained model.'
    )

    parser.add_argument(
        '--no_cuda',
        default=False,
        help='If true, cuda is not used.')
    
    parser.add_argument(
        '--gpu_id',
        type=str,
        default='3',            
        help='Gpu id')
    
    parser.add_argument(
        '--n_devices',
        type=int,
        default=1,            
        help='Gpu numbers')


    parser.add_argument(
        '--model',
        default='resnet',
        type=str,
        help='model name')
    
    
    parser.add_argument(
        '--manual_seed',
        default=1,
        type=int,
        help='Manually set random seed')
    

    args = parser.parse_args()
    args.save_folder = "../../experiments/{}/".format(args.model)
    
    return args