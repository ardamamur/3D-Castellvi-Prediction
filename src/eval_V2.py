import os
import re
import sys
import json
import argparse
import numpy as np
import pandas as pd
import torch.nn as nn
from utils._prepare_data import DataHandler
from sklearn.metrics import f1_score
from modules.DenseNetModule import DenseNet
from modules.ResNetModule import ResNet
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from dataset.VerSe import *
from utils._get_model import *
from sklearn.metrics import ConfusionMatrixDisplay

# Append path to import custom modules
sys.path.append('/data1/practical-sose23/castellvi/castellvi_prediction/bids')


# Custom module imports
from utils._prepare_data import DataHandler
from utils._get_model import *
from modules.DenseNetModule import DenseNet
from modules.ResNetModule import ResNet
from dataset.VerSe import *



class Eval:
    def __init__(self, opt) -> None:
        self.opt = opt
        self.data_size = (128,86,136)
        self.num_classes = opt.num_classes
        self.softmax = nn.Softmax(dim=1)

    def _get_version_number(self, ckpt_path):
        version = re.search(r'version_(\d+)', ckpt_path)
        if version is None:
            raise ValueError("Invalid checkpoint path: cannot extract version number.")
        version_number = version.group(1)
        return version_number

    def get_best_model_paths(self, dir_path):
        best_model_paths = []
        for root, dirs, files in os.walk(dir_path):
            if files:  # if there are files in the directory
                # Extract validation loss from file name and choose the file with the smallest loss
                best_model_file = min(files, key=lambda f: float(re.search(r'val_loss=(\d+\.\d+)', f).group(1)))
                best_model_paths.append(os.path.join(root, best_model_file))
        return best_model_paths


    def get_label_map(self, map:str='final_class'):
        if self.opt.classification_type == "right_side":
            # Define a mapping from pairs of predictions to final predictions

            if map == 'final_class':

                prediction_mapping = {
                    ('1', '1'): '2b',
                    ('2', '2'): '3b',
                    ('0', '0'): '0',
                    ('0', '1'): '2a',
                    ('1', '0'): '2a',
                    ('0', '2'): '3a',
                    ('2', '0'): '3a',
                    ('1', '2'): '4',
                    ('2', '1'): '4',
                }

            elif map == 'side_class':

                prediction_mapping = {
                    '0': '0',
                    '2a': '2',
                    '2b': '2',
                    '3a': '3',
                    '3b': '3'
                }

            elif map == 'output_class':
                
                prediction_mapping = {
                    '2': '3',
                    '1': '2',
                    '0': '0'
                }

            return prediction_mapping
        else:
            raise NotImplementedError("Only right_side classification is supported")
                    
    def load_model(self, path, params):
        checkpoint = torch.load(path)
        if params.model == 'densenet':
            model = DenseNet(opt=params, num_classes=self.num_classes, data_size=self.data_size, data_channel=1)
        elif params.model == 'resnet' or params.model == 'pretrained_resnet':
            print('resnet')
            model = ResNet(opt=params, num_classes=self.num_classes, data_size=self.data_size, data_channel=1)

        if params.weighted_loss:
            new_state_dict = {k: v for k, v in checkpoint['state_dict'].items() if k in model.state_dict()}
            model.load_state_dict(new_state_dict, strict=False)
        else:
            model.load_state_dict(checkpoint['state_dict'])
        return model
    
    def get_test_dataset(self):
        # extract the test subjects from masterlist by usıng the Split column and if Flip is 0
        masterlist = pd.read_excel(self.opt.master_list, index_col=0)
        #masterlist = masterlist.loc[masterlist['Split'] == 'test'] # extract test subjects
        masterlist = masterlist[masterlist['Full_Id'].str.contains('|'.join(self.opt.dataset))]
        if self.opt.eval_type != 'all':
            # return the subjects if FullId contains any substring from dataset array 
            masterlist = masterlist[masterlist['Split'].isin(['test', 'val'])]
        else:
            masterlist = masterlist[masterlist['Split'].isin(['test', 'val', 'train'])]

        masterlist = masterlist.loc[masterlist['Flip'] == 0] # extract subjects with Flip = 0
        test_subjects = masterlist["Full_Id"].tolist()
        return test_subjects
    
    def get_processor(self):
        processor = DataHandler(master_list=self.opt.master_list,
                            dataset=self.opt.data_root,
                            data_types=self.opt.data_types,
                            image_types=self.opt.img_types
                        )
        return processor
    
    def get_records(self, processor, test_subjects):
        test_records = []
        verse_records = processor.verse_records
        tri_records = processor.tri_records
        records = verse_records + tri_records
        for index in range(len(records)):
            record = records[index]
            # check if any element in test subhjects contains the subject in record
            if any(record['subject'] in s for s in test_subjects):
                # check for the flip value
                if record['flip'] == 0:
                    test_records.append(record)
                else:
                    continue
            else:
                continue
        print(len(test_records))
        return test_records
    
    def get_predictions(self, model, version_number):
        # TODO : save each pred results for images
        # Move model to GPU
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        # Set model to evaluation mode
        model.eval()

        # Get processor obejct
        processor = self.get_processor()

        # Get test subjects
        test_subjects = self.get_test_dataset()

        # Get records
        test_records = self.get_records(processor, test_subjects)

        y_pred = []
        y_true = []
        eval_results = []

        # Check if eval_results file exists and load it
        eval_file_path = self.opt.experiments + "/baseline_models/" + self.opt.model + "/eval_results_verse.json"
        if os.path.exists(eval_file_path):
            with open(eval_file_path, 'r') as f:
                eval_results = json.load(f)

        # Convert eval_results into a dictionary for easy update
        eval_results_dict = {item['subject_name']: item for item in eval_results}

        for record in test_records:
            
            # Get the data
            img = processor._get_cutout(record,return_seg=self.opt.use_seg, max_shape=self.data_size)


            ###Apply zeroing out and binarizing
            if self.opt.use_seg:
                if self.opt.use_zero_out:
                    l_idx = 25 if 25 in img else 24 if 24 in img else 23
                    l_mask = img == l_idx #create a mask for values belonging to lowest L
                    sac_mask = img == 26 #Sacrum is always denoted by value of 26
                    lsac_mask = (l_mask + sac_mask) != 0
                    lsac_mask = ndimage.binary_dilation(lsac_mask, iterations=2)
                    img = img * lsac_mask

                if self.opt.use_bin_seg:
                    bin_mask = img != 0
                    img = bin_mask.astype(float)
                

            elif self.opt.use_zero_out:
                #We need the segmentation mask to create the boolean zero-out mask, TODO: Use seg-subreg mask in future for better details
                seg = self.processor._get_cutout(record, return_seg=self.use_seg, max_shape=self.pad_size) 
                l_idx = 25 if 25 in seg else 24 if 24 in seg else 23
                l_mask = seg == l_idx #create a mask for values belonging to lowest L
                sac_mask = seg == 26 #Sacrum is always denoted by value of 26
                lsac_mask = (l_mask + sac_mask) != 0
                lsac_mask = ndimage.binary_dilation(lsac_mask, iterations=2)
                img = img * lsac_mask


            flip_img = np.flip(img, axis=2).copy() # Flip the image along the z-axis. In other words, flip the image horizontally
        
            # Add new axis to the image
            img = img[np.newaxis,np.newaxis, ...]
            flip_img = flip_img[np.newaxis, np.newaxis, ...]

            
            # Convert to tensor
            img = img.astype(np.float32) 
            flip_img = flip_img.astype(np.float32) 
            img = torch.from_numpy(img)
            flip_img = torch.from_numpy(flip_img)

            # Convert to float
            img = img.float()
            flip_img = flip_img.float()

            # Move to GPU
            img = img.to(device)
            flip_img = flip_img.to(device)

            # Get the label
            label = str(record["castellvi"]) # 0, 2a, 2b, 3a, 3b, 4
            y_true.append(label)

            with torch.no_grad():
                output_1 = model(img)
                output_2 = model(flip_img)

                pred_1 = self.softmax(output_1)
                pred_2 = self.softmax(output_2)

                _, pred_cls_1 = torch.max(pred_1, 1)
                _, pred_cls_2 = torch.max(pred_2, 1)

                pred_cls_1 = str(pred_cls_1.item())
                pred_cls_2 = str(pred_cls_2.item())

                # Determine the final prediction based on the pair of predictions
                pred_cls = self.get_label_map().get((pred_cls_1, pred_cls_2))

                if pred_cls is None:
                    raise ValueError("Invalid prediction pair: ", pred_cls_1, pred_cls_2)
                
                y_pred.append(pred_cls)

            acutal_r = None
            pred_r = self.get_label_map(map='output_class')[pred_cls_1]
            pred_flip_r = self.get_label_map(map='otuput_class')[pred_cls_2]

            if record['side'] == 'R':
                if label == '4':
                    acutal_r = self.get_label_map(map='side_class')['3a']
                    actual_flip_r = self.get_label_map(map='side_class')['2a']
                elif label == '3a' or label == '2a':
                    acutal_r = self.get_label_map(map='side_class')[label]
                    actual_flip_r = self.get_label_map(map='side_class')['0']
                else:
                    acutal_r = self.get_label_map(map='side_class')[label]
                    actual_flip_r = self.get_label_map(map='side_class')[label]
            else:
                if label == '4':
                    acutal_r = self.get_label_map(map='side_class')['2a']
                    actual_flip_r = self.get_label_map(map='side_class')['3a']

                elif label == '3a' or label == '2a':
                    acutal_r = self.get_label_map(map='side_class')['0']
                    actual_flip_r = self.get_label_map(map='side_class')[label]
                else:
                    acutal_r = self.get_label_map(map='side_class')[label]     
                    actual_flip_r = self.get_label_map(map='side_class')[label]       

             # Update eval_results_dict
            subject_results = eval_results_dict.get(
                record['subject'], 
                {
                    'subject_name': record['subject'],
                    'actual': label,
                    'actual_R' : acutal_r,
                    'actual_flip_R' : actual_flip_r,
                    'pred': {},
                    'pred_R': {},
                    'pred_flip_R': {},
                }
            )
            subject_results['pred'][f'experiment_{version_number}'] = pred_cls
            subject_results['pred_R'][f'experiment_{version_number}'] = pred_r
            subject_results['pred_flip_R'][f'experiment_{version_number}'] = pred_flip_r

            if pred_cls == label:
                subject_results['correct'] = subject_results.get('correct', 0) + 1
            else:
                subject_results['incorrect'] = subject_results.get('incorrect', 0) + 1
            eval_results_dict[record['subject']] = subject_results

        # Calculate F1-score
        f1 = self.get_f1_score(y_true, y_pred)
        # Add F1-score to eval_results_dict
        for _, result in eval_results_dict.items():
            result['f1_scores'] = result.get('f1_scores', {})
            result['f1_scores'][f'experiment_{version_number}'] = f1

        # Convert eval_results_dict back into a list
        eval_results = list(eval_results_dict.values())

        with open(eval_file_path, 'w') as f:
            json.dump(eval_results, f, indent=4)

        return y_true, y_pred
        
    def get_f1_score(self, y_true, y_pred):
        f1 = f1_score(y_true, y_pred, average='weighted')
        return f1
    
    def get_confusion_matrix(self, y_true, y_pred):
        cm = confusion_matrix(y_true, y_pred)
        return cm
    
    def plot_confusion_matrix(self, cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):

        # save confusion matrix as plot
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
        disp.plot(cmap=cmap)
        plt.savefig('confusion_matrix.png')
        plt.close()


                
def main(params, ckpt_path=None):
    evaluator = Eval(opt=params)
    # Get paths to the best model
    best_model= os.listdir(ckpt_path)[0]
    best_model_path = os.path.join(ckpt_path, best_model)


    # Run evaluation for each best model

    model = evaluator.load_model(path=best_model_path, params=params)

    y_true, y_pred = evaluator.get_predictions(model, version_number=str(params.version_no))
    f1 = evaluator.get_f1_score(y_true, y_pred)
    cm = evaluator.get_confusion_matrix(y_true, y_pred)

    print('Version Number:', params.version_no)
    print("Model: ", ckpt_path)
    print("F1 score: ", f1)
    print("Confusion matrix: \n ", cm)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Evaluation settings')

    parser = argparse.ArgumentParser(description='Evaluation settings')
    parser.add_argument('--data_root', nargs='+', default=['/data1/practical-sose23/castellvi/3D-Castellvi-Prediction/data/dataset-verse19', 
                                                           '/data1/practical-sose23/castellvi/3D-Castellvi-Prediction/data/dataset-verse20', 
                                                           '/data1/practical-sose23/castellvi/3D-Castellvi-Prediction/data/dataset-tri'])
    parser.add_argument('--data_types', nargs='+', default=['rawdata', 'derivatives'])
    parser.add_argument('--img_types', nargs='+', default=['ct', 'subreg', 'cortex'])
    parser.add_argument('--master_list', default='/data1/practical-sose23/castellvi/team_repo/3D-Castellvi-Prediction/src/dataset/VerSe_masterlist_V4.xlsx')
    parser.add_argument('--classification_type', default='right_side')
    parser.add_argument('--castellvi_classes', nargs='+', default=['1a', '1b', '2a', '2b', '3a', '3b', '4', '0'])
    parser.add_argument('--model', default='densenet')
    parser.add_argument('--phase', default='train')
    parser.add_argument('--scheduler', default='ReduceLROnPlateau')
    parser.add_argument('--optimizer', default='AdamW')
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--total_iterations', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--save_intervals', type=int, default=10)
    parser.add_argument('--n_epochs', type=int, default=150)
    parser.add_argument('--resume_path', default='')
    parser.add_argument('--experiments', default='/data1/practical-sose23/castellvi/team_repo/3D-Castellvi-Prediction/experiments')
    parser.add_argument('--gpu_id', default='3')
    parser.add_argument('--n_devices', type=int, default=1)
    parser.add_argument('--manual_seed', type=int, default=1)
    parser.add_argument('--num_classes', type=int, default=3)
    parser.add_argument('--port', type=int, default=6484)
    parser.add_argument('--dataset', nargs='+', default=['verse'])
    parser.add_argument('--eval_type', type=str, default='test')

  
    parser.add_argument('--use_seg', action='store_true')
    parser.add_argument('--no_cuda', action='store_true')
    parser.add_argument('--weighted_sample', action='store_true')
    parser.add_argument('--weighted_loss', action='store_true')
    parser.add_argument('--flip_all', action='store_true')
    parser.add_argument('--version_no', type=int, default=0)
    parser.add_argument('--use_bin_seg', action='store_true')
    parser.add_argument('--use_zero_out', action='store_true')

    params = parser.parse_args()
    ckpt_path = params.experiments + '/baseline_models/' + params.model + '/best_models/version_' + str(params.version_no) 
    main(params=params, ckpt_path=ckpt_path)

