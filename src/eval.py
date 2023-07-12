import os
import sys
import re
import argparse
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import ndimage
from sklearn.metrics import f1_score, confusion_matrix, ConfusionMatrixDisplay, matthews_corrcoef, cohen_kappa_score
import torch
from torch import nn
from utils.environment_settings import env_settings

# Custom module imports
from utils._prepare_data import DataHandler
from utils._get_model import *
from modules.DenseNetModule import DenseNet
from modules.ResNetModule import ResNet
from dataset.VerSe import *

class Eval:
    def __init__(self, opt) -> None:
        self.opt = opt
        self.data_size = (96,78,78) if (opt.classification_type == "right_side" or opt.classification_type == "right_side_binary") else (128,86,136)
        self.num_classes = opt.num_classes
        self.softmax = nn.Softmax(dim=1)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gt = []
        self.preds = []
        self.dilated_preds = []
        self.eroded_preds = []

    # Implementation of get_label_map function broken down into three functions.
    def get_final_class_prediction_mapping(self):
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
        return prediction_mapping

    def get_side_class_prediction_mapping(self):
        prediction_mapping = {
            '0': '0',
            '2a': '2',
            '2b': '2',
            '3a': '3',
            '3b': '3'
        }
        return prediction_mapping

    def get_output_class_prediction_mapping(self):
        prediction_mapping = {
            '2': '3',
            '1': '2',
            '0': '0'
        }
        return prediction_mapping

    def get_label_map(self, map: str = 'final_class'):
        if self.opt.classification_type == "right_side":
            if map == 'final_class':
                prediction_mapping = self.get_final_class_prediction_mapping()

            elif map == 'actual_class':
                prediction_mapping = self.get_side_class_prediction_mapping()

            elif map == 'pred_class':
                prediction_mapping = self.get_output_class_prediction_mapping()

            return prediction_mapping
        else:
            raise NotImplementedError("Only right_side classification is supported")

    # Implementation of load_model function broken down into two functions.
    def initialize_model(self, params):
        if params.model == 'densenet':
            model = DenseNet(opt=params, num_classes=self.num_classes, data_size=self.data_size, data_channel=1)
        elif params.model == 'resnet' or params.model == 'resnetc':
            model = ResNet(opt=params, num_classes=self.num_classes)
        else:
            raise NotImplementedError("Model not recognized. Implement the model initialization.")
        return model

    def load_model(self, path, params):
        model = self.initialize_model(params)
        model.load_state_dict(torch.load(path)['state_dict'])
        model.to(self.device)
        model.eval()
        return model

    def get_test_subjects(self):
        # extract the test subjects from masterlist by usıng the Split column and if Flip is 0
        masterlist = pd.read_excel(self.opt.master_list, index_col=0)
        #masterlist = masterlist.loc[masterlist['Split'] == 'test'] # extract test subjects
        masterlist = masterlist[masterlist['Full_Id'].str.contains('|'.join(self.opt.dataset))]
        if self.opt.eval_type != 'all':
            # return the subjects if FullId contains any substring from dataset array 
            masterlist = masterlist[masterlist['Split'].isin(['val'])]
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
        print('lenght of the test dataset:' , len(test_records))
        return test_records
    

    def apply_softmax(self, outputs):
        # apply softmax to get probabilities
        output_probabilities = self.softmax(outputs)
        return output_probabilities
    
    def get_max(self, output_probabilities):
        # get the max probability
        max_prob, max_index = torch.max(output_probabilities, 1)
        output_class = str(max_index.item())
        output_prob = max_prob.item()
        return output_class, output_prob
    

    def get_model_output(self, model, input):
        # get the prediction from the model
        prediction = model(input)
        return prediction
    
    def get_f1_score(self, y_true, y_pred):
        f1 = f1_score(y_true, y_pred, average='weighted')
        return f1
    
    def get_confusion_matrix(self, y_true, y_pred):
        cm = confusion_matrix(y_true, y_pred)
        return cm


    def get_results_df(self):
        path = self.opt.results_df
        if os.path.exists(path):
            results_df = pd.read_csv(path, index_col=0)
            return results_df
        else:
            # should be a pivot df where we group by subject and each subject has rows for different experiments
            return None

    def dilate_image(self, img, v_index, iterations=2):
        img = ndimage.binary_dilation(img, iterations=iterations)
        img = img * v_index
        return img
    
    def erosion_image(self, img, v_index, iterations=2):
        img = ndimage.binary_erosion(img, iterations=iterations)
        img = img * v_index
        return img

    def process_input(self, processor, record):
        img = processor._get_cutout(record,return_seg=self.opt.use_seg, max_shape=self.data_size)
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
        
        flipped_img = np.flip(img, axis=2).copy()


        l_idx = 25 if 25 in img else 24 if 24 in img else 23
        s_idx = 26
        sacrum_seg = np.where(img == s_idx)
        last_l_seg = np.where(img == l_idx)
        flipped_sacrum_seg = np.where(flipped_img == s_idx)
        flipped_last_l_seg = np.where(flipped_img == l_idx)


        dilated_sacrum_seg = self.dilate_image(sacrum_seg, s_idx, iterations=2)
        dilated_last_l_seg = self.dilate_image(last_l_seg, l_idx, iterations=2)
        dilated_img = dilated_sacrum_seg + dilated_last_l_seg
        
        flipped_dilated_sacrum_seg = self.dilate_image(flipped_sacrum_seg, s_idx, iterations=2)
        flipped_dilated_last_l_seg = self.dilate_image(flipped_last_l_seg, l_idx, iterations=2)
        flipped_dilated_img = flipped_dilated_sacrum_seg + flipped_dilated_last_l_seg


        eroded_sacrum_seg = self.erosion_image(sacrum_seg, s_idx, iterations=2)
        eroded_last_l_seg = self.erosion_image(last_l_seg, l_idx, iterations=2)
        eroded_img = eroded_sacrum_seg + eroded_last_l_seg

        flipped_eroded_sacrum_seg = self.erosion_image(flipped_sacrum_seg, s_idx, iterations=2)
        flipped_eroded_last_l_seg = self.erosion_image(flipped_last_l_seg, l_idx, iterations=2)
        flipped_eroded_img = flipped_eroded_sacrum_seg + flipped_eroded_last_l_seg


        img = img[np.newaxis,np.newaxis, ...]
        flipped_img = flipped_img[np.newaxis, np.newaxis, ...]
        dilated_img = dilated_img[np.newaxis, np.newaxis, ...]
        flipped_dilated_img = flipped_dilated_img[np.newaxis, np.newaxis, ...]
        eroded_img = eroded_img[np.newaxis, np.newaxis, ...]
        flipped_eroded_img = flipped_eroded_img[np.newaxis, np.newaxis, ...]

        # Convert to tensor
        img = img.astype(np.float32) 
        flipped_img = flipped_img.astype(np.float32) 
        dilated_img = dilated_img.astype(np.float32)
        flipped_dilated_img = flipped_dilated_img.astype(np.float32)
        eroded_img = eroded_img.astype(np.float32)
        flipped_eroded_img = flipped_eroded_img.astype(np.float32)

        img = torch.from_numpy(img)
        flipped_img = torch.from_numpy(flipped_img)
        dilated_img = torch.from_numpy(dilated_img)
        flipped_dilated_img = torch.from_numpy(flipped_dilated_img)
        eroded_img = torch.from_numpy(eroded_img)
        flipped_eroded_img = torch.from_numpy(flipped_eroded_img)


        # Convert to float
        img = img.float()
        flipped_img = flipped_img.float()
        dilated_img = dilated_img.float()
        flipped_dilated_img = flipped_dilated_img.float()
        eroded_img = eroded_img.float()
        flipped_eroded_img = flipped_eroded_img.float()


        # Move to GPU
        img = img.to(self.device)
        flipped_img = flipped_img.to(self.device)
        dilated_img = dilated_img.to(self.device)
        flipped_dilated_img = flipped_dilated_img.to(self.device)
        eroded_img = eroded_img.to(self.device)
        flipped_eroded_img = flipped_eroded_img.to(self.device)


        return img, flipped_img, dilated_img, flipped_dilated_img, eroded_img, flipped_eroded_img
    

    def get_actual_labels(self, record):
        if record['side'] == 'R':
            if record['castellvi'] == '4':
                acutal_r = self.get_label_map(map='actual_class')['3a']
                actual_flip_r = self.get_label_map(map='actual_class')['2a']
            elif record['castellvi'] in ['3a', '2a']:
                acutal_r = self.get_label_map(map='actual_class')[record['castellvi']]
                actual_flip_r = self.get_label_map(map='actual_class')['0']
            else:
                acutal_r = self.get_label_map(map='actual_class')[record['castellvi']]
                actual_flip_r = self.get_label_map(map='actual_class')[record['castellvi']]
        else:
            if record['castellvi'] == '4':
                acutal_r = self.get_label_map(map='actual_class')['2a']
                actual_flip_r = self.get_label_map(map='actual_class')['3a']
            elif record['castellvi'] in ['3a', '2a']:
                acutal_r = self.get_label_map(map='actual_class')['0']
                actual_flip_r = self.get_label_map(map='actual_class')[record['castellvi']]
            else:
                acutal_r = self.get_label_map(map='actual_class')[record['castellvi']]
                actual_flip_r = self.get_label_map(map='actual_class')[record['castellvi']]
        
        return acutal_r, actual_flip_r


    def evaluate(self, path, base_path):
        model = self.load_model(path, self.opt)
        processor = self.get_processor()
        test_subjects = self.get_test_subjects()
        records = self.get_records(processor, test_subjects)

        for record in records:
            # get input image
            input_img, flipped_img, dilated_img, flipped_dilated_img, eroded_img, flipped_eroded_img  = self.process_input(processor, record)
            self.gt.append(record['castellvi'])
            actual_side, actual_flip_side = self.get_actual_labels(record)

            with torch.no_grad():
                # get the output from the model
                output = self.get_model_output(model, input_img)
                flipped_output = self.get_model_output(model, flipped_img)
                dilated_output = self.get_model_output(model, dilated_img)
                flipped_dilated_output = self.get_model_output(model, flipped_dilated_img)
                eroded_output = self.get_model_output(model, eroded_img)
                flipped_eroded_output = self.get_model_output(model, flipped_eroded_img)


                # apply softmax to get probabilities
                output_probabilities = self.apply_softmax(output)
                flipped_output_probabilities = self.apply_softmax(flipped_output)
                dilated_output_probabilities = self.apply_softmax(dilated_output)
                flipped_dilated_output_probabilities = self.apply_softmax(flipped_dilated_output)
                eroded_output_probabilities = self.apply_softmax(eroded_output)
                flipped_eroded_output_probabilities = self.apply_softmax(flipped_eroded_output)

                
                output_class, output_prob = self.get_max(output_probabilities)
                flipped_output_class, flipped_output_prob = self.get_max(flipped_output_probabilities)
                dilated_output_class, dilated_output_prob = self.get_max(dilated_output_probabilities)
                flipped_dilated_output_class, flipped_dilated_output_prob = self.get_max(flipped_dilated_output_probabilities)
                eroded_output_class, eroded_output_prob = self.get_max(eroded_output_probabilities)
                flipped_eroded_output_class, flipped_eroded_output_prob = self.get_max(flipped_eroded_output_probabilities)


                castellvi_pred = self.get_label_map(map='final_class').get((output_class, flipped_output_class))
                dilated_castellvi_pred = self.get_label_map(map='final_class').get((dilated_output_class, flipped_dilated_output_class))
                eroded_castellvi_pred = self.get_label_map(map='final_class').get((eroded_output_class, flipped_eroded_output_class))

                self.preds.append(castellvi_pred)
                self.dilated_preds.append(dilated_castellvi_pred)
                self.eroded_preds.append(eroded_castellvi_pred)

                pred_side = self.get_label_map(map='pred_class').get((output_class))
                pred_flip_side = self.get_label_map(map='pred_class').get((flipped_output_class))
                pred_dilated_side = self.get_label_map(map='pred_class').get((dilated_output_class))
                pred_flip_dilated_side = self.get_label_map(map='pred_class').get((flipped_dilated_output_class))
                pred_eroded_side = self.get_label_map(map='pred_class').get((eroded_output_class))
                pred_flip_eroded_side = self.get_label_map(map='pred_class').get((flipped_eroded_output_class))


            results = {
                'subject' : record['subject'],
                'version' : self.opt.version_no,
                'actual' : record['castellvi'],
                'pred' : castellvi_pred,
                'dilated_pred' : dilated_castellvi_pred,
                'eroded_pred' : eroded_castellvi_pred,
                'actual_right_side' : actual_side,
                'pred_right_side' : pred_side,
                'dilated_pred_right_side' : pred_dilated_side,
                'eroded_pred_right_side' : pred_eroded_side,
                'pred_prob' : output_prob,
                'dilated_pred_prob' : dilated_output_prob,
                'eroded_pred_prob' : eroded_output_prob,
                'actual_flip_right_side' : actual_flip_side,
                'pred_flip_right_side' : pred_flip_side,
                'dilated_pred_flip_right_side' : pred_flip_dilated_side,
                'eroded_pred_flip_right_side' : pred_flip_eroded_side,
                'pred_flip_prob' : flipped_output_prob,
                'dilated_pred_flip_prob' : flipped_dilated_output_prob,
                'eroded_pred_flip_prob' : flipped_eroded_output_prob
            }

            #print(results)
            # Convert dictionary to DataFrame
            # Initialize an empty DataFrame
            results_df = pd.DataFrame.from_dict(results, orient='index').T
            #print(results_df)
            results_file = base_path + '/results.csv'
            if not os.path.isfile(results_file):
                results_df.to_csv(results_file, index=False)
            else: # else it exists so append without writing the header
                results_df.to_csv(results_file, mode='a', header=False, index=False)
        return self.gt, self.preds


def main(params, ckpt_path=None, base_path=None):
    evaluator = Eval(params)
    best_model= os.listdir(ckpt_path)[0]
    best_model_path = os.path.join(ckpt_path, best_model)
    
    # Run evaluation for each best model
    gt, preds = evaluator.evaluate(path=best_model_path, base_path=base_path)
    
    # Calculate Confusion Matrix, F1, Cohen's Kappa and Matthews Correlation Coefficient
    cm = confusion_matrix(gt, preds)
    f1 = f1_score(gt, preds, average='weighted')
    ck = cohen_kappa_score(gt, preds)
    mcc = matthews_corrcoef(gt, preds)

    cm_dilated = confusion_matrix(gt, evaluator.dilated_preds)
    f1_dilated = f1_score(gt, evaluator.dilated_preds, average='weighted')
    ck_dilated = cohen_kappa_score(gt, evaluator.dilated_preds)
    mcc_dilated = matthews_corrcoef(gt, evaluator.dilated_preds)

    cm_eroded = confusion_matrix(gt, evaluator.eroded_preds)
    f1_eroded = f1_score(gt, evaluator.eroded_preds, average='weighted')
    ck_eroded = cohen_kappa_score(gt, evaluator.eroded_preds)
    mcc_eroded = matthews_corrcoef(gt, evaluator.eroded_preds)

    print('Confusion Matrix: \n', cm)
    print('F1 Score: ', f1)
    print('Cohen\'s Kappa: ', ck)
    print('Matthews Correlation Coefficient: ', mcc)

    # TODO : Save the statistics in a csv file for each version
    # check if the file exists
    metrics_file = base_path + '/metrics.csv'
    if not os.path.isfile(metrics_file):
        # Initialize an empty DataFrame
        metrics_df = pd.DataFrame(columns=['version', 'confusion_matirx', 'dilated_confusion_matirx', 'eroded_confusion_matirx', 
                                           'f1_score', 'dilated_f1_score', 'eroded_f1_score', 
                                           'cohen_kappa', 'dilated_cohen_kappa', 'eroded_cohen_kappa',
                                           'mcc', 'dilated_mcc', 'eroded_mcc'])
        
    else: # else it exists so append without writing the header
        metrics_df = pd.read_csv(metrics_file)
    
    # TODO : change confusion matrix to save as a string
    confusion_matrix = np.array2string(cm, separator=', ')
    dilated_confusion_matrix = np.array2string(cm_dilated, separator=', ')
    eroded_confusion_matrix = np.array2string(cm_eroded, separator=', ')

    # Append rows in DataFrame
    metrics_df = metrics_df.append({'version': params.version_no, 'confusion_matirx': confusion_matrix, 'dilated_confusion_matirx': dilated_confusion_matrix, 'eroded_confusion_matirx': eroded_confusion_matrix,
                                    'f1_score': f1, 'dilated_f1_score': f1_dilated, 'eroded_f1_score': f1_eroded,
                                    'cohen_kappa': ck, 'dilated_cohen_kappa': ck_dilated, 'eroded_cohen_kappa': ck_eroded,
                                    'mcc': mcc, 'dilated_mcc': mcc_dilated, 'eroded_mcc': mcc_eroded}, ignore_index=True)

    # TODO : Save the statistics in a csv file for each version
    metrics_df.to_csv(metrics_file, index=False)



if __name__ == "__main__":

    if env_settings.CUDA_VISIBLE_DEVICES is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(env_settings.CUDA_VISIBLE_DEVICES)

    if env_settings.BIDS_PATH is not None:
        sys.path.append(env_settings.BIDS_PATH)

    if torch.cuda.is_available():
        print('Running on GPU #' + str(torch.cuda.current_device()))
    else:
        print('Running on CPU')

    parser = argparse.ArgumentParser(description='Evaluation settings')

    parser = argparse.ArgumentParser(description='Evaluation settings')
    parser.add_argument('--data_root', nargs='+', default=[str(os.path.join(env_settings.DATA, 'dataset-verse19')),
                                                           str(os.path.join(env_settings.DATA, 'dataset-verse20')),
                                                           str(os.path.join(env_settings.DATA, 'dataset-tri'))])
    parser.add_argument('--data_types', nargs='+', default=['rawdata', 'derivatives'])
    parser.add_argument('--img_types', nargs='+', default=['ct', 'subreg', 'cortex'])
    parser.add_argument('--master_list', default= str(os.path.join(env_settings.ROOT, 'src/dataset/Castellvi_list_Final_Split.xlsx')))
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
    parser.add_argument('--experiments', default=env_settings.EXPERIMENTS)
    parser.add_argument('--gpu_id', default='0')
    parser.add_argument('--n_devices', type=int, default=1)
    parser.add_argument('--manual_seed', type=int, default=1)
    parser.add_argument('--num_classes', type=int, default=3)
    parser.add_argument('--port', type=int, default=2023)
    parser.add_argument('--model_type', type=str, default='')
    parser.add_argument('--dropout_prob', type=float, default=0.3)


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
    
    parser.add_argument('--version_no', type=int, default=0)
    parser.add_argument('--dataset', nargs='+', default=['verse', 'tri'])
    parser.add_argument('--eval_type', type=str, default='test')
    
    params = parser.parse_args()
    base_experiment = params.experiments + '/baseline_models/' + params.model
    ckpt_path = base_experiment + '/best_models/version_' + str(params.version_no) 
    main(params=params, ckpt_path=ckpt_path, base_path=base_experiment)
